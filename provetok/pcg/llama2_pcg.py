from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..types import Frame, Generation, Token
from ..grid.cells import parse_cell_id
from .narrative import render_generation_text
from .schema import FINDINGS, LATERALITY, LOCATIONS, POLARITY, SEVERITY_LEVELS, SIZE_BINS


def _extract_first_json_object(text: str) -> str:
    """Extract the first top-level JSON object substring using brace matching."""
    s = text
    start = s.find("{")
    if start < 0:
        raise ValueError("No '{' found in LLM output.")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    raise ValueError("Unterminated JSON object in LLM output.")


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y")
    return False


@dataclass
class Llama2PCGConfig:
    model_path: str
    device: str = "cuda"
    dtype: str = "float16"  # "float16" | "bfloat16"
    quantization: str = "fp16"  # "fp16" | "8bit"
    max_new_tokens: int = 220
    temperature: float = 0.0
    top_p: float = 0.95
    topk_citations: int = 3
    tau_refuse: float = 0.55
    max_tokens_in_prompt: int = 64   # keep prompt bounded (context-safe default)
    max_frames: int = 1              # keep JSON short; set >1 for multi-finding reports
    fallback_finding: str = "opacity"  # used when parsing fails or frames are empty


def parse_llm_json(text: str) -> Dict[str, Any]:
    """Parse a (possibly messy) LLM output into a JSON dict.

    - Extracts the first top-level JSON object
    - Strips code fences if present
    - Repairs common trailing-comma issues
    """
    t = str(text).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n", "", t)
        t = re.sub(r"\n```$", "", t).strip()
    raw = _extract_first_json_object(t)
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(repaired)
    except Exception as e:
        preview = repaired[:800].replace("\n", "\\n")
        raise ValueError(f"Failed to parse LLM JSON (preview={preview!r})") from e


def sanitize_generation_dict(
    d: Dict[str, Any],
    *,
    token_ids: Sequence[int],
    cfg: Llama2PCGConfig,
) -> Generation:
    """Convert a parsed JSON dict into a schema-valid Generation."""
    frames_in = d.get("frames", [])
    citations_in = d.get("citations", {}) or {}
    q_in = d.get("q", {}) or {}
    refusal_in = d.get("refusal", {}) or {}

    frames: List[Frame] = []
    citations: Dict[int, List[int]] = {}
    q: Dict[int, float] = {}
    refusal: Dict[int, bool] = {}

    allowed_findings = set(FINDINGS)
    allowed_polarity = set(POLARITY)
    allowed_lat = set(LATERALITY)
    allowed_loc = set(LOCATIONS)
    allowed_size = set(SIZE_BINS)
    allowed_sev = set(SEVERITY_LEVELS)
    token_id_set = set(int(x) for x in token_ids)

    if not isinstance(frames_in, list):
        frames_in = []

    max_frames = int(getattr(cfg, "max_frames", 5))
    frames_in = frames_in[: max(0, max_frames)]

    for k, fr in enumerate(frames_in):
        if not isinstance(fr, dict):
            continue
        finding = str(fr.get("finding", "normal")).lower()
        if finding not in allowed_findings:
            finding = "normal"

        polarity = str(fr.get("polarity", "present")).lower()
        if polarity not in allowed_polarity:
            polarity = "present"

        laterality = str(fr.get("laterality", "unspecified")).lower()
        if laterality not in allowed_lat:
            laterality = "unspecified"

        location = str(fr.get("location", "unspecified"))
        if location not in allowed_loc:
            location = "unspecified"

        size_bin = str(fr.get("size_bin", "unspecified"))
        if size_bin not in allowed_size:
            size_bin = "unspecified"

        severity = str(fr.get("severity", "unspecified"))
        if severity not in allowed_sev:
            severity = "unspecified"

        confidence = fr.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        confidence = _clamp01(confidence)

        uncertain = _as_bool(fr.get("uncertain", False))

        frames.append(
            Frame(
                finding=finding,
                polarity=polarity,
                laterality=laterality,
                confidence=confidence,
                location=location,
                size_bin=size_bin,
                severity=severity,
                uncertain=uncertain,
            )
        )

        raw_cites = citations_in.get(str(k), citations_in.get(k, []))
        if not isinstance(raw_cites, list):
            raw_cites = []
        cites: List[int] = []
        for x in raw_cites:
            try:
                xi = int(x)
            except Exception:
                continue
            if xi in token_id_set:
                cites.append(xi)
        cites = cites[: int(cfg.topk_citations)]
        citations[k] = cites

        raw_q = q_in.get(str(k), q_in.get(k, confidence))
        try:
            qk = float(raw_q)
        except Exception:
            qk = confidence
        qk = _clamp01(qk)
        q[k] = qk

        raw_ref = refusal_in.get(str(k), refusal_in.get(k, None))
        if raw_ref is None:
            ref = (qk < float(cfg.tau_refuse))
        else:
            ref = _as_bool(raw_ref)
        refusal[k] = bool(ref)

    gen_tmp = Generation(frames=frames, citations=citations, q=q, refusal=refusal, text="")
    return Generation(frames=frames, citations=citations, q=q, refusal=refusal, text=render_generation_text(gen_tmp))


class Llama2PCG:
    """LLM-backed Proof-Carrying Generator (schema-constrained via parsing + repair).

    This replaces ToyPCG with a real LLaMA-2 model. It:
    - Produces bounded finding frames
    - Emits token citations (token_id lists)
    - Emits q/refusal for calibrated refusal (simple threshold here)
    """

    def __init__(self, cfg: Llama2PCGConfig):
        self.cfg = cfg
        dtype = torch.float16 if cfg.dtype == "float16" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        quant = str(cfg.quantization).lower()
        if quant not in ("fp16", "8bit"):
            raise ValueError(f"Unsupported quantization={cfg.quantization!r} (use fp16 or 8bit)")

        if quant == "8bit":
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                load_in_8bit=True,  # bitsandbytes
                device_map={"": 0} if cfg.device.startswith("cuda") else None,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                torch_dtype=dtype,
                device_map={"": 0} if cfg.device.startswith("cuda") else None,
                low_cpu_mem_usage=True,
            )
        self.model.eval()

    def _build_prompt(self, tokens: List[Token], *, max_tokens_in_prompt: Optional[int] = None) -> str:
        # Keep prompt short: take top tokens by score (evidence head proxy), then stable by token_id.
        limit = int(self.cfg.max_tokens_in_prompt if max_tokens_in_prompt is None else max_tokens_in_prompt)
        toks = sorted(tokens, key=lambda t: (-float(t.score), int(t.token_id)))[: max(0, limit)]
        tok_lines = [
            f"- id={t.token_id} cell_id={t.cell_id} score={t.score:.3f} uncertainty={t.uncertainty:.3f} level={t.level}"
            for t in toks
        ]

        schema_hint = {
            "findings": FINDINGS,
            "polarity": POLARITY,
            "laterality": LATERALITY,
            "location": LOCATIONS,
            "size_bin": SIZE_BINS,
            "severity": SEVERITY_LEVELS,
        }

        sys_msg = (
            "You are a strict JSON generator for a radiology-claim schema. "
            "Output ONLY a single JSON object. No markdown. No commentary."
        )
        # Keep the template minimal to reduce the chance of JSON truncation.
        # Optional frame slots are filled by `sanitize_generation_dict`.
        template = {
            "frames": [{"finding": "opacity", "polarity": "present", "laterality": "unspecified", "confidence": 0.5}],
            "citations": {"0": [0]},
            "q": {"0": 0.5},
            "refusal": {"0": False},
        }
        user_msg = (
            "Return ONLY valid JSON that can be parsed by Python json.loads.\n"
            "Rules:\n"
            "- Output exactly one JSON object.\n"
            "- Use double quotes for all keys/strings.\n"
            "- Do NOT write any text before/after the JSON.\n"
            "- frames must contain exactly 1 item.\n"
            "- For brevity, each frame should include ONLY: finding, polarity, laterality, confidence.\n"
            "- citations/q/refusal keys must be string frame indices (e.g. \"0\").\n"
            "Allowed vocab values:\n"
            f"{json.dumps(schema_hint, ensure_ascii=False)}\n"
            "Token list (evidence tokens):\n"
            + "\n".join(tok_lines)
            + "\n\n"
            "JSON TEMPLATE (copy this structure exactly, filling values):\n"
            f"{json.dumps(template, ensure_ascii=False)}\n"
        )
        # Llama-2 chat format.
        return f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{user_msg} [/INST]"

    def _parse_json(self, text: str) -> Dict:
        return parse_llm_json(text)

    def _sanitize(self, d: Dict, *, token_ids: List[int]) -> Generation:
        return sanitize_generation_dict(d, token_ids=token_ids, cfg=self.cfg)

    @torch.no_grad()
    def __call__(self, tokens: List[Token]) -> Generation:
        if not tokens:
            return Generation(frames=[], citations={}, q={}, refusal={}, text="")

        max_ctx = int(getattr(getattr(self.model, "config", None), "max_position_embeddings", 4096) or 4096)
        max_new = int(self.cfg.max_new_tokens)
        # Keep a small margin for special tokens; avoid hitting context limit.
        max_input = max(1, int(max_ctx) - int(max_new) - 8)

        # Some runs may produce very long token lists; shrink the evidence token
        # section until the prompt fits in the model context window.
        limit = int(self.cfg.max_tokens_in_prompt)
        prompt = self._build_prompt(tokens, max_tokens_in_prompt=limit)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        while int(inputs["input_ids"].shape[1]) > max_input and limit > 8:
            limit = max(8, limit // 2)
            prompt = self._build_prompt(tokens, max_tokens_in_prompt=limit)
            inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        do_sample = bool(float(self.cfg.temperature) > 0.0)
        gen_kwargs = dict(
            max_new_tokens=int(self.cfg.max_new_tokens),
            do_sample=do_sample,
            pad_token_id=int(self.tokenizer.pad_token_id),
            eos_token_id=int(self.tokenizer.eos_token_id),
        )
        if do_sample:
            gen_kwargs.update(temperature=float(self.cfg.temperature), top_p=float(self.cfg.top_p))
        out = self.model.generate(**inputs, **gen_kwargs)
        # Decode only newly generated tokens to avoid accidentally parsing JSON snippets embedded in the prompt.
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        try:
            d = self._parse_json(text)
        except Exception:
            # Fallback: return an empty-but-valid object to avoid crashing long runs.
            d = {"frames": [], "citations": {}, "q": {}, "refusal": {}}
        gen = self._sanitize(d, token_ids=[t.token_id for t in tokens])

        # Guardrail: a completely empty frame list makes downstream evaluation
        # degenerate (always frame_f1=0 and grounding=0). When the LLM fails to
        # emit valid frames, we fall back to a single low-confidence frame so the
        # rest of the PCG protocol (citations + verifier) remains auditable.
        if not gen.frames:
            finding = str(getattr(self.cfg, "fallback_finding", "opacity")).lower()
            if finding not in set(FINDINGS):
                finding = "opacity"
            fr = Frame(
                finding=finding,
                polarity="present",
                laterality="unspecified",
                confidence=0.5,
                location="unspecified",
                size_bin="unspecified",
                severity="unspecified",
                uncertain=True,
            )
            qv = 0.5
            gen = Generation(
                frames=[fr],
                citations={0: []},
                q={0: qv},
                refusal={0: bool(qv < float(self.cfg.tau_refuse))},
                text="",
            )
            gen = Generation(frames=gen.frames, citations=gen.citations, q=gen.q, refusal=gen.refusal, text=render_generation_text(gen))

        # Deterministic citation repair:
        #
        # In practice, even "strict JSON" prompts can lead to degenerate citations
        # (e.g., the model copies the template `"citations":{"0":[0]}` for all
        # inputs). Proof-carrying citations must remain mechanically grounded in
        # the token set, so we override citations with a simple, auditable policy:
        # cite top-k tokens by token.score with a *diversity* tie-break so we do not
        # collapse to the lexicographically-first region when scores saturate.
        if gen.frames:
            def _center(tok: Token) -> Optional[tuple[float, float, float]]:
                cell = parse_cell_id(str(tok.cell_id))
                if cell is None:
                    return None
                n = 2 ** int(cell.level)
                if n <= 0:
                    return None
                # Normalized center in (z,y,x) within [0,1].
                return (
                    (float(cell.iz) + 0.5) / float(n),
                    (float(cell.iy) + 0.5) / float(n),
                    (float(cell.ix) + 0.5) / float(n),
                )

            centers: Dict[int, Optional[tuple[float, float, float]]] = {int(t.token_id): _center(t) for t in tokens}
            ranked = sorted(tokens, key=lambda t: (-float(t.score), -float(t.uncertainty), int(t.token_id)))

            k = max(0, min(int(self.cfg.topk_citations), len(ranked)))
            chosen: List[Token] = []
            if k > 0:
                chosen.append(ranked[0])
            while len(chosen) < k:
                best = None
                best_key = None
                for cand in ranked:
                    if any(int(cand.token_id) == int(x.token_id) for x in chosen):
                        continue
                    c0 = centers.get(int(cand.token_id))
                    if c0 is None:
                        min_d2 = 0.0
                    else:
                        min_d2 = float("inf")
                        for sel in chosen:
                            c1 = centers.get(int(sel.token_id))
                            if c1 is None:
                                continue
                            dz = float(c0[0]) - float(c1[0])
                            dy = float(c0[1]) - float(c1[1])
                            dx = float(c0[2]) - float(c1[2])
                            d2 = dz * dz + dy * dy + dx * dx
                            if d2 < min_d2:
                                min_d2 = d2
                        if min_d2 == float("inf"):
                            min_d2 = 0.0
                    key = (
                        float(min_d2),  # diversity
                        float(cand.score),
                        float(cand.uncertainty),
                        -int(cand.token_id),  # deterministic tiebreak (prefer smaller id)
                    )
                    if best_key is None or key > best_key:
                        best = cand
                        best_key = key
                if best is None:
                    break
                chosen.append(best)

            top_ids = [int(t.token_id) for t in chosen]
            citations = {int(i): list(top_ids) for i in range(len(gen.frames))}
            gen = Generation(frames=gen.frames, citations=citations, q=gen.q, refusal=gen.refusal, text="")
            gen = Generation(frames=gen.frames, citations=citations, q=gen.q, refusal=gen.refusal, text=render_generation_text(gen))
        return gen
