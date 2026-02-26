from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.frame_extractor import FrameExtractor, frames_to_report
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
    contract_mode: str = "full"        # "free_form" | "schema_only" | "schema_citations" | "full" | "inline_citation"
    citation_source: str = "score_override"  # "score_override" | "llm"
    lora_adapter_path: str = ""  # optional PEFT adapter (LoRA) path
    lora_merge: bool = False        # optionally merge adapter weights for inference
    inline_citation_prefix: str = "CIT"
    inline_citation_digits: int = 3


def build_llama2_json_prompt(tokens: List[Token], *, cfg: Llama2PCGConfig, max_tokens_in_prompt: Optional[int] = None) -> str:
    """Build a bounded Llama-2 chat prompt that requests a single JSON object."""
    limit = int(cfg.max_tokens_in_prompt if max_tokens_in_prompt is None else max_tokens_in_prompt)
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

    mode = str(getattr(cfg, "contract_mode", "full")).strip().lower()
    max_frames = max(0, int(getattr(cfg, "max_frames", 1)))
    require_citations = mode in ("schema_citations", "full")

    sys_msg = (
        "You are a strict JSON generator for a radiology-claim schema. "
        "Output ONLY a single JSON object. No markdown. No commentary."
    )
    # Keep the template minimal to reduce the chance of JSON truncation.
    # Optional frame slots are filled by `sanitize_generation_dict`.
    template: Dict[str, Any] = {
        "frames": [{"finding": "opacity", "polarity": "present", "laterality": "unspecified", "confidence": 0.5}],
        "citations": ({"0": [0]} if require_citations else {}),
        "q": {"0": 0.5},
        "refusal": {"0": False},
    }

    cite_rule = (
        "- citations must be a JSON object mapping each frame index (as a string) to a list of token ids.\n"
        if require_citations
        else "- citations must be an empty JSON object: {}.\n"
    )
    user_msg = (
        "Return ONLY valid JSON that can be parsed by Python json.loads.\n"
        "Rules:\n"
        "- Output exactly one JSON object.\n"
        "- Use double quotes for all keys/strings.\n"
        "- Do NOT write any text before/after the JSON.\n"
        f"- frames must contain at most {max_frames} items.\n"
        "- For brevity, each frame should include ONLY: finding, polarity, laterality, confidence.\n"
        + cite_rule +
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


def build_llama2_free_form_prompt(tokens: List[Token], *, cfg: Llama2PCGConfig, max_tokens_in_prompt: Optional[int] = None) -> str:
    """Build a bounded prompt that asks for a free-form radiology report text."""
    limit = int(cfg.max_tokens_in_prompt if max_tokens_in_prompt is None else max_tokens_in_prompt)
    toks = sorted(tokens, key=lambda t: (-float(t.score), int(t.token_id)))[: max(0, limit)]
    tok_lines = [
        f"- id={t.token_id} cell_id={t.cell_id} score={t.score:.3f} uncertainty={t.uncertainty:.3f} level={t.level}"
        for t in toks
    ]

    max_frames = max(0, int(getattr(cfg, "max_frames", 1)))
    sys_msg = "You are a radiology report generator. Output ONLY plain text. No JSON. No markdown."
    user_msg = (
        "Write a short radiology report (1-4 sentences) describing up to "
        f"{max_frames} findings.\n"
        "Do not include any JSON.\n"
        "Do not include token ids or citations.\n"
        f"Allowed finding terms (preferred): {', '.join(FINDINGS)}.\n"
        "Evidence tokens (for your internal grounding, do not copy verbatim):\n"
        + "\n".join(tok_lines)
        + "\n"
    )
    return f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{user_msg} [/INST]"


def _format_citation_tag(idx: int, *, cfg: Llama2PCGConfig) -> str:
    prefix = str(getattr(cfg, "inline_citation_prefix", "CIT") or "CIT").strip().upper()
    if not re.match(r"^[A-Z][A-Z0-9_]*$", prefix):
        prefix = "CIT"
    digits = max(2, int(getattr(cfg, "inline_citation_digits", 3)))
    return f"{prefix}_{int(idx):0{digits}d}"


def build_llama2_inline_citation_prompt(
    tokens: List[Token],
    *,
    cfg: Llama2PCGConfig,
    max_tokens_in_prompt: Optional[int] = None,
) -> tuple[str, Dict[str, int]]:
    """Build a free-form report prompt with explicit inline citation tags."""
    limit = int(cfg.max_tokens_in_prompt if max_tokens_in_prompt is None else max_tokens_in_prompt)
    toks = sorted(tokens, key=lambda t: (-float(t.score), int(t.token_id)))[: max(0, limit)]

    tag_to_token_id: Dict[str, int] = {}
    tok_lines: List[str] = []
    for i, t in enumerate(toks, start=1):
        tag = _format_citation_tag(i, cfg=cfg)
        tag_to_token_id[tag] = int(t.token_id)
        tok_lines.append(
            f"- [{tag}] token_id={t.token_id} cell_id={t.cell_id} score={t.score:.3f} "
            f"uncertainty={t.uncertainty:.3f} level={t.level}"
        )

    max_frames = max(0, int(getattr(cfg, "max_frames", 1)))
    topk = max(1, int(getattr(cfg, "topk_citations", 3)))
    sys_msg = (
        "You are a radiology report generator. Output ONLY plain text. "
        "Use inline evidence citations from the allowed tags."
    )
    user_msg = (
        "Write a short radiology report (1-4 sentences) describing up to "
        f"{max_frames} findings.\n"
        "For each positive finding sentence, include 1-" + str(topk) + " inline citation tags.\n"
        "Format example: Right upper lobe nodule [CIT_001, CIT_002] suspicious for malignancy.\n"
        "Do not invent tags. Use ONLY tags from the list below.\n"
        "Allowed finding terms (preferred): " + ", ".join(FINDINGS) + ".\n"
        "Evidence tag list:\n"
        + "\n".join(tok_lines)
        + "\n"
    )
    return f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{user_msg} [/INST]", tag_to_token_id


def _extract_inline_citations(
    report_text: str,
    *,
    tag_to_token_id: Dict[str, int],
    topk_citations: int,
    max_frames: int,
) -> tuple[List[Frame], Dict[int, List[int]], str]:
    """Parse inline citation tags from report text and align them to extracted frames."""
    raw = str(report_text or "").strip()
    if not raw:
        return [], {}, ""

    # Normalize common full-width punctuations for multilingual report drafts.
    norm = (
        raw.replace("，", ",")
        .replace("；", ";")
        .replace("。", ".")
        .replace("！", "!")
        .replace("？", "?")
        .replace("【", "[")
        .replace("】", "]")
        .replace("（", "(")
        .replace("）", ")")
    )

    def _canon_tag(tag: str) -> str:
        t = str(tag or "").strip().upper()
        m = re.match(r"^([A-Z][A-Z0-9_]*?)_?(\d{2,5})$", t)
        if m is None:
            return t
        return f"{m.group(1)}_{m.group(2)}"

    # Normalize citation tags to upper-case for robust matching.
    tag_to_tid = {_canon_tag(str(k)): int(v) for k, v in (tag_to_token_id or {}).items()}
    sent_split = re.compile(r"[\n\.\!\?;]+")
    sents = [s.strip() for s in sent_split.split(norm) if s.strip()]

    sent_citations: Dict[int, List[int]] = {}
    cleaned_sents: List[str] = []
    for i, sent in enumerate(sents):
        tags_raw = re.findall(r"\b([A-Za-z][A-Za-z0-9_]*_?\d{2,5})\b", sent)
        cited_ids: List[int] = []
        seen: set[int] = set()
        for tr in tags_raw:
            tu = _canon_tag(tr)
            if tu not in tag_to_tid:
                continue
            tid = int(tag_to_tid[tu])
            if tid in seen:
                continue
            seen.add(tid)
            cited_ids.append(tid)
            if len(cited_ids) >= max(1, int(topk_citations)):
                break
        sent_citations[int(i)] = cited_ids

        # Remove bracketed and standalone citation tags before frame extraction.
        no_brackets = re.sub(r"\[[^\]]*?[A-Za-z][A-Za-z0-9_]*_?\d{2,5}[^\]]*?\]", " ", sent)
        no_tags = re.sub(r"\b[A-Za-z][A-Za-z0-9_]*_?\d{2,5}\b", " ", no_brackets)
        cleaned_sents.append(" ".join(no_tags.split()))

    cleaned_text = ". ".join([s for s in cleaned_sents if s]).strip()
    extractor = FrameExtractor()
    frames = extractor.extract_frames(cleaned_text)
    frames = frames[: max(0, int(max_frames))]

    citations: Dict[int, List[int]] = {}
    for k in range(len(frames)):
        citations[int(k)] = list(sent_citations.get(int(k), []))
    return frames, citations, cleaned_text


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
    citations_ref: Dict[int, List[str]] = {}
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
        # Stable-id citations (pp.md-style). When Token.ref is not available at
        # this stage, we use the token_id string as the stable reference.
        citations_ref[k] = [str(int(x)) for x in cites]

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

    gen_tmp = Generation(frames=frames, citations=citations, q=q, refusal=refusal, citations_ref=citations_ref, text="")
    return Generation(
        frames=frames,
        citations=citations,
        q=q,
        refusal=refusal,
        citations_ref=citations_ref,
        text=render_generation_text(gen_tmp),
        report_text=frames_to_report(frames),
    )


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

        # Optional LoRA/PEFT adapter loading (pp.md §6.6).
        lora_path = str(getattr(cfg, "lora_adapter_path", "") or "").strip()
        if lora_path:
            try:
                from peft import PeftModel  # type: ignore
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "Loading LoRA adapters requires optional dependency `peft`. "
                    "Install via `pip install peft` (and bitsandbytes for 8-bit base models)."
                ) from e
            self.model = PeftModel.from_pretrained(self.model, lora_path, is_trainable=False)
            if bool(getattr(cfg, "lora_merge", False)) and hasattr(self.model, "merge_and_unload"):
                try:
                    self.model = self.model.merge_and_unload()
                except Exception:
                    # Some quantized backends may not support merging; keep adapter attached.
                    pass
            self.model.eval()

    def _build_prompt(self, tokens: List[Token], *, max_tokens_in_prompt: Optional[int] = None) -> str:
        mode = str(getattr(self.cfg, "contract_mode", "full")).strip().lower()
        if mode == "free_form":
            return build_llama2_free_form_prompt(tokens, cfg=self.cfg, max_tokens_in_prompt=max_tokens_in_prompt)
        return build_llama2_json_prompt(tokens, cfg=self.cfg, max_tokens_in_prompt=max_tokens_in_prompt)

    def _parse_json(self, text: str) -> Dict:
        return parse_llm_json(text)

    def _sanitize(self, d: Dict, *, token_ids: List[int]) -> Generation:
        return sanitize_generation_dict(d, token_ids=token_ids, cfg=self.cfg)

    def _apply_score_override(self, gen: Generation, tokens: List[Token]) -> Generation:
        """Deterministic score-based citation repair with diversity tie-break."""
        if not gen.frames:
            return gen

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
        citations_ref = {int(i): [str(int(x)) for x in top_ids] for i in range(len(gen.frames))}
        gen = Generation(
            frames=gen.frames,
            citations=citations,
            q=gen.q,
            refusal=gen.refusal,
            citations_ref=citations_ref,
            text="",
            impression=str(getattr(gen, 'impression', '') or ''),
            report_text=str(getattr(gen, 'report_text', '') or ''),
        )
        return Generation(
            frames=gen.frames,
            citations=citations,
            q=gen.q,
            refusal=gen.refusal,
            citations_ref=gen.citations_ref,
            text=render_generation_text(gen),
            impression=str(getattr(gen, 'impression', '') or ''),
            report_text=str(getattr(gen, 'report_text', '') or ''),
        )

    @torch.no_grad()
    def __call__(self, tokens: List[Token]) -> Generation:
        if not tokens:
            return Generation(frames=[], citations={}, q={}, refusal={}, text="")

        max_ctx = int(getattr(getattr(self.model, "config", None), "max_position_embeddings", 4096) or 4096)
        max_new = int(self.cfg.max_new_tokens)
        # Keep a small margin for special tokens; avoid hitting context limit.
        max_input = max(1, int(max_ctx) - int(max_new) - 8)

        mode = str(getattr(self.cfg, "contract_mode", "full")).strip().lower()
        inline_tag_map: Dict[str, int] = {}

        # Some runs may produce very long token lists; shrink the evidence token
        # section until the prompt fits in the model context window.
        limit = int(self.cfg.max_tokens_in_prompt)
        if mode == "inline_citation":
            prompt, inline_tag_map = build_llama2_inline_citation_prompt(
                tokens,
                cfg=self.cfg,
                max_tokens_in_prompt=limit,
            )
        else:
            prompt = self._build_prompt(tokens, max_tokens_in_prompt=limit)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        while int(inputs["input_ids"].shape[1]) > max_input and limit > 8:
            limit = max(8, limit // 2)
            if mode == "inline_citation":
                prompt, inline_tag_map = build_llama2_inline_citation_prompt(
                    tokens,
                    cfg=self.cfg,
                    max_tokens_in_prompt=limit,
                )
            else:
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
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if mode == "free_form":
            extractor = FrameExtractor()
            frames = extractor.extract_frames(text)[: max(0, int(getattr(self.cfg, "max_frames", 1)))]
            if not frames:
                finding = str(getattr(self.cfg, "fallback_finding", "opacity")).lower()
                if finding not in set(FINDINGS):
                    finding = "opacity"
                frames = [
                    Frame(
                        finding=finding,
                        polarity="present",
                        laterality="unspecified",
                        confidence=0.5,
                        location="unspecified",
                        size_bin="unspecified",
                        severity="unspecified",
                        uncertain=True,
                    )
                ]
            citations = {int(i): [] for i in range(len(frames))}
            q = {int(i): _clamp01(float(getattr(fr, "confidence", 0.5))) for i, fr in enumerate(frames)}
            refusal = {
                int(i): bool(q[int(i)] < float(self.cfg.tau_refuse) and str(getattr(fr, "polarity", "")) in ("present", "positive"))
                for i, fr in enumerate(frames)
            }
            # Keep the raw free-form report text separate from the dual-channel narrative.
            citations_ref = {int(i): [] for i in range(len(frames))}
            tmp = Generation(
                frames=frames,
                citations=citations,
                q=q,
                refusal=refusal,
                citations_ref=citations_ref,
                text="",
                report_text=str(text),
            )

        if mode == "inline_citation":
            frames, citations, _ = _extract_inline_citations(
                text,
                tag_to_token_id=inline_tag_map,
                topk_citations=int(self.cfg.topk_citations),
                max_frames=int(getattr(self.cfg, "max_frames", 1)),
            )
            if not frames:
                finding = str(getattr(self.cfg, "fallback_finding", "opacity")).lower()
                if finding not in set(FINDINGS):
                    finding = "opacity"
                frames = [
                    Frame(
                        finding=finding,
                        polarity="present",
                        laterality="unspecified",
                        confidence=0.5,
                        location="unspecified",
                        size_bin="unspecified",
                        severity="unspecified",
                        uncertain=True,
                    )
                ]
                citations = {0: []}

            q = {int(i): _clamp01(float(getattr(fr, "confidence", 0.5))) for i, fr in enumerate(frames)}
            refusal = {
                int(i): bool(q[int(i)] < float(self.cfg.tau_refuse) and str(getattr(fr, "polarity", "")) in ("present", "positive"))
                for i, fr in enumerate(frames)
            }
            citations_ref = {
                int(i): [str(int(x)) for x in (citations.get(int(i), []) or [])]
                for i in range(len(frames))
            }
            tmp = Generation(
                frames=frames,
                citations={int(i): list(citations.get(int(i), []) or []) for i in range(len(frames))},
                q=q,
                refusal=refusal,
                citations_ref=citations_ref,
                text="",
                report_text=str(text),
            )
            gen = Generation(
                frames=tmp.frames,
                citations=tmp.citations,
                q=tmp.q,
                refusal=tmp.refusal,
                citations_ref=tmp.citations_ref,
                text=render_generation_text(tmp),
                report_text=str(text),
            )
            citation_source = str(getattr(self.cfg, "citation_source", "llm")).strip().lower()
            if citation_source == "llm":
                return gen
            return self._apply_score_override(gen, tokens)
            return Generation(
                frames=frames,
                citations=citations,
                q=q,
                refusal=refusal,
                citations_ref=citations_ref,
                text=render_generation_text(tmp),
                report_text=str(text),
            )

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
                citations_ref={0: []},
                text="",
                report_text=frames_to_report([fr]),
            )
            gen = Generation(
                frames=gen.frames,
                citations=gen.citations,
                q=gen.q,
                refusal=gen.refusal,
                citations_ref=gen.citations_ref,
                text=render_generation_text(gen),
                report_text=str(getattr(gen, 'report_text', '') or ''),
            )

        # Contract modes:
        # - schema_only: citations are intentionally absent
        # - schema_citations/full: citations are required; default is deterministic score-based override.
        if mode == "schema_only":
            citations = {int(i): [] for i in range(len(gen.frames))}
            citations_ref = {int(i): [] for i in range(len(gen.frames))}
            gen = Generation(
                frames=gen.frames,
                citations=citations,
                q=gen.q,
                refusal=gen.refusal,
                citations_ref=citations_ref,
                text=gen.text,
                impression=str(getattr(gen, 'impression', '') or ''),
                report_text=str(getattr(gen, 'report_text', '') or ''),
            )
            return gen

        citation_source = str(getattr(self.cfg, "citation_source", "score_override")).strip().lower()
        if citation_source == "llm":
            # Keep sanitized citations from the model output (already filtered to valid token ids).
            return gen
        return self._apply_score_override(gen, tokens)
