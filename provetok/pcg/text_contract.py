from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..types import Frame, Generation, TokenRef
from ..data.frame_extractor import FrameExtractor


_CITE_RE = re.compile(r"\[\s*CITE\s*:\s*(?P<ids>[^\]]*)\]", re.IGNORECASE)


def _frame_to_sentence(frame: Frame) -> str:
    if str(frame.polarity) in ("absent", "negative"):
        return f"No {frame.finding}."
    lat = ""
    if str(frame.laterality) in ("left", "right", "bilateral"):
        lat = f"{frame.laterality.capitalize()} "
    hedged = "Possible " if bool(getattr(frame, "uncertain", False)) else ""
    return f"{hedged}{lat}{frame.finding}.".strip()


def render_findings(gen: Generation, *, k_max: int = 8) -> List[str]:
    """Render pp.md-style Findings lines with trailing `[CITE: ...]`.

    Notes:
    - Uses `gen.citations_ref` when available; otherwise falls back to `gen.citations`.
    - Always emits a `[CITE: ...]` suffix to keep the contract parseable, even for absent frames.
    """
    out: List[str] = []
    for idx, fr in enumerate(gen.frames):
        sent = _frame_to_sentence(fr)
        refs: List[str] = []
        if gen.citations_ref and idx in (gen.citations_ref or {}):
            refs = [str(x) for x in (gen.citations_ref or {}).get(idx, [])]
        elif idx in (gen.citations or {}):
            refs = [str(int(x)) for x in (gen.citations or {}).get(idx, [])]
        refs = refs[: max(0, int(k_max))]
        out.append(f"{sent} [CITE: {', '.join(refs)}]")
    return out


def render_impression(findings_lines: Sequence[str]) -> str:
    """Render a conservative Impression that introduces no new citations."""
    if not findings_lines:
        return "Impression: No significant abnormality."
    return "Impression: Please refer to Findings."


@dataclass(frozen=True)
class ParsedFindingLine:
    text: str
    citations: List[str]


def parse_findings_lines(lines: Sequence[str]) -> List[ParsedFindingLine]:
    """Parse pp.md-style finding lines, extracting `[CITE: ...]` lists."""
    out: List[ParsedFindingLine] = []
    for ln in lines:
        s = str(ln or "").strip()
        if not s:
            continue
        m = _CITE_RE.search(s)
        cites: List[str] = []
        if m is not None:
            raw = str(m.group("ids") or "")
            cites = [x.strip() for x in raw.split(",") if x.strip()]
            s = (s[: m.start()] + s[m.end() :]).strip()
        out.append(ParsedFindingLine(text=s, citations=cites))
    return out


def parse_findings_text(text: str) -> Tuple[List[Frame], Dict[int, List[TokenRef]]]:
    """Parse a multi-line Findings block into frames + stable-id citations.

    This is used by the post-hoc citation wrapper and for auditing external baselines.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    parsed = parse_findings_lines(lines)
    extractor = FrameExtractor()
    frames: List[Frame] = []
    cites_ref: Dict[int, List[TokenRef]] = {}
    for i, row in enumerate(parsed):
        frs = extractor.extract_frames(row.text)
        # Keep one frame per line for auditability; fall back to "normal" frame.
        fr = frs[0] if frs else Frame(finding="normal", polarity="absent", laterality="unspecified", confidence=0.5)
        frames.append(fr)
        cites_ref[int(i)] = [str(x) for x in row.citations]
    return frames, cites_ref


def enforce_pp_contract(
    gen: Generation,
    tokens,
    *,
    k_max: int = 8,
) -> Generation:
    """Enforce pp.md Findings citation contract on a structured Generation.

    Ensures:
    - citations contain only token ids present in `tokens`
    - citations are truncated to k_max
    - citations_ref is populated with stable token refs (Token.ref), when available
    """
    toks = list(tokens or [])
    token_by_id = {int(getattr(t, "token_id", -1)): t for t in toks}

    citations: Dict[int, List[int]] = {}
    citations_ref: Dict[int, List[str]] = {}
    for idx, _ in enumerate(gen.frames or []):
        idx_i = int(idx)
        raw = (gen.citations or {}).get(idx_i, []) or []
        cleaned: List[int] = []
        for x in raw:
            try:
                tid = int(x)
            except Exception:
                continue
            if tid in token_by_id:
                cleaned.append(int(tid))
        cleaned = cleaned[: max(0, int(k_max))]
        citations[idx_i] = list(cleaned)
        refs = [str(getattr(token_by_id[tid], "ref", tid)) for tid in cleaned if tid in token_by_id]
        citations_ref[idx_i] = refs

    return Generation(
        frames=list(gen.frames or []),
        citations=citations,
        q=dict(gen.q or {}),
        refusal=dict(gen.refusal or {}),
        citations_ref=citations_ref,
        text=str(getattr(gen, "text", "") or ""),
        impression=str(getattr(gen, "impression", "") or ""),
    )
