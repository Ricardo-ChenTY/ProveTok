from __future__ import annotations

from typing import Dict, List

from ..pcg.narrative import render_generation_text
from ..types import Generation


def apply_no_citation(gen: Generation) -> Generation:
    empty: Dict[int, List[int]] = {int(i): [] for i in range(len(gen.frames or []))}
    tmp = Generation(
        frames=list(gen.frames or []),
        citations=dict(empty),
        q=dict(gen.q or {}),
        refusal=dict(gen.refusal or {}),
        citations_ref={int(k): [] for k in empty.keys()},
        text="",
        impression=str(getattr(gen, "impression", "") or ""),
        report_text=str(getattr(gen, "report_text", "") or ""),
    )
    return Generation(
        frames=tmp.frames,
        citations=tmp.citations,
        q=tmp.q,
        refusal=tmp.refusal,
        citations_ref=tmp.citations_ref,
        text=render_generation_text(tmp),
        impression=tmp.impression,
        report_text=tmp.report_text,
    )


def apply_citation_only(gen: Generation) -> Generation:
    """Keep citations but disable refusal (as a protocol ablation scaffold)."""
    refusal: Dict[int, bool] = {int(k): False for k in range(len(gen.frames or []))}
    tmp = Generation(
        frames=list(gen.frames or []),
        citations={int(k): list(v or []) for k, v in (gen.citations or {}).items()},
        q=dict(gen.q or {}),
        refusal=refusal,
        citations_ref={int(k): [str(x) for x in (v or [])] for k, v in (gen.citations_ref or {}).items()},
        text="",
        impression=str(getattr(gen, "impression", "") or ""),
        report_text=str(getattr(gen, "report_text", "") or ""),
    )
    return Generation(
        frames=tmp.frames,
        citations=tmp.citations,
        q=tmp.q,
        refusal=tmp.refusal,
        citations_ref=tmp.citations_ref,
        text=render_generation_text(tmp),
        impression=tmp.impression,
        report_text=tmp.report_text,
    )
