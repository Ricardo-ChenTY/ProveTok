from __future__ import annotations

from typing import Dict, List

from ..types import Generation


def apply_no_citation(gen: Generation) -> Generation:
    empty = {k: [] for k in gen.citations.keys()}
    return Generation(frames=gen.frames, citations=empty, q=gen.q, refusal=gen.refusal)


def apply_citation_only(gen: Generation) -> Generation:
    """Keep citations but disable refusal (as a protocol ablation scaffold)."""
    refusal: Dict[int, bool] = {k: False for k in range(len(gen.frames))}
    return Generation(frames=gen.frames, citations=gen.citations, q=gen.q, refusal=refusal)

