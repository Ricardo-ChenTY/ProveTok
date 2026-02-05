from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Frame, Generation, Issue, Token


def reindex_tokens(tokens: Sequence[Token]) -> List[Token]:
    """Return a new token list with token_id = position (required by ToyPCG scaffold)."""
    out: List[Token] = []
    for i, t in enumerate(tokens):
        out.append(
            Token(
                token_id=i,
                cell_id=t.cell_id,
                level=t.level,
                embedding=t.embedding,
                score=t.score,
                uncertainty=t.uncertainty,
            )
        )
    return out


def permute_cell_ids(tokens: Sequence[Token], *, seed: int) -> List[Token]:
    """Ω-permutation: permute cell_id while keeping other fields (incl embedding) fixed.

    Important: permute within each token level to avoid confounding Ω-location with Ω-size.
    """
    rng = np.random.RandomState(seed)
    permuted_cell_ids = [t.cell_id for t in tokens]
    by_level: Dict[int, List[int]] = {}
    for idx, t in enumerate(tokens):
        by_level.setdefault(int(t.level), []).append(int(idx))

    for _, idxs in by_level.items():
        if len(idxs) < 2:
            continue
        ids = [permuted_cell_ids[i] for i in idxs]
        perm = rng.permutation(len(ids))
        for j, i in enumerate(idxs):
            permuted_cell_ids[i] = ids[int(perm[j])]

    out: List[Token] = []
    for i, t in enumerate(tokens):
        out.append(
            Token(
                token_id=t.token_id,
                cell_id=permuted_cell_ids[i],
                level=t.level,
                embedding=t.embedding,
                score=t.score,
                uncertainty=t.uncertainty,
            )
        )
    return out


def permute_embeddings(tokens: Sequence[Token], *, seed: int) -> List[Token]:
    """Token-permutation: permute embeddings while keeping Ω (cell_id) fixed."""
    rng = np.random.RandomState(seed)
    embs = [t.embedding for t in tokens]
    perm = rng.permutation(len(embs))
    permuted_embs = [embs[i] for i in perm]

    out: List[Token] = []
    for i, t in enumerate(tokens):
        out.append(
            Token(
                token_id=t.token_id,
                cell_id=t.cell_id,
                level=t.level,
                embedding=permuted_embs[i],
                score=t.score,
                uncertainty=t.uncertainty,
            )
        )
    return out


def swap_citations(gen: Generation, *, seed: int) -> Generation:
    """Citation-swap: swap C_k within a single report while preserving |C_k| distribution."""
    rng = np.random.RandomState(seed)
    keys = sorted(gen.citations.keys())
    if len(keys) < 2:
        return gen
    citation_lists = [gen.citations[k] for k in keys]
    perm = rng.permutation(len(citation_lists))
    swapped = {k: citation_lists[int(perm[i])] for i, k in enumerate(keys)}
    return Generation(frames=gen.frames, citations=swapped, q=gen.q, refusal=gen.refusal)


def drop_cited_tokens(tokens: Sequence[Token], gen: Generation) -> List[Token]:
    """Evidence-drop: remove all cited tokens (union over frames)."""
    cited = set()
    for cites in gen.citations.values():
        cited.update(int(x) for x in cites)
    kept = [t for t in tokens if t.token_id not in cited]
    return reindex_tokens(kept)


def remove_all_citations(gen: Generation) -> Generation:
    empty = {k: [] for k in gen.citations.keys()}
    return Generation(frames=gen.frames, citations=empty, q=gen.q, refusal=gen.refusal)


def issue_counts(issues: Sequence[Issue]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for iss in issues:
        counts[iss.issue_type] = counts.get(iss.issue_type, 0) + 1
    return counts


def issue_rate(issues: Sequence[Issue], *, num_frames: int) -> Dict[str, float]:
    denom = max(int(num_frames), 1)
    counts = issue_counts(issues)
    return {k: v / denom for k, v in counts.items()}
