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
                ref=str(getattr(t, "ref", t.cell_id)),
                bounds_voxel=tuple(getattr(t, "bounds_voxel", (0, 0, 0, 0, 0, 0))),
                center_voxel=tuple(getattr(t, "center_voxel", (0.0, 0.0, 0.0))),
                bounds_mm=getattr(t, "bounds_mm", None),
                center_mm=getattr(t, "center_mm", None),
            )
        )
    return out


def permute_cell_ids(tokens: Sequence[Token], *, seed: int) -> List[Token]:
    """Ω-permutation: permute cell_id while keeping other fields (incl embedding) fixed.

    Important: permute within each token level to avoid confounding Ω-location with Ω-size.
    """
    rng = np.random.RandomState(seed)
    permuted_cell_ids = [t.cell_id for t in tokens]
    permuted_refs = [str(getattr(t, "ref", t.cell_id)) for t in tokens]
    permuted_bounds = [tuple(getattr(t, "bounds_voxel", (0, 0, 0, 0, 0, 0))) for t in tokens]
    permuted_centers = [tuple(getattr(t, "center_voxel", (0.0, 0.0, 0.0))) for t in tokens]
    permuted_bounds_mm = [getattr(t, "bounds_mm", None) for t in tokens]
    permuted_centers_mm = [getattr(t, "center_mm", None) for t in tokens]
    by_level: Dict[int, List[int]] = {}
    for idx, t in enumerate(tokens):
        by_level.setdefault(int(t.level), []).append(int(idx))

    for _, idxs in by_level.items():
        if len(idxs) < 2:
            continue
        # Permute Ω (cell_id) consistently with its geometry metadata.
        ids = [permuted_cell_ids[i] for i in idxs]
        refs = [permuted_refs[i] for i in idxs]
        bds = [permuted_bounds[i] for i in idxs]
        cts = [permuted_centers[i] for i in idxs]
        bds_mm = [permuted_bounds_mm[i] for i in idxs]
        cts_mm = [permuted_centers_mm[i] for i in idxs]

        perm = rng.permutation(len(ids))
        for j, i in enumerate(idxs):
            src = int(perm[j])
            permuted_cell_ids[i] = ids[src]
            permuted_refs[i] = refs[src]
            permuted_bounds[i] = bds[src]
            permuted_centers[i] = cts[src]
            permuted_bounds_mm[i] = bds_mm[src]
            permuted_centers_mm[i] = cts_mm[src]

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
                ref=str(permuted_refs[i] or permuted_cell_ids[i]),
                bounds_voxel=tuple(permuted_bounds[i]),
                center_voxel=tuple(permuted_centers[i]),
                bounds_mm=permuted_bounds_mm[i],
                center_mm=permuted_centers_mm[i],
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
                ref=str(getattr(t, "ref", t.cell_id)),
                bounds_voxel=tuple(getattr(t, "bounds_voxel", (0, 0, 0, 0, 0, 0))),
                center_voxel=tuple(getattr(t, "center_voxel", (0.0, 0.0, 0.0))),
                bounds_mm=getattr(t, "bounds_mm", None),
                center_mm=getattr(t, "center_mm", None),
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
    swapped_ref = None
    if getattr(gen, "citations_ref", None):
        cite_ref_lists = [list((gen.citations_ref or {}).get(k, [])) for k in keys]
        swapped_ref = {k: cite_ref_lists[int(perm[i])] for i, k in enumerate(keys)}
    return Generation(frames=gen.frames, citations=swapped, q=gen.q, refusal=gen.refusal, citations_ref=swapped_ref or {})


def drop_cited_tokens(tokens: Sequence[Token], gen: Generation) -> List[Token]:
    """Evidence-drop: remove all cited tokens (union over frames)."""
    cited = set()
    for cites in gen.citations.values():
        cited.update(int(x) for x in cites)
    kept = [t for t in tokens if t.token_id not in cited]
    return reindex_tokens(kept)


def remove_all_citations(gen: Generation) -> Generation:
    empty = {k: [] for k in gen.citations.keys()}
    empty_ref = {k: [] for k in (gen.citations_ref or {}).keys()}
    return Generation(frames=gen.frames, citations=empty, q=gen.q, refusal=gen.refusal, citations_ref=empty_ref)


def issue_counts(issues: Sequence[Issue]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for iss in issues:
        counts[iss.issue_type] = counts.get(iss.issue_type, 0) + 1
    return counts


def issue_rate(issues: Sequence[Issue], *, num_frames: int) -> Dict[str, float]:
    denom = max(int(num_frames), 1)
    counts = issue_counts(issues)
    return {k: v / denom for k, v in counts.items()}
