from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
import torch

from ..bet.tokenize import TokenEncoder, encode_tokens
from ..grid.cells import Cell, root_cell, split, cell_bounds
from ..types import Token


class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, volume: torch.Tensor, *, budget_tokens: int, emb_dim: int, seed: int) -> List[Token]:
        pass


def _heuristic_cell_score(volume: torch.Tensor, cell: Cell) -> float:
    """Cheap ROI heuristic score based on local variance (no embedding compute)."""
    bounds = cell_bounds(cell, shape=tuple(volume.shape))
    patch = volume[bounds[0], bounds[1], bounds[2]]
    var = float(patch.var(unbiased=False).item())
    return float(1.0 / (1.0 + np.exp(-3.0 * (var - 0.5))))


def _uniform_cells_to_budget(budget_tokens: int, *, max_depth: int = 6) -> List[Cell]:
    """Deterministically refine a full-covering cell set until reaching budget."""
    cells: List[Cell] = [root_cell()]
    while len(cells) < budget_tokens:
        # Split the lexicographically smallest splittable cell to avoid randomness.
        splittable = [c for c in cells if c.level < max_depth]
        if not splittable:
            break
        c_star = sorted(splittable, key=lambda c: c.id())[0]
        cells = [c for c in cells if c.id() != c_star.id()] + split(c_star)
    return sorted(cells, key=lambda c: c.id())[:budget_tokens]


def _level_for_coverage_budget(budget_tokens: int, *, min_cover_tokens: int, max_depth: int) -> int:
    """Pick an initial grid level such that (2^L)^3 <= min(budget_tokens, min_cover_tokens).

    This provides a coarse full-volume coverage, then lets adaptive tokenizers refine
    where useful.
    """
    target = max(1, min(int(budget_tokens), int(min_cover_tokens)))
    level = 0
    while level < int(max_depth):
        nxt = level + 1
        if (2**nxt) ** 3 <= int(target):
            level = nxt
        else:
            break
    return int(level)


class BETAlgTokenizer(BaseTokenizer):
    """Adaptive greedy refinement baseline ("BET-Alg").

    Starts from a coarse full-grid coverage, then repeatedly splits the highest-score
    leaf cell until reaching the token budget. Scores are deterministic:
    - If `token_score_fn` is provided, uses it on token embeddings.
    - Otherwise uses a cheap variance heuristic (`_heuristic_cell_score`).
    """

    def __init__(
        self,
        *,
        max_depth: int = 6,
        min_cover_tokens: int = 64,
        token_score_fn=None,
        token_score_level_power: float = 0.0,
    ):
        self.max_depth = int(max_depth)
        self.min_cover_tokens = int(min_cover_tokens)
        self.token_score_fn = token_score_fn
        self.token_score_level_power = float(token_score_level_power)

    def tokenize(self, volume: torch.Tensor, *, budget_tokens: int, emb_dim: int, seed: int) -> List[Token]:
        import heapq

        budget_tokens = int(budget_tokens)
        if budget_tokens <= 0:
            return []

        token_encoder = None
        if self.token_score_fn is not None:
            token_encoder = TokenEncoder(volume=volume, emb_dim=int(emb_dim), seed=int(seed), encoder=None)

        init_level = _level_for_coverage_budget(
            budget_tokens,
            min_cover_tokens=int(self.min_cover_tokens),
            max_depth=int(self.max_depth),
        )
        n = 2 ** int(init_level)
        cells: List[Cell] = [Cell(level=int(init_level), ix=ix, iy=iy, iz=iz) for ix in range(n) for iy in range(n) for iz in range(n)]

        # Cache per-cell scores so we can truncate by score if we overshoot the budget.
        score_cache: dict[str, float] = {}

        def score_cell(cell: Cell) -> float:
            cid = cell.id()
            if cid in score_cache:
                return float(score_cache[cid])
            if self.token_score_fn is None:
                s = float(_heuristic_cell_score(volume, cell))
            else:
                assert token_encoder is not None
                tok = token_encoder.encode(
                    [cell],
                    token_score_fn=self.token_score_fn,
                    token_score_level_power=float(self.token_score_level_power),
                )
                s = float(tok[0].score) if tok else 0.0
            score_cache[cid] = float(s)
            return float(s)

        # Greedy refinement using a max-heap. Store (-score, cell_id, cell).
        leaf_by_id = {c.id(): c for c in cells}
        heap = [(-score_cell(c), c.id(), c) for c in cells]
        heapq.heapify(heap)

        while len(leaf_by_id) < budget_tokens:
            # Find the best current leaf that can still be split.
            best = None
            while heap:
                neg_s, cid, c = heapq.heappop(heap)
                if cid not in leaf_by_id:
                    continue
                if int(c.level) >= int(self.max_depth):
                    continue
                best = c
                break
            if best is None:
                break

            # Replace the selected leaf with its 8 children.
            del leaf_by_id[best.id()]
            for ch in split(best):
                leaf_by_id[ch.id()] = ch
                heapq.heappush(heap, (-score_cell(ch), ch.id(), ch))

        # If we overshoot (splits add +7 tokens), keep the top-scored leaves.
        leaf_cells = list(leaf_by_id.values())
        leaf_cells.sort(key=lambda c: (-float(score_cell(c)), c.id()))
        leaf_cells = leaf_cells[:budget_tokens]

        # Encode with the same score function (if any) for consistency with selection.
        return encode_tokens(
            volume,
            leaf_cells,
            emb_dim=emb_dim,
            seed=seed,
            token_score_fn=self.token_score_fn,
            token_score_level_power=float(self.token_score_level_power),
        )


class NoRefineTokenizer(BaseTokenizer):
    """Coarse-only baseline: fixed depth grid without any refinement loop."""

    def __init__(self, *, level: int = 3, max_level: int = 6):
        self.level = int(level)
        self.max_level = int(max_level)

    def tokenize(self, volume: torch.Tensor, *, budget_tokens: int, emb_dim: int, seed: int) -> List[Token]:
        level = int(self.level)
        while level < int(self.max_level) and (2**level) ** 3 < int(budget_tokens):
            level += 1
        n = 2 ** int(level)
        cells = [Cell(level=int(level), ix=ix, iy=iy, iz=iz) for ix in range(n) for iy in range(n) for iz in range(n)]
        cells = sorted(cells, key=lambda c: c.id())[:budget_tokens]
        return encode_tokens(volume, cells, emb_dim=emb_dim, seed=seed)


class FixedGridTokenizer(BaseTokenizer):
    """Fixed-grid baseline implemented as uniform dyadic refinement (octree-like)."""

    def __init__(self, *, max_depth: int = 6):
        self.max_depth = max_depth

    def tokenize(self, volume: torch.Tensor, *, budget_tokens: int, emb_dim: int, seed: int) -> List[Token]:
        cells = _uniform_cells_to_budget(budget_tokens, max_depth=self.max_depth)
        return encode_tokens(volume, cells, emb_dim=emb_dim, seed=seed)


class FixedGridTokenizerScored(FixedGridTokenizer):
    """Fixed-grid tokenizer that overrides token.score via a provided score function."""

    def __init__(self, *, max_depth: int = 6, token_score_fn=None, token_score_level_power: float = 0.0):
        super().__init__(max_depth=max_depth)
        self.token_score_fn = token_score_fn
        self.token_score_level_power = float(token_score_level_power)

    def tokenize(self, volume: torch.Tensor, *, budget_tokens: int, emb_dim: int, seed: int) -> List[Token]:
        cells = _uniform_cells_to_budget(budget_tokens, max_depth=self.max_depth)
        return encode_tokens(
            volume,
            cells,
            emb_dim=emb_dim,
            seed=seed,
            token_score_fn=self.token_score_fn,
            token_score_level_power=float(self.token_score_level_power),
        )


class SliceTokenizer2D(BaseTokenizer):
    """2D slice sampling baseline (approximate).

    This keeps the 3D cell family but selects cells concentrated on a subset of z-bins.
    """

    def __init__(self, *, level: int = 3, max_level: int = 6):
        self.level = int(level)
        self.max_level = int(max_level)

    def tokenize(self, volume: torch.Tensor, *, budget_tokens: int, emb_dim: int, seed: int) -> List[Token]:
        rng = np.random.RandomState(seed)
        level = int(self.level)
        while level < int(self.max_level) and (2**level) ** 3 < int(budget_tokens):
            level += 1
        n = 2 ** int(level)
        # Pick enough z indices so that (n*n*|z_choices|) >= budget_tokens when possible.
        per_slice = n * n
        need = int(np.ceil(float(budget_tokens) / max(per_slice, 1)))
        z_count = max(1, min(n, need))
        z_choices = sorted(rng.choice(np.arange(n), size=z_count, replace=False).tolist())
        cells: List[Cell] = []
        for ix in range(n):
            for iy in range(n):
                for iz in z_choices:
                    cells.append(Cell(level=int(level), ix=ix, iy=iy, iz=iz))
        cells = sorted(cells, key=lambda c: c.id())[:budget_tokens]
        return encode_tokens(volume, cells, emb_dim=emb_dim, seed=seed)


class SliceTokenizer2p5D(BaseTokenizer):
    """2.5D slice baseline: select a thin band of z bins (contiguous), keep 3D cell ids.

    This approximates 2.5D processing while staying within the Ω=cell_id contract.
    """

    def __init__(self, *, level: int = 3, band: int = 3, max_level: int = 6):
        self.level = int(level)
        self.band = max(1, int(band))
        self.max_level = int(max_level)

    def tokenize(self, volume: torch.Tensor, *, budget_tokens: int, emb_dim: int, seed: int) -> List[Token]:
        rng = np.random.RandomState(seed)
        level = int(self.level)
        while level < int(self.max_level) and (2**level) ** 3 < int(budget_tokens):
            level += 1
        n = 2 ** int(level)

        center = int(rng.randint(0, n))
        # Expand the band so that we can produce >= budget_tokens cells when possible.
        per_slice = n * n
        need = int(np.ceil(float(budget_tokens) / max(per_slice, 1)))
        band = max(self.band, min(n, max(1, need)))
        half = band // 2
        z_choices = [(center + dz) % n for dz in range(-half, half + 1)]
        z_choices = sorted(set(z_choices))

        cells: List[Cell] = []
        for ix in range(n):
            for iy in range(n):
                for iz in z_choices:
                    cells.append(Cell(level=int(level), ix=ix, iy=iy, iz=iz))
        cells = sorted(cells, key=lambda c: c.id())[:budget_tokens]
        return encode_tokens(volume, cells, emb_dim=emb_dim, seed=seed)


class ROICropTokenizer(BaseTokenizer):
    """ROI-crop baseline: select a high-score cell and only tokenize within it."""

    def __init__(self, *, candidate_level: int = 4, roi_max_depth: int = 7, max_depth_limit: int = 8):
        self.candidate_level = candidate_level
        self.roi_max_depth = roi_max_depth
        self.max_depth_limit = int(max_depth_limit)

    def tokenize(self, volume: torch.Tensor, *, budget_tokens: int, emb_dim: int, seed: int) -> List[Token]:
        n = 2 ** self.candidate_level
        candidates = [Cell(level=self.candidate_level, ix=ix, iy=iy, iz=iz) for ix in range(n) for iy in range(n) for iz in range(n)]
        if not candidates:
            return []

        # Pick ROI cell by cheap heuristic score, tie-break by cell_id (deterministic).
        scored = [(c, _heuristic_cell_score(volume, c)) for c in candidates]
        roi_cell = sorted(scored, key=lambda x: (-x[1], x[0].id()))[0][0]

        # Deterministically refine within ROI until reaching budget.
        roi_max_depth = int(self.roi_max_depth)
        # Ensure capacity: a single cell refined to depth D yields at most 8^(D-L) leaves.
        while roi_max_depth < int(self.max_depth_limit) and (8 ** max(0, roi_max_depth - int(roi_cell.level))) < int(budget_tokens):
            roi_max_depth += 1
        cells: List[Cell] = [roi_cell]
        while len(cells) < budget_tokens:
            splittable = [c for c in cells if c.level < roi_max_depth]
            if not splittable:
                break
            c_star = sorted(splittable, key=lambda c: c.id())[0]
            cells = [c for c in cells if c.id() != c_star.id()] + split(c_star)
        cells = sorted(cells, key=lambda c: c.id())[:budget_tokens]
        tokens = encode_tokens(volume, cells, emb_dim=emb_dim, seed=seed)
        # Reindex token_id to be contiguous
        out: List[Token] = []
        for i, t in enumerate(tokens):
            out.append(Token(token_id=i, cell_id=t.cell_id, level=t.level, embedding=t.embedding, score=t.score, uncertainty=t.uncertainty))
        return out


class ROIVarianceTokenizer(BaseTokenizer):
    """ROI-like baseline: pick high-variance cells from a candidate grid."""

    def __init__(self, *, candidate_level: int = 4):
        self.candidate_level = candidate_level

    def tokenize(self, volume: torch.Tensor, *, budget_tokens: int, emb_dim: int, seed: int) -> List[Token]:
        n = 2 ** self.candidate_level
        candidates = [Cell(level=self.candidate_level, ix=ix, iy=iy, iz=iz) for ix in range(n) for iy in range(n) for iz in range(n)]

        # Select top-K candidate cells by cheap heuristic score (variance proxy).
        scored = [(c, _heuristic_cell_score(volume, c)) for c in candidates]
        top_cells = [c for (c, _) in sorted(scored, key=lambda x: (-x[1], x[0].id()))[: min(int(budget_tokens), len(scored))]]

        # If budget exceeds candidate grid capacity (e.g., 7e6 -> b_enc>512 at level=3),
        # deterministically refine within selected cells to reach the requested token count.
        cells: List[Cell] = sorted(top_cells, key=lambda c: c.id())
        max_depth = max(6, int(self.candidate_level))
        while len(cells) < int(budget_tokens):
            splittable = [c for c in cells if c.level < max_depth]
            if not splittable:
                break
            c_star = sorted(splittable, key=lambda c: c.id())[0]
            cells = [c for c in cells if c.id() != c_star.id()] + split(c_star)
            cells = sorted(cells, key=lambda c: c.id())
        cells = cells[: int(budget_tokens)]

        # Encode only the selected cells to get embeddings (keeps Ω=cell_id contract).
        tokens = encode_tokens(volume, cells, emb_dim=emb_dim, seed=seed)
        # Reindex token_id to be contiguous (important for ToyPCG citations)
        out: List[Token] = []
        for i, t in enumerate(tokens):
            out.append(Token(token_id=i, cell_id=t.cell_id, level=t.level, embedding=t.embedding, score=t.score, uncertainty=t.uncertainty))
        return out
