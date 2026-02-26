from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..grid.cells import Cell, cell_bounds, cell_bounds_int, cell_center_voxel, cell_stable_id
from ..types import Token


def _stable_int_hash(text: str) -> int:
    """Stable 32-bit hash for determinism across Python processes."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="big", signed=False)


# Global projection matrix cache for the toy embedder.
#
# Important: learned heads (e.g. lesionness) assume the embedding function is
# *stable* across runs/seeds. In real systems, encoder weights are fixed at
# inference time; so the toy embedder must not depend on experiment seeds.
_TOY_PROJ_CACHE: Dict[int, torch.Tensor] = {}


def _toy_proj_matrix(emb_dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    emb_dim = int(emb_dim)
    W = _TOY_PROJ_CACHE.get(emb_dim)
    if W is None:
        g = torch.Generator(device="cpu")
        # Fixed seed -> stable random feature mapping across all runs.
        g.manual_seed(1337 + 17 * emb_dim)
        W = torch.randn((emb_dim, 4), generator=g, device="cpu", dtype=torch.float32) / float(np.sqrt(4))
        _TOY_PROJ_CACHE[emb_dim] = W
    return W.to(device=device, dtype=dtype)


# TODO: replace with a real 3D encoder feature map + ROI pooling.
def _toy_embed_patch(vol: torch.Tensor, slc: Tuple[slice, slice, slice], emb_dim: int, seed: int) -> torch.Tensor:
    patch = vol[slc[0], slc[1], slc[2]]
    mean = patch.mean()
    std = patch.std(unbiased=False)
    mn = patch.min()
    mx = patch.max()
    stats = torch.stack([mean, std, mn, mx]).float()

    # Keep the toy projection stable across seeds/runs so downstream learned
    # modules can be trained once and reused (mirrors real encoder behavior).
    W = _toy_proj_matrix(int(emb_dim), device=stats.device, dtype=stats.dtype)
    emb = torch.tanh(W @ stats.to(dtype=W.dtype))
    return emb


def _cell_bounds_mm(
    bounds_voxel: Tuple[int, int, int, int, int, int],
    center_voxel: Tuple[float, float, float],
    affine_zyx: np.ndarray,
) -> Tuple[Optional[Tuple[float, float, float, float, float, float]], Optional[Tuple[float, float, float]]]:
    """Compute (bounds_mm, center_mm) in (z,y,x) order given an affine.

    `affine_zyx` maps (z,y,x,1) voxel indices from our tensors to world (RAS)
    coordinates (x,y,z). We store mm geometry in the same axis order as voxel
    bounds: (z0,z1,y0,y1,x0,x1) and (cz,cy,cx).
    """
    try:
        A = np.asarray(affine_zyx, dtype=np.float64)
        if A.shape != (4, 4):
            return None, None

        z0, z1, y0, y1, x0, x1 = (float(v) for v in bounds_voxel)
        corners = np.array(
            [
                [z0, y0, x0],
                [z0, y0, x1],
                [z0, y1, x0],
                [z0, y1, x1],
                [z1, y0, x0],
                [z1, y0, x1],
                [z1, y1, x0],
                [z1, y1, x1],
            ],
            dtype=np.float64,
        )
        hom = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float64)], axis=1)
        world = hom @ A.T  # (N,4) in world (x,y,z,1)
        xs = world[:, 0]
        ys = world[:, 1]
        zs = world[:, 2]

        bounds_mm = (
            float(zs.min()),
            float(zs.max()),
            float(ys.min()),
            float(ys.max()),
            float(xs.min()),
            float(xs.max()),
        )

        cz, cy, cx = (float(v) for v in center_voxel)
        center_h = np.array([cz, cy, cx, 1.0], dtype=np.float64)
        center_world = center_h @ A.T
        center_mm = (float(center_world[2]), float(center_world[1]), float(center_world[0]))
        return bounds_mm, center_mm
    except Exception:  # noqa: BLE001
        return None, None


@dataclass
class _CachedTokenFields:
    embedding: torch.Tensor
    score: float
    uncertainty: float
    level: int
    bounds_voxel: Tuple[int, int, int, int, int, int]
    center_voxel: Tuple[float, float, float]
    bounds_mm: Optional[Tuple[float, float, float, float, float, float]] = None
    center_mm: Optional[Tuple[float, float, float]] = None
    anatomy_label: Optional[str] = None


class TokenEncoder:
    """Cache-aware token encoder.

    - If `encoder` is provided, it runs the encoder once and pools per-cell embeddings.
    - Otherwise it falls back to deterministic toy patch embedding.
    - Caches per-cell outputs to avoid recomputation during BET refinement.

    If `affine_zyx` is provided, mm-space geometry is populated on each token.
    """

    def __init__(
        self,
        *,
        volume: torch.Tensor,
        emb_dim: int = 32,
        seed: int = 0,
        encoder: Optional[Any] = None,
        affine_zyx: Optional[np.ndarray] = None,
        anatomy_labels: Optional[np.ndarray] = None,
        anatomy_label_map: Optional[Dict[int, str]] = None,
    ):
        self.volume = volume
        self.emb_dim = int(emb_dim)
        self.seed = int(seed)
        self.encoder = encoder
        self.affine_zyx = np.asarray(affine_zyx, dtype=np.float64) if affine_zyx is not None else None
        self.anatomy_labels = np.asarray(anatomy_labels) if anatomy_labels is not None else None
        self.anatomy_label_map = {int(k): str(v) for k, v in (anatomy_label_map or {}).items()}

        self._cache: Dict[str, _CachedTokenFields] = {}

        self._features: Optional[torch.Tensor] = None
        self._vol_shape: Tuple[int, int, int] = tuple(volume.shape)  # (D,H,W)
        # Scoring normalization:
        # - Some datasets store CT volumes as non-HU uint16-like values (e.g., 0..3000).
        # - Our toy ROI heuristics operate on variance; without normalization it saturates.
        # We infer a simple HU-ish mapping (optional -1024 shift) and scale to ~[-1,1].
        self._score_shift = 0.0
        self._score_div = 1.0
        try:
            vmin = float(self.volume.min().item())
            vmax = float(self.volume.max().item())
            if vmin >= 0.0 and vmax > 50.0:
                self._score_shift = -1024.0
                vmin = vmin + self._score_shift
                vmax = vmax + self._score_shift
            if max(abs(vmin), abs(vmax)) > 10.0:
                self._score_div = 1000.0
        except Exception:
            self._score_shift = 0.0
            self._score_div = 1.0

        self._feat_shape: Optional[Tuple[int, int, int]] = None

        if self.encoder is not None:
            self._build_feature_cache()

    def _build_feature_cache(self) -> None:
        encoder = self.encoder
        if encoder is None:
            return

        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = self.volume.device

        encoder.eval()
        with torch.no_grad():
            x = self.volume
            if x.dim() == 3:
                x = x.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
            elif x.dim() == 4:
                x = x.unsqueeze(0)  # (1,C,D,H,W)
            x = x.to(device)
            feats = encoder(x)
            if feats.dim() != 5:
                raise ValueError(
                    f"encoder.forward must return a feature map (B,C,D,H,W), got shape={tuple(feats.shape)}"
                )
            self._features = feats.detach()
            self._feat_shape = tuple(self._features.shape[2:])

    def _map_bounds(
        self,
        bounds: Tuple[slice, slice, slice],
        vol_shape: Tuple[int, int, int],
        feat_shape: Tuple[int, int, int],
    ) -> Tuple[slice, slice, slice]:
        mapped = []
        for slc, vs, fs in zip(bounds, vol_shape, feat_shape):
            ratio = fs / vs
            start = int((slc.start or 0) * ratio)
            stop = int((slc.stop or vs) * ratio)
            if stop <= start:
                stop = start + 1
            mapped.append(slice(start, min(stop, fs)))
        return tuple(mapped)

    def _encode_cell(self, cell: Cell) -> _CachedTokenFields:
        cid = cell.id()
        bounds = cell_bounds(cell, shape=self._vol_shape)
        bounds_int = cell_bounds_int(cell, shape=self._vol_shape)
        center = cell_center_voxel(bounds_int)

        bounds_mm: Optional[Tuple[float, float, float, float, float, float]] = None
        center_mm: Optional[Tuple[float, float, float]] = None
        anatomy_label: Optional[str] = None
        if self.affine_zyx is not None:
            bounds_mm, center_mm = _cell_bounds_mm(bounds_int, center, self.affine_zyx)
        if self.anatomy_labels is not None and self.anatomy_labels.ndim == 3:
            try:
                cz = int(max(0, min(int(round(center[0])), int(self.anatomy_labels.shape[0]) - 1)))
                cy = int(max(0, min(int(round(center[1])), int(self.anatomy_labels.shape[1]) - 1)))
                cx = int(max(0, min(int(round(center[2])), int(self.anatomy_labels.shape[2]) - 1)))
                label_id = int(self.anatomy_labels[cz, cy, cx])
                anatomy_label = self.anatomy_label_map.get(label_id, str(label_id))
            except Exception:
                anatomy_label = None

        patch = self.volume[bounds[0], bounds[1], bounds[2]]
        if patch.numel() > 0:
            p = patch.float()
            if float(self._score_shift) != 0.0:
                p = p + float(self._score_shift)
            if float(self._score_div) != 1.0:
                p = p.clamp(min=-1000.0, max=1000.0) / float(self._score_div)
            else:
                p = p.clamp(min=-1.0, max=1.0)
            var = float(p.var(unbiased=False).item())
        else:
            var = 0.0
        score = float(1.0 / (1.0 + np.exp(-3.0 * (var - 0.5))))
        uncertainty = float(1.0 - score)

        if self.encoder is None or self._features is None or self._feat_shape is None:
            emb = _toy_embed_patch(
                self.volume,
                bounds,
                emb_dim=self.emb_dim,
                seed=0,
            )
            return _CachedTokenFields(
                embedding=emb,
                score=score,
                uncertainty=uncertainty,
                level=cell.level,
                bounds_voxel=bounds_int,
                center_voxel=center,
                bounds_mm=bounds_mm,
                center_mm=center_mm,
                anatomy_label=anatomy_label,
            )

        # Encoder-backed embedding: pool region from cached feature map.
        feat_bounds = self._map_bounds(bounds, self._vol_shape, self._feat_shape)
        region = self._features[:, :, feat_bounds[0], feat_bounds[1], feat_bounds[2]]
        pooled = region.mean(dim=(0, 2, 3, 4)).float()  # (C,)

        if pooled.numel() != self.emb_dim:
            # Deterministic projection to emb_dim to keep downstream contracts stable.
            g = torch.Generator(device=pooled.device)
            g.manual_seed(self.seed + _stable_int_hash(cid) + 424242)
            W = torch.randn((self.emb_dim, pooled.numel()), generator=g, device=pooled.device) / float(
                np.sqrt(max(pooled.numel(), 1))
            )
            pooled = torch.tanh(W @ pooled)

        emb = pooled.detach().cpu()
        return _CachedTokenFields(
            embedding=emb,
            score=score,
            uncertainty=uncertainty,
            level=cell.level,
            bounds_voxel=bounds_int,
            center_voxel=center,
            bounds_mm=bounds_mm,
            center_mm=center_mm,
            anatomy_label=anatomy_label,
        )

    def encode(
        self,
        cells: List[Cell],
        *,
        token_score_fn: Optional[Any] = None,
        token_score_level_power: float = 0.0,
    ) -> List[Token]:
        for c in cells:
            cid = c.id()
            if cid not in self._cache:
                self._cache[cid] = self._encode_cell(c)

        ordered_cells = sorted(cells, key=lambda x: x.id())
        fields = [self._cache[c.id()] for c in ordered_cells]

        scores: List[float]
        if token_score_fn is None:
            scores = [float(f.score) for f in fields]
        else:
            emb = torch.stack([f.embedding for f in fields], dim=0)
            scores_t = token_score_fn(emb)
            if not isinstance(scores_t, torch.Tensor):
                scores_t = torch.tensor(scores_t)  # type: ignore[arg-type]
            scores = scores_t.detach().cpu().flatten().tolist()
            if len(scores) != len(fields):
                raise ValueError(f"token_score_fn must return N scores, got {len(scores)} for N={len(fields)}")

        if float(token_score_level_power) != 0.0:
            p = float(token_score_level_power)
            scores = [float(s) * float((1 + int(f.level)) ** p) for s, f in zip(scores, fields)]

        tokens: List[Token] = []
        for c, f, s in zip(ordered_cells, fields, scores):
            # Stable token id derived from octree path / Morton code (pp.md §3.2).
            token_id = int(cell_stable_id(c))
            tokens.append(
                Token(
                    token_id=token_id,
                    cell_id=c.id(),
                    level=f.level,
                    embedding=f.embedding,
                    score=float(s),
                    uncertainty=f.uncertainty,
                    # Stable reference string for pp.md-style citations.
                    ref=str(token_id),
                    bounds_voxel=tuple(int(x) for x in f.bounds_voxel),
                    center_voxel=tuple(float(x) for x in f.center_voxel),
                    bounds_mm=f.bounds_mm,
                    center_mm=f.center_mm,
                    anatomy_label=f.anatomy_label,
                )
            )
        return tokens


def encode_tokens(
    volume: torch.Tensor,
    cells: List[Cell],
    emb_dim: int = 32,
    seed: int = 0,
    *,
    encoder: Optional[Any] = None,
    affine_zyx: Optional[np.ndarray] = None,
    anatomy_labels: Optional[np.ndarray] = None,
    anatomy_label_map: Optional[Dict[int, str]] = None,
    token_score_fn: Optional[Any] = None,
    token_score_level_power: float = 0.0,
) -> List[Token]:
    """Encode cells into evidence tokens.

    If `encoder` is provided, uses a cached encoder feature map; otherwise uses
    deterministic toy patch embedding.
    """
    return TokenEncoder(
        volume=volume,
        emb_dim=emb_dim,
        seed=seed,
        encoder=encoder,
        affine_zyx=affine_zyx,
        anatomy_labels=anatomy_labels,
        anatomy_label_map=anatomy_label_map,
    ).encode(
        cells,
        token_score_fn=token_score_fn,
        token_score_level_power=float(token_score_level_power),
    )
