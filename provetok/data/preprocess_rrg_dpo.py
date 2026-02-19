from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RRGDPOPreprocessSpec:
    """RRG-DPO comparable preprocessing (pp.md §6.2).

    Notes:
    - This repo uses tensors indexed as (D,H,W) = (z,y,x).
    - Spacing is therefore given as (sz,sy,sx) in mm.
    """

    target_spacing_zyx: Tuple[float, float, float] = (1.5, 0.75, 0.75)
    target_shape_dhw: Tuple[int, int, int] = (240, 480, 480)


def infer_spacing_zyx(affine_zyx: np.ndarray) -> Tuple[float, float, float]:
    """Infer voxel spacing (sz,sy,sx) in mm from an affine that maps (z,y,x)->world."""
    A = np.asarray(affine_zyx, dtype=np.float64)
    if A.shape != (4, 4):
        raise ValueError(f"affine_zyx must be (4,4), got shape={A.shape}")
    # Column i gives world displacement for +1 step in axis i (z,y,x).
    sz = float(np.linalg.norm(A[:3, 0]))
    sy = float(np.linalg.norm(A[:3, 1]))
    sx = float(np.linalg.norm(A[:3, 2]))
    return (sz, sy, sx)


def _with_target_spacing(affine_zyx: np.ndarray, target_spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    """Return an affine with the same axis directions but target voxel spacing."""
    A = np.asarray(affine_zyx, dtype=np.float64).copy()
    if A.shape != (4, 4):
        raise ValueError(f"affine_zyx must be (4,4), got shape={A.shape}")
    cur = infer_spacing_zyx(A)
    for i, (s_cur, s_tgt) in enumerate(zip(cur, target_spacing_zyx)):
        if s_cur <= 0:
            continue
        A[:3, i] = A[:3, i] * (float(s_tgt) / float(s_cur))
    return A


def resample_dhw(
    vol: torch.Tensor,
    *,
    affine_zyx: Optional[np.ndarray],
    target_spacing_zyx: Tuple[float, float, float],
    mode: str,
) -> Tuple[torch.Tensor, Optional[np.ndarray], Tuple[float, float, float]]:
    """Resample a (D,H,W) tensor to a target spacing.

    Returns:
        (vol_resampled, affine_resampled, orig_spacing_zyx)
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected vol (D,H,W), got shape={tuple(vol.shape)}")
    if affine_zyx is None:
        raise ValueError("affine_zyx is required for spacing-based resampling")

    orig_spacing = infer_spacing_zyx(affine_zyx)
    sz0, sy0, sx0 = (float(x) for x in orig_spacing)
    sz1, sy1, sx1 = (float(x) for x in target_spacing_zyx)
    D0, H0, W0 = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))

    def _new_len(n0: int, s0: float, s1: float) -> int:
        if n0 <= 0:
            return 1
        if s0 <= 0 or s1 <= 0:
            return int(n0)
        return max(1, int(round(float(n0) * float(s0) / float(s1))))

    D1 = _new_len(D0, sz0, sz1)
    H1 = _new_len(H0, sy0, sy1)
    W1 = _new_len(W0, sx0, sx1)

    if (D1, H1, W1) == (D0, H0, W0):
        return vol, _with_target_spacing(affine_zyx, target_spacing_zyx), orig_spacing

    x = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    if mode == "trilinear":
        y = F.interpolate(x, size=(D1, H1, W1), mode="trilinear", align_corners=False)
    else:
        y = F.interpolate(x, size=(D1, H1, W1), mode=str(mode))
    out = y[0, 0]
    aff1 = _with_target_spacing(affine_zyx, target_spacing_zyx)
    return out, aff1, orig_spacing


def center_crop_pad_dhw(
    vol: torch.Tensor,
    *,
    affine_zyx: Optional[np.ndarray],
    target_shape_dhw: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, Optional[np.ndarray], Tuple[int, int, int]]:
    """Center crop/pad a (D,H,W) tensor to target_shape_dhw, updating affine translation."""
    if vol.ndim != 3:
        raise ValueError(f"Expected vol (D,H,W), got shape={tuple(vol.shape)}")
    D0, H0, W0 = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))
    Dt, Ht, Wt = (int(target_shape_dhw[0]), int(target_shape_dhw[1]), int(target_shape_dhw[2]))

    def _crop_start(n0: int, nt: int) -> int:
        return max(0, (n0 - nt) // 2)

    def _pad_before(n0: int, nt: int) -> int:
        return max(0, (nt - n0) // 2)

    z0 = _crop_start(D0, Dt)
    y0 = _crop_start(H0, Ht)
    x0 = _crop_start(W0, Wt)
    vol2 = vol[z0 : z0 + min(D0, Dt), y0 : y0 + min(H0, Ht), x0 : x0 + min(W0, Wt)]

    D1, H1, W1 = (int(vol2.shape[0]), int(vol2.shape[1]), int(vol2.shape[2]))
    pz0 = _pad_before(D1, Dt)
    py0 = _pad_before(H1, Ht)
    px0 = _pad_before(W1, Wt)
    pz1 = max(0, Dt - D1 - pz0)
    py1 = max(0, Ht - H1 - py0)
    px1 = max(0, Wt - W1 - px0)

    if any(int(x) > 0 for x in (pz0, pz1, py0, py1, px0, px1)):
        vol2 = F.pad(vol2, (px0, px1, py0, py1, pz0, pz1), mode="constant", value=float(pad_value))

    if affine_zyx is None:
        return vol2, None, (D0, H0, W0)

    offset = np.array([float(z0 - pz0), float(y0 - py0), float(x0 - px0)], dtype=np.float64)
    A = np.asarray(affine_zyx, dtype=np.float64).copy()
    if A.shape != (4, 4):
        raise ValueError(f"affine_zyx must be (4,4), got shape={A.shape}")
    A[:3, 3] = A[:3, 3] + A[:3, :3] @ offset
    return vol2, A, (D0, H0, W0)


def preprocess_rrg_dpo(
    vol: torch.Tensor,
    *,
    affine_zyx: Optional[np.ndarray],
    spec: RRGDPOPreprocessSpec = RRGDPOPreprocessSpec(),
    is_mask: bool = False,
) -> Tuple[torch.Tensor, Optional[np.ndarray], dict]:
    """Apply RRG-DPO comparable preprocessing: resample spacing + center crop/pad.

    Returns:
        (vol_out, affine_out, meta)

    Meta includes orig_spacing/shape and target spec.
    """
    mode = "nearest" if bool(is_mask) else "trilinear"
    vol1, aff1, orig_spacing = resample_dhw(
        vol,
        affine_zyx=affine_zyx,
        target_spacing_zyx=tuple(float(x) for x in spec.target_spacing_zyx),
        mode=mode,
    )
    vol2, aff2, orig_shape = center_crop_pad_dhw(
        vol1,
        affine_zyx=aff1,
        target_shape_dhw=tuple(int(x) for x in spec.target_shape_dhw),
        pad_value=0.0,
    )
    meta = {
        "orig_shape_dhw": [int(x) for x in orig_shape],
        "orig_spacing_zyx": [float(x) for x in orig_spacing],
        "target_shape_dhw": [int(x) for x in spec.target_shape_dhw],
        "target_spacing_zyx": [float(x) for x in spec.target_spacing_zyx],
    }
    return vol2, aff2, meta
