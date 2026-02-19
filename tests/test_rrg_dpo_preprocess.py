from __future__ import annotations

import numpy as np
import torch


def _affine_diag_zyx(sz: float, sy: float, sx: float, tz: float = 0.0, ty: float = 0.0, tx: float = 0.0) -> np.ndarray:
    # Affine that maps (z,y,x,1) -> world with axis-aligned spacing.
    return np.array(
        [
            [sz, 0.0, 0.0, tz],
            [0.0, sy, 0.0, ty],
            [0.0, 0.0, sx, tx],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_infer_spacing_zyx() -> None:
    from provetok.data.preprocess_rrg_dpo import infer_spacing_zyx

    aff = _affine_diag_zyx(4.0, 3.0, 2.0)
    sz, sy, sx = infer_spacing_zyx(aff)
    assert abs(sz - 4.0) < 1e-6
    assert abs(sy - 3.0) < 1e-6
    assert abs(sx - 2.0) < 1e-6


def test_resample_updates_shape_and_spacing() -> None:
    from provetok.data.preprocess_rrg_dpo import infer_spacing_zyx, resample_dhw

    vol = torch.zeros((10, 20, 30), dtype=torch.float32)
    aff0 = _affine_diag_zyx(2.0, 2.0, 2.0)
    vol2, aff2, orig_spacing = resample_dhw(
        vol,
        affine_zyx=aff0,
        target_spacing_zyx=(1.0, 1.0, 1.0),
        mode="trilinear",
    )
    assert tuple(int(x) for x in orig_spacing) == (2, 2, 2)
    assert tuple(int(x) for x in vol2.shape) == (20, 40, 60)
    assert aff2 is not None
    sz, sy, sx = infer_spacing_zyx(np.asarray(aff2))
    assert abs(sz - 1.0) < 1e-6
    assert abs(sy - 1.0) < 1e-6
    assert abs(sx - 1.0) < 1e-6


def test_center_crop_updates_translation() -> None:
    from provetok.data.preprocess_rrg_dpo import center_crop_pad_dhw

    vol = torch.zeros((10, 10, 10), dtype=torch.float32)
    aff0 = _affine_diag_zyx(1.0, 1.0, 1.0)
    vol2, aff2, orig_shape = center_crop_pad_dhw(vol, affine_zyx=aff0, target_shape_dhw=(6, 8, 10), pad_value=0.0)
    assert tuple(int(x) for x in orig_shape) == (10, 10, 10)
    assert tuple(int(x) for x in vol2.shape) == (6, 8, 10)
    assert aff2 is not None
    # Expected crop starts: z0=2, y0=1, x0=0 -> translation shift +[2,1,0].
    assert np.allclose(np.asarray(aff2)[:3, 3], np.array([2.0, 1.0, 0.0], dtype=np.float64))


def test_center_pad_updates_translation() -> None:
    from provetok.data.preprocess_rrg_dpo import center_crop_pad_dhw

    vol = torch.zeros((6, 6, 6), dtype=torch.float32)
    aff0 = _affine_diag_zyx(1.0, 1.0, 1.0)
    vol2, aff2, _ = center_crop_pad_dhw(vol, affine_zyx=aff0, target_shape_dhw=(10, 8, 6), pad_value=0.0)
    assert tuple(int(x) for x in vol2.shape) == (10, 8, 6)
    assert aff2 is not None
    # Expected pad_before: z=2,y=1,x=0 -> translation shift -[2,1,0].
    assert np.allclose(np.asarray(aff2)[:3, 3], np.array([-2.0, -1.0, 0.0], dtype=np.float64))


def test_preprocess_rrg_dpo_meta_and_shape() -> None:
    from provetok.data.preprocess_rrg_dpo import RRGDPOPreprocessSpec, preprocess_rrg_dpo

    vol = torch.zeros((10, 10, 10), dtype=torch.float32)
    aff0 = _affine_diag_zyx(1.5, 0.75, 0.75)
    spec = RRGDPOPreprocessSpec(target_spacing_zyx=(1.5, 0.75, 0.75), target_shape_dhw=(12, 8, 6))
    vol2, aff2, meta = preprocess_rrg_dpo(vol, affine_zyx=aff0, spec=spec, is_mask=False)
    assert tuple(int(x) for x in vol2.shape) == (12, 8, 6)
    assert aff2 is not None
    assert meta["orig_shape_dhw"] == [10, 10, 10]
    assert meta["target_shape_dhw"] == [12, 8, 6]
