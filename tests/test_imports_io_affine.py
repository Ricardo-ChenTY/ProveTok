from __future__ import annotations

import subprocess
import sys

import numpy as np


def test_imports_no_circular() -> None:
    import provetok.data  # noqa: F401
    import provetok.experiments  # noqa: F401
    from provetok.pcg import Llama2PCGConfig  # noqa: F401


def test_run_baselines_help() -> None:
    p = subprocess.run(
        [sys.executable, "-m", "provetok.experiments.run_baselines", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    assert "run_baselines" in (p.stdout + p.stderr)


def test_load_volume_shape_param() -> None:
    from provetok.data.io import load_volume

    vol = load_volume(None, seed=0, shape=(5, 6, 7))
    assert tuple(int(x) for x in vol.shape) == (5, 6, 7)


def test_nifti_dhw_and_affine_consistency(tmp_path) -> None:
    import nibabel as nib

    from provetok.bet.tokenize import TokenEncoder
    from provetok.data.io import load_volume_and_affine
    from provetok.grid.cells import root_cell

    # Create a tiny canonical NIfTI (X,Y,Z) with a simple affine.
    arr = np.zeros((4, 5, 6), dtype=np.float32)  # (X,Y,Z)
    arr[1, 2, 3] = 1.0
    affine_xyz = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    img = nib.Nifti1Image(arr, affine_xyz)
    p = tmp_path / "toy.nii.gz"
    nib.save(img, str(p))

    vol, aff_zyx = load_volume_and_affine(str(p))
    assert tuple(int(x) for x in vol.shape) == (6, 5, 4)  # (D,H,W)=(Z,Y,X)
    assert aff_zyx is not None and tuple(int(x) for x in aff_zyx.shape) == (4, 4)

    # Consistency check: world coord of (x=1,y=2,z=3) equals world coord of (z=3,y=2,x=1)
    # under the returned (z,y,x) affine.
    v1 = affine_xyz @ np.array([1.0, 2.0, 3.0, 1.0], dtype=np.float64)
    v2 = np.asarray(aff_zyx, dtype=np.float64) @ np.array([3.0, 2.0, 1.0, 1.0], dtype=np.float64)
    assert np.allclose(v1, v2)

    # Token mm geometry should be populated when we pass the affine.
    enc = TokenEncoder(volume=vol, emb_dim=8, seed=0, affine_zyx=aff_zyx)
    toks = enc.encode([root_cell()])
    assert len(toks) == 1
    assert toks[0].bounds_mm is not None
    assert toks[0].center_mm is not None


def test_npz_affine_roundtrip(tmp_path) -> None:
    from provetok.data.io import load_volume_and_affine

    vol = np.zeros((3, 4, 5), dtype=np.float32)
    aff = np.array(
        [
            [1.0, 0.0, 0.0, 7.0],
            [0.0, 2.0, 0.0, 8.0],
            [0.0, 0.0, 3.0, 9.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    p = tmp_path / "toy.npz"
    np.savez_compressed(str(p), volume=vol, affine_zyx=aff)

    v2, a2 = load_volume_and_affine(str(p))
    assert tuple(int(x) for x in v2.shape) == (3, 4, 5)
    assert a2 is not None
    assert np.allclose(np.asarray(a2), aff)
