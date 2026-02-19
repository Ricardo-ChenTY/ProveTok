from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch


def _try_import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except Exception:  # noqa: BLE001
        nib = None  # type: ignore
    return nib


def _as_dhw(arr_xyz: np.ndarray) -> np.ndarray:
    """Convert a NIfTI array in (X,Y,Z) to our canonical (D,H,W) = (Z,Y,X)."""
    if arr_xyz.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr_xyz.shape}")
    return np.transpose(arr_xyz, (2, 1, 0))


def _affine_zyx_from_xyz(affine_xyz: np.ndarray) -> np.ndarray:
    """Convert an affine that maps (x,y,z) voxel indices to one that maps (z,y,x).

    Our tensors are indexed as (D,H,W) = (z,y,x). This returns an affine that can
    be applied to [z, y, x, 1] to obtain world (RAS) coordinates.
    """
    A = np.asarray(affine_xyz, dtype=np.float64)
    if A.shape != (4, 4):
        raise ValueError(f"Expected affine (4,4), got shape={A.shape}")
    P = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return A @ P


def load_volume(
    path: Optional[str] = None,
    *,
    seed: int = 0,
    shape: Tuple[int, int, int] = (64, 64, 64),
) -> torch.Tensor:
    """Load a 3D volume (D,H,W) from disk, or fall back to a deterministic synthetic volume.

    - For NIfTI inputs, we reorient to closest canonical (RAS+) when nibabel is
      available, then return the data as (D,H,W)=(Z,Y,X).
    - For synthetic volumes (path is None/missing), `shape` controls the output
      size.
    """
    if path is None or not str(path):
        rng = np.random.RandomState(int(seed))
        vol = rng.randn(*tuple(int(x) for x in shape)).astype(np.float32)
        return torch.from_numpy(vol)

    p = Path(str(path))
    if not p.exists():
        rng = np.random.RandomState(int(seed) + 1337)
        vol = rng.randn(*tuple(int(x) for x in shape)).astype(np.float32)
        return torch.from_numpy(vol)

    suf = "".join(p.suffixes).lower()
    if suf.endswith(".npy"):
        arr = np.load(str(p))
        return torch.from_numpy(arr.astype(np.float32))
    if suf.endswith(".npz"):
        z = np.load(str(p))
        if "volume" in z:
            arr = z["volume"]
        else:
            arr = z[z.files[0]]
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))
    if suf.endswith(".nii") or suf.endswith(".nii.gz"):
        nib = _try_import_nibabel()
        if nib is None:
            raise RuntimeError("Loading NIfTI requires nibabel. Install via `pip install nibabel`.")
        img = nib.load(str(p))
        # Reorient to canonical RAS+ (pp.md laterality auditing assumes a stable x-axis).
        img = nib.as_closest_canonical(img)
        arr = np.asarray(img.dataobj)
        arr = np.asarray(arr, dtype=np.float32)
        arr = _as_dhw(arr)
        return torch.from_numpy(arr)

    raise ValueError(f"Unsupported volume format: {p}")


def load_mask(path: str) -> Any:
    """Load a lesion mask.

    Returns:
        - np.ndarray bool with shape (D,H,W), or
        - np.ndarray bool with shape (K,D,H,W) for stacked masks.

    Notes:
        - For NIfTI inputs, we reorient to closest canonical (RAS+) when nibabel
          is available, then return masks as (D,H,W)=(Z,Y,X).
    """
    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(str(p))
    suf = "".join(p.suffixes).lower()
    if suf.endswith(".npy"):
        arr = np.load(str(p))
        return np.asarray(arr).astype(bool)
    if suf.endswith(".npz"):
        z = np.load(str(p))
        arr = z[z.files[0]]
        return np.asarray(arr).astype(bool)
    if suf.endswith(".nii") or suf.endswith(".nii.gz"):
        nib = _try_import_nibabel()
        if nib is None:
            raise RuntimeError("Loading NIfTI requires nibabel. Install via `pip install nibabel`.")
        img = nib.load(str(p))
        img = nib.as_closest_canonical(img)
        arr = np.asarray(img.dataobj)
        arr = np.asarray(arr, dtype=np.float32)
        arr = _as_dhw(arr)
        return (arr > 0.5)
    raise ValueError(f"Unsupported mask format: {p}")


def load_affine(path: str) -> Optional[np.ndarray]:
    """Load an affine matrix matching our (D,H,W) tensor indexing.

    For NIfTI inputs, we return an affine that maps (z,y,x) voxel indices from
    the returned tensor to world (RAS) coordinates.

    For `.npz` inputs, if an `affine_zyx` array is present, we return it.
    """
    p = Path(str(path))
    suf = "".join(p.suffixes).lower()
    if not p.exists():
        return None

    if suf.endswith(".npz"):
        try:
            z = np.load(str(p))
            if "affine_zyx" not in z:
                return None
            aff = np.asarray(z["affine_zyx"], dtype=np.float64)
            if aff.shape != (4, 4):
                return None
            return aff
        except Exception:
            return None

    if suf.endswith(".nii") or suf.endswith(".nii.gz"):
        nib = _try_import_nibabel()
        if nib is None:
            return None
        img = nib.load(str(p))
        img = nib.as_closest_canonical(img)
        aff = np.asarray(getattr(img, "affine", None))
        if aff is None or aff.shape != (4, 4):
            return None
        return _affine_zyx_from_xyz(aff)

    return None


def load_volume_and_affine(
    path: Optional[str],
    *,
    seed: int = 0,
    shape: Tuple[int, int, int] = (64, 64, 64),
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """Load a volume and the matching affine (if available) in one pass."""
    if path is None or not str(path):
        return load_volume(None, seed=seed, shape=shape), None

    p = Path(str(path))
    suf = "".join(p.suffixes).lower()
    if not p.exists():
        return load_volume(None, seed=seed, shape=shape), None

    if suf.endswith(".npz"):
        z = np.load(str(p))
        if "volume" in z:
            arr = z["volume"]
        else:
            arr = z[z.files[0]]
        vol = torch.from_numpy(np.asarray(arr, dtype=np.float32))

        aff = None
        if "affine_zyx" in z:
            try:
                a = np.asarray(z["affine_zyx"], dtype=np.float64)
                if a.shape == (4, 4):
                    aff = a
            except Exception:
                aff = None
        return vol, aff

    if suf.endswith(".nii") or suf.endswith(".nii.gz"):
        nib = _try_import_nibabel()
        if nib is None:
            raise RuntimeError("Loading NIfTI requires nibabel. Install via `pip install nibabel`.")
        img = nib.load(str(p))
        img = nib.as_closest_canonical(img)
        arr = np.asarray(img.dataobj)
        arr = np.asarray(arr, dtype=np.float32)
        arr = _as_dhw(arr)
        vol = torch.from_numpy(arr)
        aff = np.asarray(getattr(img, "affine", None))
        aff_zyx = _affine_zyx_from_xyz(aff) if aff is not None and aff.shape == (4, 4) else None
        return vol, aff_zyx

    # Fallback: formats without affine.
    return load_volume(str(p), seed=seed, shape=shape), None
