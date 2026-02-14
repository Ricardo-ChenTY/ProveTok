from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _try_import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except Exception:  # noqa: BLE001
        nib = None  # type: ignore
    return nib


def load_volume(path: Optional[str] = None, *, seed: int = 0) -> torch.Tensor:
    """Load a 3D volume (D,H,W) from disk, or fall back to a deterministic synthetic volume."""
    if path is None or not str(path):
        rng = np.random.RandomState(int(seed))
        vol = rng.randn(64, 64, 64).astype(np.float32)
        return torch.from_numpy(vol)

    p = Path(str(path))
    if not p.exists():
        # Deterministic fallback to keep scripts usable without data.
        rng = np.random.RandomState(int(seed) + 1337)
        vol = rng.randn(64, 64, 64).astype(np.float32)
        return torch.from_numpy(vol)

    suf = "".join(p.suffixes).lower()
    if suf.endswith(".npy"):
        arr = np.load(str(p))
        return torch.from_numpy(arr.astype(np.float32))
    if suf.endswith(".npz"):
        z = np.load(str(p))
        # Common keys: "volume", "arr_0"
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
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
        # NIfTI is commonly (H,W,D) or (X,Y,Z); convert to (D,H,W) by taking last axis as D.
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI, got shape={arr.shape}")
        # Heuristic: treat axis-2 as depth and transpose to (D,H,W).
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    raise ValueError(f"Unsupported volume format: {p}")


def load_mask(path: str) -> Any:
    """Load a lesion mask.

    Returns:
        - np.ndarray bool with shape (D,H,W), or
        - np.ndarray bool with shape (K,D,H,W) for stacked masks.
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
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D mask NIfTI, got shape={arr.shape}")
        arr = np.transpose(arr, (2, 0, 1))
        return (arr > 0.5)
    raise ValueError(f"Unsupported mask format: {p}")


def load_affine(path: str) -> Optional[np.ndarray]:
    """Load an affine matrix from a NIfTI file, if available."""
    p = Path(str(path))
    suf = "".join(p.suffixes).lower()
    if not p.exists():
        return None
    if suf.endswith(".nii") or suf.endswith(".nii.gz"):
        nib = _try_import_nibabel()
        if nib is None:
            return None
        img = nib.load(str(p))
        return np.asarray(getattr(img, "affine", None))
    return None
