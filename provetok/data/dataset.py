from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..types import Frame
from .frame_extractor import FrameExtractor
from .io import load_mask, load_volume, load_volume_and_affine
from .manifest_schema import ManifestRecord, get_record_mask_path, load_manifest


def _resize_volume(vol: torch.Tensor, *, resize_shape: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(int(x) for x in vol.shape) == tuple(int(x) for x in resize_shape):
        return vol
    x = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    y = F.interpolate(x, size=tuple(int(x) for x in resize_shape), mode="trilinear", align_corners=False)
    return y[0, 0]


def _resize_mask(mask: np.ndarray, *, resize_shape: Tuple[int, int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim not in (3, 4):
        return mask
    if mask.ndim == 4:
        # (K,D,H,W)
        out = []
        for k in range(mask.shape[0]):
            out.append(_resize_mask(mask[k], resize_shape=resize_shape))
        return np.stack(out, axis=0)
    if tuple(int(x) for x in mask.shape) == tuple(int(x) for x in resize_shape):
        return mask.astype(bool)
    x = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    tgt = tuple(int(x) for x in resize_shape)
    y = F.interpolate(x, size=tgt, mode="nearest")
    return (y[0, 0].numpy() > 0.5)


def _synthetic_volume_and_masks(*, vol_shape: Tuple[int, int, int], n_lesions: int, seed: int) -> Tuple[torch.Tensor, Dict[int, np.ndarray]]:
    rng = np.random.RandomState(int(seed))
    volume = rng.randn(*vol_shape).astype(np.float32) * 0.1

    lesion_masks: Dict[int, np.ndarray] = {}
    for i in range(int(n_lesions)):
        center = [rng.randint(max(1, s // 4), max(2, 3 * s // 4)) for s in vol_shape]
        radius = int(rng.randint(3, 8))
        mask = np.zeros(vol_shape, dtype=bool)
        zz, yy, xx = np.ogrid[:vol_shape[0], :vol_shape[1], :vol_shape[2]]
        dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
        mask[dist <= radius] = True
        lesion_masks[int(i)] = mask
        volume[mask] += 0.5
    return torch.from_numpy(volume), lesion_masks


class SyntheticDataset(Dataset):
    def __init__(
        self,
        *,
        num_samples: int = 100,
        vol_shape: Tuple[int, int, int] = (64, 64, 64),
        n_lesions: int = 3,
        seed: int = 0,
        split: str = "train",
    ):
        self.num_samples = int(num_samples)
        self.vol_shape = tuple(int(x) for x in vol_shape)
        self.n_lesions = int(n_lesions)
        self.seed = int(seed)
        self.split = str(split)
        self.extractor = FrameExtractor()

    def __len__(self) -> int:
        return int(self.num_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seed = int(self.seed) + 10_000 * (0 if self.split == "train" else 1) + int(idx)
        vol, lesion_masks = _synthetic_volume_and_masks(vol_shape=self.vol_shape, n_lesions=self.n_lesions, seed=seed)
        # Stable, parseable reference text.
        report_text = (
            "There is a small left pleural effusion. "
            "A 5mm nodule is seen in the right upper lobe. "
            "No pneumothorax."
        )
        frames = self.extractor.extract_frames(report_text)
        return {
            "sample_id": f"synthetic_{self.split}_{idx}",
            "volume": vol,
            "frames": frames,
            "lesion_masks": lesion_masks,
            "report_text": report_text,
        }


class ManifestDataset(Dataset):
    def __init__(
        self,
        *,
        manifest_path: str,
        split: str = "train",
        max_samples: int = 0,
        resize_shape: Tuple[int, int, int] = (64, 64, 64),
        seed: int = 0,
    ):
        self.manifest_path = str(manifest_path)
        self.split = str(split)
        self.max_samples = int(max_samples)
        self.resize_shape = tuple(int(x) for x in resize_shape)
        self.seed = int(seed)
        self.extractor = FrameExtractor()

        recs = [r for r in load_manifest(self.manifest_path) if str(r.split) == self.split]
        recs = sorted(recs, key=lambda r: r.scan_hash)
        if self.max_samples > 0:
            recs = recs[: self.max_samples]
        self.records: List[ManifestRecord] = recs

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[int(idx)]
        vol, affine_zyx = load_volume_and_affine(r.volume_path, seed=self.seed + int(idx))
        orig_shape = tuple(int(x) for x in vol.shape)
        vol = _resize_volume(vol, resize_shape=self.resize_shape)

        # If we resized the tensor, adjust the affine approximately so token mm
        # geometry remains consistent with the returned volume.
        if affine_zyx is not None and tuple(int(x) for x in vol.shape) != orig_shape:
            dz0, dy0, dx0 = (float(orig_shape[0]), float(orig_shape[1]), float(orig_shape[2]))
            dz1, dy1, dx1 = (float(vol.shape[0]), float(vol.shape[1]), float(vol.shape[2]))
            if dz1 > 0 and dy1 > 0 and dx1 > 0:
                S = np.array(
                    [
                        [dz0 / dz1, 0.0, 0.0, 0.0],
                        [0.0, dy0 / dy1, 0.0, 0.0],
                        [0.0, 0.0, dx0 / dx1, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                affine_zyx = np.asarray(affine_zyx, dtype=np.float64) @ S

        report_text = str(r.report_text or "")
        frames = self.extractor.extract_frames(report_text) if report_text else []

        lesion_masks: Dict[int, Any] = {}
        mp = get_record_mask_path(r)
        if mp:
            try:
                m = load_mask(str(mp))
                if isinstance(m, np.ndarray) and m.ndim == 4:
                    m = _resize_mask(m, resize_shape=self.resize_shape)
                    lesion_masks = {int(j): m[j].astype(bool) for j in range(m.shape[0])}
                elif isinstance(m, np.ndarray) and m.ndim == 3:
                    m = _resize_mask(m, resize_shape=self.resize_shape)
                    lesion_masks = {0: m.astype(bool)}
            except Exception:
                lesion_masks = {}

        return {
            "sample_id": str(r.scan_hash),
            "volume": vol,
            "frames": frames,
            "lesion_masks": lesion_masks,
            "report_text": report_text,
            "record": r,
            "affine_zyx": affine_zyx,
        }


class CTRateDataset(Dataset):
    """Compatibility dataset name used throughout the repo.

    In open-source scaffolds we do not ship CT-RATE, so this class gracefully
    falls back to deterministic synthetic data when the root is missing.
    """

    def __init__(
        self,
        *,
        data_root: str,
        split: str = "train",
        max_samples: int = 100,
        seed: int = 0,
        vol_shape: Tuple[int, int, int] = (64, 64, 64),
    ):
        self.data_root = str(data_root)
        self.split = str(split)
        self.max_samples = int(max_samples)
        self.seed = int(seed)
        self.vol_shape = tuple(int(x) for x in vol_shape)
        self.extractor = FrameExtractor()

        self._use_synth = not Path(self.data_root).exists()
        self._paths: List[Path] = []
        if not self._use_synth:
            root = Path(self.data_root)
            # Expect files like volumes/<split>/*.npy; keep it permissive.
            self._paths = sorted(root.glob("**/*.npy"))
            if self.max_samples > 0:
                self._paths = self._paths[: self.max_samples]

    def __len__(self) -> int:
        return int(self.max_samples if self._use_synth else len(self._paths))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._use_synth:
            ds = SyntheticDataset(num_samples=self.max_samples, vol_shape=self.vol_shape, n_lesions=3, seed=self.seed, split=self.split)
            return ds[int(idx)]

        p = self._paths[int(idx)]
        vol = load_volume(str(p), seed=self.seed + int(idx))
        report_text = ""
        frames: List[Frame] = []
        return {
            "sample_id": p.stem,
            "volume": vol,
            "frames": frames,
            "lesion_masks": {},
            "report_text": report_text,
        }


def _collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {"volume": torch.empty(0), "frames": [], "sample_id": []}

    vols = [s["volume"] for s in samples]
    vols_t = torch.stack([v if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v)) for v in vols], dim=0)
    frames = [s.get("frames", []) for s in samples]
    sample_id = [s.get("sample_id", "") for s in samples]
    lesion_masks = [s.get("lesion_masks", {}) for s in samples]
    report_text = [s.get("report_text", "") for s in samples]
    affines = [s.get("affine_zyx", None) for s in samples]
    out: Dict[str, Any] = {
        "volume": vols_t,
        "frames": frames,
        "sample_id": sample_id,
        "lesion_masks": lesion_masks,
        "report_text": report_text,
    }
    # Keep raw records when present (manifest datasets).
    if any(a is not None for a in affines):
        out["affine_zyx"] = affines
    if any("record" in s for s in samples):
        out["record"] = [s.get("record") for s in samples]
    return out


def make_dataloader(cfg: Dict[str, Any], *, split: str = "train") -> DataLoader:
    dataset_type = str(cfg.get("dataset_type", "synthetic"))
    batch_size = int(cfg.get("batch_size", 1))
    num_workers = int(cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 0))

    if dataset_type == "manifest":
        ds = ManifestDataset(
            manifest_path=str(cfg.get("manifest_path", "")),
            split=str(split),
            max_samples=int(cfg.get("max_samples", 0)),
            resize_shape=tuple(int(x) for x in cfg.get("resize_shape", (64, 64, 64))),
            seed=seed,
        )
        shuffle = bool(str(split) == "train")
    elif dataset_type == "ct_rate":
        ds = CTRateDataset(
            data_root=str(cfg.get("data_root", "")),
            split=str(split),
            max_samples=int(cfg.get("max_samples", cfg.get("num_samples", 100))),
            seed=seed,
            vol_shape=tuple(int(x) for x in cfg.get("vol_shape", (64, 64, 64))),
        )
        shuffle = bool(str(split) == "train")
    else:
        ds = SyntheticDataset(
            num_samples=int(cfg.get("num_samples", 100)),
            vol_shape=tuple(int(x) for x in cfg.get("vol_shape", (64, 64, 64))),
            n_lesions=int(cfg.get("n_lesions", cfg.get("n_lesions_per_sample", 3))),
            seed=seed,
            split=str(split),
        )
        shuffle = bool(str(split) == "train")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
    )
