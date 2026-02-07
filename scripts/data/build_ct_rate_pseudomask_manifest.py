from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path when running as `python scripts/data/*.py ...`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.data.io import load_volume
from provetok.data.manifest_schema import (
    ManifestRecord,
    compute_manifest_revision,
    find_exact_duplicate_reports,
    load_manifest,
    save_manifest_jsonl,
)
from provetok.models.saliency_cnn3d import load_saliency_cnn3d


def _preprocess_volume(vol: torch.Tensor) -> torch.Tensor:
    # Keep consistent with train/usage paths in experiments.
    v = vol.float().clamp(min=-1000.0, max=1000.0)
    v = v / 1000.0
    return v


def _resize_volume(vol: torch.Tensor, resize_shape: Sequence[int]) -> torch.Tensor:
    if len(resize_shape) != 3:
        raise ValueError(f"resize_shape must have 3 dims, got {resize_shape!r}")
    target = tuple(int(x) for x in resize_shape)
    if tuple(int(x) for x in vol.shape) == target:
        return vol
    x = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    y = F.interpolate(x, size=target, mode="trilinear", align_corners=False)
    return y[0, 0]


def _binarize_prob(
    prob: np.ndarray,
    *,
    threshold: float,
    min_voxels: int,
    target_mask_ratio: float,
    max_mask_ratio: float,
) -> np.ndarray:
    if prob.ndim != 3:
        raise ValueError(f"Expected 3D prob map, got shape={prob.shape}")

    p = np.asarray(prob, dtype=np.float32)
    mask = p >= float(threshold)

    n_vox = int(mask.size)
    min_keep = max(1, int(min_voxels))
    ratio = float(mask.mean())

    # Domain shift can make fixed thresholds degenerate (all-empty/all-full).
    # Fall back to a fixed top-k ratio to keep weak labels usable but bounded.
    if int(mask.sum()) < min_keep or ratio > float(max_mask_ratio):
        k = int(round(float(target_mask_ratio) * float(n_vox)))
        k = max(min_keep, min(k, n_vox))
        flat = p.reshape(-1)
        top_idx = np.argpartition(flat, -k)[-k:]
        out = np.zeros(n_vox, dtype=bool)
        out[top_idx] = True
        mask = out.reshape(p.shape)

    return mask.astype(bool)


def _select_records(
    records: List[ManifestRecord],
    *,
    splits: Sequence[str],
    max_samples_per_split: int,
    seed: int,
) -> List[ManifestRecord]:
    rng = np.random.RandomState(int(seed))
    selected: List[ManifestRecord] = []
    split_set = {str(s) for s in splits}

    for split in ("train", "val", "test"):
        if split not in split_set:
            continue
        rows = [r for r in records if r.split == split]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r.scan_hash)
        if max_samples_per_split > 0 and len(rows) > int(max_samples_per_split):
            idx = rng.choice(len(rows), size=int(max_samples_per_split), replace=False)
            rows = [rows[int(i)] for i in sorted(idx.tolist())]
        selected.extend(rows)
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a weak-label pseudo-mask manifest for CT-RATE using a trained SaliencyCNN3D.\n\n"
            "This creates a new manifest with `mask_path` filled for selected records so that\n"
            "grounding experiments can run on CT-RATE as a weak-label (not gold-mask) proxy."
        )
    )
    ap.add_argument("--in-manifest", type=str, required=True, help="Input CT-RATE manifest.jsonl")
    ap.add_argument("--out-manifest", type=str, required=True, help="Output pseudo-mask manifest.jsonl")
    ap.add_argument("--pseudo-mask-dir", type=str, required=True, help="Directory to save generated mask .npy files")
    ap.add_argument("--saliency-weights", type=str, required=True, help="Path to saliency_cnn3d.pt")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--splits", type=str, nargs="+", default=["test"], choices=["train", "val", "test"])
    ap.add_argument("--max-samples-per-split", type=int, default=50, help="0 means use all records in selected splits.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64], help="Pseudo-mask prediction shape (D,H,W).")
    ap.add_argument("--threshold", type=float, default=0.5, help="Primary probability threshold for binarization.")
    ap.add_argument("--min-voxels", type=int, default=64, help="Fallback minimum positive voxels per mask.")
    ap.add_argument("--target-mask-ratio", type=float, default=0.005, help="Fallback top-k ratio when threshold is degenerate.")
    ap.add_argument("--max-mask-ratio", type=float, default=0.2, help="If thresholded mask is larger than this ratio, fallback to top-k.")
    ap.add_argument("--dataset-name", type=str, default="ct_rate_100g_pseudomask")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy pseudo masks.")
    args = ap.parse_args()

    in_manifest = Path(args.in_manifest).resolve()
    out_manifest = Path(args.out_manifest).resolve()
    pseudo_mask_dir = Path(args.pseudo_mask_dir).resolve()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pseudo_mask_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(str(in_manifest))
    selected = _select_records(
        records,
        splits=[str(s) for s in args.splits],
        max_samples_per_split=int(args.max_samples_per_split),
        seed=int(args.seed),
    )
    if not selected:
        raise SystemExit("No records selected. Check --splits and --max-samples-per-split.")

    device = torch.device(str(args.device))
    model = load_saliency_cnn3d(str(args.saliency_weights), map_location="cpu")
    model = model.to(device=device).eval()

    resize_shape = tuple(int(x) for x in args.resize_shape)
    out_records: List[ManifestRecord] = []
    ratio_by_split: Dict[str, List[float]] = {"train": [], "val": [], "test": []}

    for i, r in enumerate(selected):
        vol = load_volume(r.volume_path)
        vol = _resize_volume(vol, resize_shape=resize_shape)
        x = _preprocess_volume(vol).unsqueeze(0).unsqueeze(0).to(device=device)
        with torch.no_grad():
            prob = model.predict_proba(x).detach().cpu()[0, 0].numpy()
        mask = _binarize_prob(
            prob,
            threshold=float(args.threshold),
            min_voxels=int(args.min_voxels),
            target_mask_ratio=float(args.target_mask_ratio),
            max_mask_ratio=float(args.max_mask_ratio),
        )

        out_mask = pseudo_mask_dir / str(r.split) / f"{r.scan_hash}.npy"
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or (not out_mask.exists()):
            np.save(out_mask, mask.astype(np.uint8))

        d = r.to_dict()
        d["dataset"] = str(args.dataset_name or r.dataset)
        d["mask_path"] = str(out_mask)
        d["pseudo_mask_source"] = "saliency_cnn3d"
        d["pseudo_mask_is_weak_label"] = True
        d["pseudo_mask_threshold"] = float(args.threshold)
        d["pseudo_mask_resize_shape"] = [int(x) for x in resize_shape]
        d["pseudo_mask_ratio"] = float(mask.mean())
        d["pseudo_mask_saliency_weights"] = str(Path(args.saliency_weights).resolve())
        out_records.append(ManifestRecord.from_dict(d))
        ratio_by_split.setdefault(str(r.split), []).append(float(mask.mean()))

        if i == 0 or ((i + 1) % 10) == 0:
            print(
                json.dumps(
                    {
                        "progress": f"{i+1}/{len(selected)}",
                        "split": r.split,
                        "scan_hash": r.scan_hash,
                        "mask_ratio": float(mask.mean()),
                    }
                ),
                flush=True,
            )

    save_manifest_jsonl(out_records, str(out_manifest))
    revision = compute_manifest_revision(out_records)
    dupes = find_exact_duplicate_reports(out_records)
    split_manifest = {
        "train": sorted([r.scan_hash for r in out_records if r.split == "train"]),
        "val": sorted([r.scan_hash for r in out_records if r.split == "val"]),
        "test": sorted([r.scan_hash for r in out_records if r.split == "test"]),
    }
    mean_ratio = {
        k: (float(np.mean(v)) if v else 0.0)
        for k, v in ratio_by_split.items()
    }
    meta: Dict[str, Any] = {
        "in_manifest": str(in_manifest),
        "out_manifest": str(out_manifest),
        "pseudo_mask_dir": str(pseudo_mask_dir),
        "saliency_weights": str(Path(args.saliency_weights).resolve()),
        "device": str(args.device),
        "splits": [str(s) for s in args.splits],
        "max_samples_per_split": int(args.max_samples_per_split),
        "resize_shape": [int(x) for x in resize_shape],
        "threshold": float(args.threshold),
        "min_voxels": int(args.min_voxels),
        "target_mask_ratio": float(args.target_mask_ratio),
        "max_mask_ratio": float(args.max_mask_ratio),
        "num_records": len(out_records),
        "mean_mask_ratio_by_split": mean_ratio,
        "revision": revision,
        "num_duplicate_report_groups": len(dupes),
        "duplicate_report_groups_preview": dupes[:3],
    }

    out_meta = out_manifest.with_suffix(out_manifest.suffix + ".meta.json")
    out_splits = out_manifest.with_suffix(out_manifest.suffix + ".splits.json")
    out_dupes = out_manifest.with_suffix(out_manifest.suffix + ".dupes.json")
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_splits.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_dupes.write_text(json.dumps({"exact_duplicate_report_groups": dupes}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "out_manifest": str(out_manifest),
                "revision": revision,
                "num_records": len(out_records),
                "mean_mask_ratio_by_split": mean_ratio,
                "meta": str(out_meta),
                "splits": str(out_splits),
                "dupes": str(out_dupes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
