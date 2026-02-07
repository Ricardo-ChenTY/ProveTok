from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import nibabel as nib
import numpy as np

# Ensure repo root is on sys.path when running as `python scripts/data/*.py ...`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.data.manifest_schema import (
    ManifestRecord,
    compute_manifest_revision,
    find_exact_duplicate_reports,
    load_manifest,
    save_manifest_jsonl,
)


def _select_records(
    records: List[ManifestRecord],
    *,
    splits: Sequence[str],
    max_samples: int,
    seed: int,
) -> List[ManifestRecord]:
    split_set = {str(s) for s in splits}
    rows = [r for r in records if r.split in split_set]
    rows = sorted(rows, key=lambda r: r.scan_hash)
    if max_samples > 0 and len(rows) > int(max_samples):
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(len(rows), size=int(max_samples), replace=False)
        rows = [rows[int(i)] for i in sorted(idx.tolist())]
    return rows


def _case_id(record: ManifestRecord) -> str:
    volume_name = str(record.extra.get("volume_name") or record.series_uid or "")
    if volume_name.endswith(".nii.gz"):
        return volume_name[: -len(".nii.gz")]
    if volume_name.endswith(".nii"):
        return volume_name[: -len(".nii")]
    return volume_name


def _load_union_mask_xyz(pleural_path: Path, pericardial_path: Path) -> np.ndarray:
    pleural = np.asarray(nib.load(str(pleural_path)).get_fdata())
    pericardial = np.asarray(nib.load(str(pericardial_path)).get_fdata())
    if pleural.ndim != 3 or pericardial.ndim != 3:
        raise ValueError(
            f"Expected 3D masks, got pleural={pleural.shape} pericardial={pericardial.shape}"
        )
    if pleural.shape != pericardial.shape:
        raise ValueError(
            f"Mask shape mismatch: pleural={pleural.shape} pericardial={pericardial.shape}"
        )
    return np.logical_or(pleural > 0, pericardial > 0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build CT-RATE eval-only manifest with mask_path from TotalSegmentator "
            "pleural/pericardial effusion masks."
        )
    )
    ap.add_argument("--in-manifest", type=str, required=True, help="Input manifest.jsonl")
    ap.add_argument("--out-manifest", type=str, required=True, help="Output manifest.jsonl")
    ap.add_argument("--mask-root", type=str, required=True, help="Root dir with <case_id>/pleural_effusion.nii.gz and pericardial_effusion.nii.gz")
    ap.add_argument("--out-mask-dir", type=str, required=True, help="Directory to save union masks (.npy)")
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--set-split", type=str, default="test", choices=["train", "val", "test"], help="Overwrite split for output records (eval-only usually uses test).")
    ap.add_argument("--max-samples", type=int, default=0, help="0 means keep all selected records.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-name", type=str, default="ct_rate_tsseg_effusion_eval")
    ap.add_argument("--drop-missing", action="store_true", help="Drop records when required mask files are missing.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing union masks.")
    args = ap.parse_args()

    in_manifest = Path(args.in_manifest).resolve()
    out_manifest = Path(args.out_manifest).resolve()
    mask_root = Path(args.mask_root).resolve()
    out_mask_dir = Path(args.out_mask_dir).resolve()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(str(in_manifest))
    selected = _select_records(
        records,
        splits=[str(s) for s in args.splits],
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    if not selected:
        raise SystemExit("No records selected. Check --splits and --max-samples.")

    out_records: List[ManifestRecord] = []
    missing_cases: List[str] = []
    empty_masks = 0

    for i, r in enumerate(selected):
        cid = _case_id(r)
        pleural_path = mask_root / cid / "pleural_effusion.nii.gz"
        pericardial_path = mask_root / cid / "pericardial_effusion.nii.gz"
        if not (pleural_path.exists() and pericardial_path.exists()):
            if args.drop_missing:
                missing_cases.append(cid)
                continue
            raise FileNotFoundError(f"Missing masks for case={cid}: {pleural_path} | {pericardial_path}")

        mask_xyz = _load_union_mask_xyz(pleural_path=pleural_path, pericardial_path=pericardial_path)
        # Convert from provider axis order (X,Y,Z) to loader order (D,H,W) == (Z,Y,X).
        mask_zyx = np.transpose(mask_xyz.astype(np.uint8), (2, 1, 0))

        if int(mask_zyx.sum()) == 0:
            empty_masks += 1

        out_mask = out_mask_dir / str(args.set_split) / f"{r.scan_hash}.npy"
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or (not out_mask.exists()):
            np.save(out_mask, mask_zyx)

        d = r.to_dict()
        d["dataset"] = str(args.dataset_name or r.dataset)
        d["split"] = str(args.set_split)
        d["mask_path"] = str(out_mask)
        d["mask_quality"] = "silver_auto_unverified"
        d["tsseg_mask_source"] = "totalsegmentator_pleural_pericard_effusion"
        d["tsseg_mask_is_unverified_auto_label"] = True
        d["tsseg_mask_components"] = ["pleural_effusion", "pericardial_effusion"]
        d["tsseg_mask_case_id"] = cid
        out_records.append(ManifestRecord.from_dict(d))

        if i == 0 or ((i + 1) % 10) == 0:
            print(
                json.dumps(
                    {
                        "progress": f"{i+1}/{len(selected)}",
                        "case_id": cid,
                        "scan_hash": r.scan_hash,
                        "mask_ratio": float(mask_zyx.mean()),
                    }
                ),
                flush=True,
            )

    if not out_records:
        raise SystemExit("All selected records were dropped due to missing masks.")

    save_manifest_jsonl(out_records, str(out_manifest))
    revision = compute_manifest_revision(out_records)
    dupes = find_exact_duplicate_reports(out_records)
    split_manifest = {
        "train": sorted([r.scan_hash for r in out_records if r.split == "train"]),
        "val": sorted([r.scan_hash for r in out_records if r.split == "val"]),
        "test": sorted([r.scan_hash for r in out_records if r.split == "test"]),
    }
    ratios = [float(np.load(Path(r.extra["mask_path"])).mean()) for r in out_records]
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0

    meta: Dict[str, Any] = {
        "in_manifest": str(in_manifest),
        "out_manifest": str(out_manifest),
        "mask_root": str(mask_root),
        "out_mask_dir": str(out_mask_dir),
        "splits": [str(s) for s in args.splits],
        "set_split": str(args.set_split),
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "dataset_name": str(args.dataset_name),
        "num_records": len(out_records),
        "num_missing_cases_dropped": len(missing_cases),
        "missing_cases_preview": missing_cases[:10],
        "num_empty_union_masks": int(empty_masks),
        "mean_mask_ratio": mean_ratio,
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
                "mean_mask_ratio": mean_ratio,
                "num_missing_cases_dropped": len(missing_cases),
                "meta": str(out_meta),
                "splits": str(out_splits),
                "dupes": str(out_dupes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
