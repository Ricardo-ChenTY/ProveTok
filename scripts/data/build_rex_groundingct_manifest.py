from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure repo root is on sys.path when running as `python scripts/data/*.py ...`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.data.manifest_schema import (  # noqa: E402
    ManifestRecord,
    compute_manifest_revision,
    find_exact_duplicate_reports,
    save_manifest_jsonl,
)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict):
                yield rec


def _resolve_path(value: str, *, data_root: Path) -> str:
    p = Path(str(value))
    if not str(value):
        return ""
    if p.is_absolute():
        return str(p)
    if str(data_root):
        return str((data_root / p).resolve())
    return str(p.resolve())


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception:  # noqa: BLE001
        np = None  # type: ignore
    return np


def _try_import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except Exception:  # noqa: BLE001
        nib = None  # type: ignore
    return nib


def _convert_mask_to_npy(mask_path: Path, *, out_dir: Path, overwrite: bool) -> Path:
    np = _try_import_numpy()
    nib = _try_import_nibabel()
    if np is None or nib is None:
        missing = []
        if np is None:
            missing.append("numpy")
        if nib is None:
            missing.append("nibabel")
        raise RuntimeError(
            "Mask conversion requires optional deps: " + ", ".join(missing) + ". "
            "Install via `pip install -r requirements.txt`."
        )

    img = nib.load(str(mask_path))
    arr = np.asarray(img.get_fdata(), dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape={arr.shape} for {mask_path}")
    # Provider convention is commonly (X,Y,Z) or (H,W,D); normalize to (D,H,W).
    mask_zyx = np.transpose(arr, (2, 1, 0)) > 0.5

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (mask_path.stem.replace(".nii", "") + ".npy")
    if overwrite or (not out_path.exists()):
        np.save(out_path, mask_zyx.astype("uint8"))
    return out_path


def _assign_split(
    key: str,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> str:
    # Deterministic split by hashing `key`.
    import hashlib

    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    s = sum(ratios)
    if s <= 0:
        return "test"
    tr, vr, _ = (ratios[0] / s, ratios[1] / s, ratios[2] / s)
    h = hashlib.sha1(str(key).encode("utf-8")).digest()
    u = int.from_bytes(h[:4], "little", signed=False) / 2**32
    if u < tr:
        return "train"
    if u < tr + vr:
        return "val"
    return "test"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Normalize a RadGenome/ReXGroundingCT-style index JSONL into a ProveTok manifest JSONL.\n\n"
            "This is intentionally conservative: it does not download data.\n"
            "It can optionally rewrite relative paths under --data-root and convert NIfTI masks to .npy.\n"
        )
    )
    ap.add_argument("--in-index", type=str, required=True, help="Input index JSONL (ManifestRecord schema)")
    ap.add_argument("--out-manifest", type=str, required=True, help="Output manifest JSONL")
    ap.add_argument("--data-root", type=str, default="", help="Prefix for relative volume/mask paths")
    ap.add_argument("--dataset-name", type=str, default="", help="Override dataset field")
    ap.add_argument("--skip-split", action="store_true", help="Keep input split labels instead of re-splitting")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--max-records", type=int, default=0, help="Optional cap (0 means keep all)")
    ap.add_argument("--drop-missing", action="store_true", help="Drop records whose volume/mask path is missing")
    ap.add_argument("--convert-masks-to-npy", action="store_true", help="Convert .nii/.nii.gz masks to .npy (requires numpy+nibabel)")
    ap.add_argument("--out-mask-dir", type=str, default="", help="Where to write converted mask .npy files (required when --convert-masks-to-npy)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite converted masks when present")
    args = ap.parse_args()

    in_index = Path(args.in_index).resolve()
    out_manifest = Path(args.out_manifest).resolve()
    data_root = Path(args.data_root).resolve() if str(args.data_root) else Path()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.convert_masks_to_npy and not str(args.out_mask_dir):
        raise SystemExit("--out-mask-dir is required when --convert-masks-to-npy is set.")
    out_mask_dir = Path(args.out_mask_dir).resolve() if str(args.out_mask_dir) else Path()

    out_records: List[ManifestRecord] = []
    dropped = 0
    for i, d in enumerate(_iter_jsonl(in_index)):
        if args.max_records and int(args.max_records) > 0 and len(out_records) >= int(args.max_records):
            break

        # Normalize known path fields (also accept aliases).
        volume_path = str(d.get("volume_path") or d.get("volume") or "")
        mask_path = str(d.get("mask_path") or d.get("lesion_mask_path") or "")

        volume_path = _resolve_path(volume_path, data_root=data_root) if volume_path else ""
        mask_path = _resolve_path(mask_path, data_root=data_root) if mask_path else ""

        if mask_path and args.convert_masks_to_npy:
            mp = Path(mask_path)
            if mp.suffixes[-2:] in [[".nii", ".gz"]] or mp.suffix == ".nii":
                mask_path = str(_convert_mask_to_npy(mp, out_dir=out_mask_dir, overwrite=bool(args.overwrite)))

        d2 = dict(d)
        if args.dataset_name:
            d2["dataset"] = str(args.dataset_name)
        if volume_path:
            d2["volume_path"] = volume_path
        if mask_path:
            d2["mask_path"] = mask_path

        rec = ManifestRecord.from_dict(d2, default_dataset=str(args.dataset_name or "unknown"))
        if not args.skip_split:
            key = rec.patient_id or rec.series_uid or rec.scan_hash
            rec = ManifestRecord.from_dict(
                {
                    **rec.to_dict(),
                    "split": _assign_split(
                        key,
                        train_ratio=float(args.train_ratio),
                        val_ratio=float(args.val_ratio),
                        test_ratio=float(args.test_ratio),
                    ),
                }
            )

        # Missing-path handling.
        if args.drop_missing:
            ok = True
            if rec.volume_path and not Path(rec.volume_path).exists():
                ok = False
            mp = rec.extra.get("mask_path")
            if isinstance(mp, str) and mp and not Path(mp).exists():
                ok = False
            if not ok:
                dropped += 1
                continue

        out_records.append(rec)

        if i == 0 or ((i + 1) % 5000) == 0:
            print(json.dumps({"progress": f"{i+1}", "kept": len(out_records), "dropped": int(dropped)}), flush=True)

    if not out_records:
        raise SystemExit("No records written (empty input or all dropped).")

    save_manifest_jsonl(out_records, str(out_manifest))
    revision = compute_manifest_revision(out_records)
    dupes = find_exact_duplicate_reports(out_records)
    split_manifest = {
        "train": sorted([r.scan_hash for r in out_records if r.split == "train"]),
        "val": sorted([r.scan_hash for r in out_records if r.split == "val"]),
        "test": sorted([r.scan_hash for r in out_records if r.split == "test"]),
    }

    meta: Dict[str, Any] = {
        "in_index": str(in_index),
        "out_manifest": str(out_manifest),
        "data_root": str(data_root) if str(args.data_root) else "",
        "dataset_name": str(args.dataset_name),
        "skip_split": bool(args.skip_split),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "max_records": int(args.max_records),
        "drop_missing": bool(args.drop_missing),
        "convert_masks_to_npy": bool(args.convert_masks_to_npy),
        "out_mask_dir": str(out_mask_dir) if str(args.out_mask_dir) else "",
        "overwrite": bool(args.overwrite),
        "num_records": len(out_records),
        "num_dropped": int(dropped),
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
                "num_records": len(out_records),
                "num_dropped": int(dropped),
                "revision": revision,
                "meta": str(out_meta),
                "splits": str(out_splits),
                "dupes": str(out_dupes),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
