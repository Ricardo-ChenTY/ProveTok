from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# Ensure repo root is on sys.path when running as `python scripts/data/*.py ...`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.data.manifest_schema import ManifestRecord, save_manifest_jsonl


def _human_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.2f} GB"


def _derive_patient_id(volume_name: str) -> str:
    # Example: train_1741_b_2.nii.gz -> train_1741 ; valid_14_a_1.nii.gz -> valid_14
    stem = volume_name
    if stem.endswith(".nii.gz"):
        stem = stem[: -len(".nii.gz")]
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def _anatomy_leaf(anatomy: str) -> str:
    a = str(anatomy or "").strip()
    if not a:
        return ""
    # Provider convention: hierarchical paths separated by "/".
    return a.split("/")[-1].strip()


_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(s: str) -> str:
    # Turn mask name into a filesystem-friendly component.
    t = str(s or "").strip()
    if not t:
        return "empty"
    t = t.replace("/", "_")
    t = re.sub(r"\s+", "_", t)
    t = _SAFE_CHARS_RE.sub("_", t)
    t = t.strip("_")
    return t or "empty"


def _collect_local_ct_names(ct_root: Path) -> Dict[str, str]:
    # Map filename -> absolute path
    out: Dict[str, str] = {}
    for p in ct_root.rglob("*.nii.gz"):
        out[p.name] = str(p.resolve())
    return out


@dataclass(frozen=True)
class _Row:
    row_idx: int
    volume_name: str
    anatomy_path: str
    anatomy_leaf: str
    sentence: str


class _MultiFileReader(io.RawIOBase):
    """Read bytes sequentially from multiple files, exposing a single stream."""

    def __init__(self, paths: List[Path]):
        super().__init__()
        self._paths = list(paths)
        self._files = [p.open("rb") for p in self._paths]
        self._idx = 0

    def readable(self) -> bool:  # pragma: no cover
        return True

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        if self._idx >= len(self._files):
            return b""
        if size == 0:
            return b""
        if size < 0:
            # Avoid unbounded reads on huge streams.
            raise ValueError("MultiFileReader.read(size=-1) is not supported for huge streams.")

        buf = bytearray()
        remaining = int(size)
        while remaining > 0 and self._idx < len(self._files):
            chunk = self._files[self._idx].read(remaining)
            if not chunk:
                self._files[self._idx].close()
                self._idx += 1
                continue
            buf.extend(chunk)
            remaining -= len(chunk)
        return bytes(buf)

    def close(self) -> None:  # pragma: no cover
        try:
            for f in self._files:
                try:
                    f.close()
                except Exception:
                    pass
        finally:
            super().close()


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def _extract_masks_from_sharded_tar_gz(
    *,
    shard_paths: List[Path],
    member_to_dst: Dict[str, Path],
    overwrite: bool,
) -> Tuple[Set[str], int]:
    """Extract a set of members from a sharded `.tar.gz` stream.

    RadGenome provides masks as multiple shard files that are actually one contiguous
    gzip stream. We open them as a single stream and iterate tar members sequentially.

    Returns:
        (missing_members, extracted_count)
    """
    wanted: Set[str] = set(member_to_dst.keys())
    missing: Set[str] = set(wanted)
    extracted = 0

    # Skip members that already exist (unless overwrite).
    if not overwrite:
        for m, dst in list(member_to_dst.items()):
            if dst.exists():
                missing.discard(m)

    if not missing:
        return set(), 0

    reader = _MultiFileReader(shard_paths)
    try:
        # Streaming mode: sequential scan; safe for huge archives.
        tar = tarfile.open(fileobj=reader, mode="r|gz")
        try:
            for ti in tar:
                if not missing:
                    break
                name = ti.name
                if name not in missing:
                    continue
                if not ti.isfile():
                    continue
                dst = member_to_dst.get(name)
                if dst is None:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                src_f = tar.extractfile(ti)
                if src_f is None:
                    continue
                with src_f, dst.open("wb") as out_f:
                    # Avoid shutil to keep dependencies minimal.
                    while True:
                        chunk = src_f.read(1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
                extracted += 1
                missing.discard(name)
        finally:
            try:
                tar.close()
            except Exception:
                pass
    finally:
        try:
            reader.close()
        except Exception:
            pass

    return missing, extracted


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Create a ProveTok input index (manifest schema) for RadGenome-ChestCT.\n\n"
            "This script:\n"
            "1) Scans local RadGenome preprocessed CT volumes (no downloads).\n"
            "2) Samples a small subset of (volume, anatomy, sentence) rows.\n"
            "3) Extracts the corresponding anatomy masks from the sharded tar.gz stream.\n"
            "4) Writes an index JSONL that can be normalized via build_rex_groundingct_manifest.py.\n\n"
            "Recommended next step:\n"
            "  python scripts/data/build_rex_groundingct_manifest.py --in-index <out_index.jsonl> --data-root / --skip-split --out-manifest <manifest.jsonl>\n"
        )
    )
    ap.add_argument("--radgenome-root", type=str, required=True, help="RadGenome-ChestCT root (contains dataset/)")
    ap.add_argument(
        "--split-csv",
        type=str,
        default="train",
        choices=["train"],
        help="Which RadGenome CSV to use. Only train is supported with local masks (train_anatomy_mask_*).",
    )
    ap.add_argument("--out-index", type=str, required=True, help="Output index JSONL (manifest schema)")
    ap.add_argument("--dataset-name", type=str, default="radgenome_chestct_mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-volumes", type=int, default=100, help="Max unique CT volumes to include")
    ap.add_argument("--max-records-per-volume", type=int, default=3, help="Max anatomy sentences per volume")
    ap.add_argument("--out-mask-dir", type=str, required=True, help="Where to write extracted mask files")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split label to assign")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted masks")
    ap.add_argument(
        "--leaf-allowlist",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional allowlist for anatomy leaf names (e.g., lung heart bone aorta). "
            "When provided, only rows whose leaf is in this set are considered."
        ),
    )
    args = ap.parse_args()

    rg_root = Path(args.radgenome_root).resolve()
    dataset_dir = rg_root / "dataset"
    if not dataset_dir.exists():
        raise SystemExit(f"Missing {dataset_dir} (expected RadGenome-ChestCT root)")

    csv_path = dataset_dir / "radgenome_files" / ("train_region_report.csv" if args.split_csv == "train" else "")
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")

    ct_root = dataset_dir / "train_preprocessed"
    if not ct_root.exists():
        raise SystemExit(f"Missing {ct_root} (expected local preprocessed CT volumes)")

    shard_paths = [
        dataset_dir / "train_anatomy_mask_aa",
        dataset_dir / "train_anatomy_mask_ab",
        dataset_dir / "train_anatomy_mask_ac",
        dataset_dir / "train_anatomy_mask_ad",
        dataset_dir / "train_anatomy_mask_ae",
    ]
    for p in shard_paths:
        if not p.exists():
            raise SystemExit(f"Missing mask shard: {p}")

    ct_names = _collect_local_ct_names(ct_root)
    if not ct_names:
        raise SystemExit(f"No CT volumes found under {ct_root}")

    rng = np.random.RandomState(int(args.seed))
    max_volumes = int(args.max_volumes)
    max_per_vol = int(args.max_records_per_volume)
    if max_volumes <= 0:
        raise SystemExit("--max-volumes must be > 0")
    if max_per_vol <= 0:
        raise SystemExit("--max-records-per-volume must be > 0")

    available_vols = sorted([v for v in ct_names.keys() if v.startswith("train_") and v.endswith(".nii.gz")])
    if not available_vols:
        raise SystemExit("No local RadGenome train volumes found (expected files like train_XXXX_a_1.nii.gz)")

    rng.shuffle(available_vols)
    chosen_vols = set(available_vols[:max_volumes])

    allow_leaf: Optional[Set[str]] = None
    if args.leaf_allowlist:
        allow_leaf = set([str(x).strip() for x in args.leaf_allowlist if str(x).strip()])

    # Reservoir-sample rows per volume while streaming the huge CSV.
    per_vol_seen: Dict[str, int] = {}
    per_vol_rows: Dict[str, List[_Row]] = {}

    for row_idx, row in enumerate(_iter_csv_rows(csv_path)):
        vol = str(row.get("Volumename", "") or "").strip()
        if not vol or vol not in chosen_vols:
            continue
        anatomy = str(row.get("Anatomy", "") or "").strip()
        if not anatomy or anatomy.lower() == "nan":
            continue
        sentence = str(row.get("Sentence", "") or "").strip()
        if not sentence:
            continue

        leaf = _anatomy_leaf(anatomy)
        if not leaf:
            continue
        if allow_leaf is not None and leaf not in allow_leaf:
            continue

        seen = per_vol_seen.get(vol, 0) + 1
        per_vol_seen[vol] = seen
        bucket = per_vol_rows.setdefault(vol, [])

        candidate = _Row(
            row_idx=int(row_idx),
            volume_name=vol,
            anatomy_path=anatomy,
            anatomy_leaf=leaf,
            sentence=sentence,
        )

        if len(bucket) < max_per_vol:
            bucket.append(candidate)
            continue

        # Reservoir replacement.
        j = int(rng.randint(0, seen))
        if j < max_per_vol:
            bucket[j] = candidate

    # Flatten selected rows (deterministic order for reproducibility).
    rows: List[_Row] = []
    for vol in sorted(per_vol_rows.keys()):
        # Keep per-volume ordering stable across runs for same seed+data.
        rows.extend(sorted(per_vol_rows[vol], key=lambda r: (r.anatomy_leaf, r.row_idx)))

    if not rows:
        raise SystemExit(
            "No candidate RadGenome rows found. "
            "Try increasing --max-volumes or removing --leaf-allowlist."
        )

    out_index = Path(args.out_index).resolve()
    out_index.parent.mkdir(parents=True, exist_ok=True)
    out_mask_dir = Path(args.out_mask_dir).resolve()
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    member_to_dst: Dict[str, Path] = {}
    row_to_member: Dict[Tuple[str, int, str], str] = {}
    for r in rows:
        volume_stem = r.volume_name
        if volume_stem.endswith(".nii.gz"):
            volume_stem = volume_stem[: -len(".nii.gz")]
        member = f"train_anatomy_mask/seg_{volume_stem}/{r.anatomy_leaf}.nii.gz"
        dst = out_mask_dir / f"{volume_stem}__{_slug(r.anatomy_leaf)}.nii.gz"
        member_to_dst.setdefault(member, dst)
        row_to_member[(r.volume_name, r.row_idx, r.anatomy_leaf)] = member

    missing, extracted = _extract_masks_from_sharded_tar_gz(
        shard_paths=shard_paths,
        member_to_dst=member_to_dst,
        overwrite=bool(args.overwrite),
    )

    # Build manifest index records only for rows whose masks exist.
    kept: List[ManifestRecord] = []
    missing_rows = 0
    unique_volumes: Set[str] = set()
    for r in rows:
        member = row_to_member[(r.volume_name, r.row_idx, r.anatomy_leaf)]
        if member in missing:
            missing_rows += 1
            continue
        dst = member_to_dst[member]
        if not dst.exists():
            missing_rows += 1
            continue

        vol_path = ct_names.get(r.volume_name)
        if not vol_path:
            # Should not happen due to chosen_vols, but keep robust.
            missing_rows += 1
            continue

        patient_id = _derive_patient_id(r.volume_name)
        series_uid = f"{r.volume_name}::rg_{r.row_idx}"
        kept.append(
            ManifestRecord.from_dict(
                {
                    "dataset": args.dataset_name,
                    "split": args.split,
                    "patient_id": patient_id,
                    "study_date": "00000000",
                    "series_uid": series_uid,
                    "volume_path": vol_path,
                    "report_text": r.sentence,
                    "mask_path": str(dst),
                    "radgenome_volume_name": r.volume_name,
                    "radgenome_row_idx": int(r.row_idx),
                    "radgenome_anatomy_path": r.anatomy_path,
                    "radgenome_anatomy_leaf": r.anatomy_leaf,
                    "radgenome_mask_member": member,
                }
            )
        )
        unique_volumes.add(r.volume_name)

    if not kept:
        raise SystemExit(
            "Failed to build any records: all selected (volume, anatomy) masks were missing. "
            "Try increasing --max-volumes or removing --leaf-allowlist."
        )

    # Stats: estimate unique CT size.
    ct_bytes = 0
    for v in sorted(unique_volumes):
        vp = ct_names.get(v)
        if not vp:
            continue
        try:
            ct_bytes += int(Path(vp).stat().st_size)
        except Exception:
            pass

    # Estimate extracted mask bytes.
    mask_bytes = 0
    for r in kept:
        mp = r.extra.get("mask_path")
        if not mp:
            continue
        try:
            mask_bytes += int(Path(str(mp)).stat().st_size)
        except Exception:
            pass

    save_manifest_jsonl(kept, str(out_index))

    print(
        json.dumps(
            {
                "out_index": str(out_index),
                "dataset": args.dataset_name,
                "split": args.split,
                "num_records": len(kept),
                "unique_volumes": len(unique_volumes),
                "ct_total_gb_est": ct_bytes / 1e9,
                "mask_total_gb_est": mask_bytes / 1e9,
                "masks_extracted_new": int(extracted),
                "missing_mask_members": len(missing),
                "missing_rows_dropped": int(missing_rows),
                "radgenome_root": str(rg_root),
                "ct_root": str(ct_root),
                "csv": str(csv_path),
                "mask_shards": [str(p) for p in shard_paths],
                "out_mask_dir": str(out_mask_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

