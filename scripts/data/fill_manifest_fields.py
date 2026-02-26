from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repo root is on sys.path when running as `python scripts/data/*.py ...`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.data.manifest_schema import (  # noqa: E402
    ManifestRecord,
    compute_manifest_revision,
    compute_scan_hash,
    load_manifest,
    save_manifest_jsonl,
)


def _strip_known_suffixes(name: str) -> str:
    s = str(name or "").strip()
    for suf in (".nii.gz", ".nii", ".npz", ".npy"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s


def _infer_ids_from_path(volume_path: str) -> Tuple[str, str]:
    p = Path(str(volume_path or ""))
    stem = _strip_known_suffixes(p.name)
    parts = stem.split("_")
    if len(parts) >= 3:
        patient_id = "_".join(parts[:2])
        series_uid = "_".join(parts[:3])
        return patient_id, series_uid
    if len(parts) == 2:
        return stem, stem
    return stem, stem


def _normalize_split(split: str, default_split: str) -> str:
    s = str(split or "").strip().lower()
    mp = {
        "train": "train",
        "tr": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "test": "test",
        "te": "test",
    }
    return mp.get(s, str(default_split))


def _pick_report_text(rec: ManifestRecord) -> str:
    txt = str(getattr(rec, "report_text", "") or "").strip()
    if txt:
        return txt
    extra = rec.extra or {}
    for k in ("report", "findings", "Findings", "text", "impression"):
        v = extra.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _build_split_lock(records: List[ManifestRecord]) -> Dict[str, List[str]]:
    out = {"train": [], "val": [], "test": []}
    for r in records:
        sp = str(getattr(r, "split", "") or "").strip()
        if sp not in out:
            continue
        out[sp].append(str(getattr(r, "scan_hash", "") or ""))
    for k in out.keys():
        out[k] = sorted([x for x in out[k] if x])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill/normalize manifest fields and freeze splits.")
    ap.add_argument("--in-manifest", type=str, required=True, help="Input manifest (.jsonl/.csv)")
    ap.add_argument("--out-manifest", type=str, required=True, help="Output normalized manifest (.jsonl)")
    ap.add_argument("--default-dataset", type=str, default="unknown")
    ap.add_argument("--default-split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--set-split", type=str, default="", choices=["", "train", "val", "test"], help="Force all rows to one split.")
    ap.add_argument("--strict-scan-hash", action="store_true", help="If IDs are present, force SHA256(patient_id||study_date||series_uid).")
    args = ap.parse_args()

    records = load_manifest(str(args.in_manifest))
    out: List[ManifestRecord] = []
    stats: Dict[str, int] = {
        "n_records": 0,
        "filled_patient_id": 0,
        "filled_series_uid": 0,
        "filled_scan_hash": 0,
        "normalized_split": 0,
        "filled_report_text": 0,
        "forced_scan_hash": 0,
    }

    for rec in records:
        stats["n_records"] += 1
        d = rec.to_dict()

        dataset = str(d.get("dataset") or args.default_dataset)

        split_old = str(d.get("split") or "")
        split_new = str(args.set_split) if str(args.set_split).strip() else _normalize_split(split_old, str(args.default_split))
        if split_new != split_old:
            stats["normalized_split"] += 1

        volume_path = str(d.get("volume_path", d.get("volume", "")) or "")
        patient_id = str(d.get("patient_id") or "").strip()
        series_uid = str(d.get("series_uid") or "").strip()
        study_date = str(d.get("study_date") or "").strip()
        scan_hash = str(d.get("scan_hash") or "").strip()
        report_text = str(d.get("report_text") or "").strip()

        if not patient_id or not series_uid:
            p_infer, s_infer = _infer_ids_from_path(volume_path)
            if not patient_id and p_infer:
                patient_id = p_infer
                stats["filled_patient_id"] += 1
            if not series_uid and s_infer:
                series_uid = s_infer
                stats["filled_series_uid"] += 1

        if not report_text:
            rt = _pick_report_text(rec)
            if rt:
                report_text = rt
                stats["filled_report_text"] += 1

        if bool(args.strict_scan_hash) and patient_id and series_uid:
            forced = compute_scan_hash(patient_id, study_date, series_uid)
            if scan_hash != forced:
                scan_hash = forced
                stats["forced_scan_hash"] += 1

        if not scan_hash:
            if patient_id and series_uid:
                scan_hash = compute_scan_hash(patient_id, study_date, series_uid)
            elif series_uid:
                scan_hash = series_uid
            elif volume_path:
                scan_hash = _strip_known_suffixes(Path(volume_path).name)
            else:
                scan_hash = f"scan_{stats['n_records']:08d}"
            stats["filled_scan_hash"] += 1

        out_rec = ManifestRecord.from_dict(
            {
                **d,
                "dataset": dataset,
                "split": split_new,
                "patient_id": patient_id,
                "study_date": study_date,
                "series_uid": series_uid,
                "scan_hash": scan_hash,
                "report_text": report_text,
                "volume_path": volume_path,
            }
        )
        out.append(out_rec)

    out_manifest = Path(str(args.out_manifest)).resolve()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    save_manifest_jsonl(out, str(out_manifest))

    split_lock = _build_split_lock(out)
    revision = compute_manifest_revision(out)

    lock_path = out_manifest.with_suffix(out_manifest.suffix + ".splits.json")
    meta_path = out_manifest.with_suffix(out_manifest.suffix + ".meta.json")
    lock_path.write_text(json.dumps(split_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "input_manifest": str(Path(str(args.in_manifest)).resolve()),
                "output_manifest": str(out_manifest),
                "stats": stats,
                "revision": revision,
                "split_counts": {k: len(v) for k, v in split_lock.items()},
                "split_lock_path": str(lock_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_manifest": str(out_manifest),
                "revision": revision,
                "stats": stats,
                "split_counts": {k: len(v) for k, v in split_lock.items()},
                "meta": str(meta_path),
                "splits": str(lock_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

