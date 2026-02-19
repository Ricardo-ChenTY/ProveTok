#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def _split_to_ct2rep_dir(split: str) -> Optional[str]:
    s = str(split).strip().lower()
    if s == "train":
        return "train"
    if s in ("val", "valid", "validation"):
        return "valid"
    if s == "test":
        return "test"
    return None


def _iter_records(
    manifest: str,
    *,
    splits: List[str],
    max_records_per_split: int,
) -> Iterable[dict]:
    splits_norm = {str(s).strip().lower() for s in splits}
    counts: Dict[str, int] = {}
    for r in _read_jsonl(manifest):
        split = str(r.get("split", "")).strip().lower()
        if splits_norm and split not in splits_norm:
            continue
        if max_records_per_split > 0:
            n = counts.get(split, 0)
            if n >= max_records_per_split:
                continue
            counts[split] = n + 1
        yield r


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a CT2Rep-style folder dataset + reports.xlsx from a ProveTok manifest.\n\n"
            "CT2Rep expects:\n"
            "- <split>/patient_id/accession_id/*.npz\n"
            "- an Excel file with columns: AccessionNo, Findings_EN\n\n"
            "We set accession_id = scan_hash, and symlink existing .npz volumes to avoid duplication."
        )
    )
    ap.add_argument("--manifest", type=str, required=True, help="Input manifest jsonl")
    ap.add_argument("--out-root", type=str, required=True, help="Output root (contains train/valid/test)")
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Which manifest splits to include",
    )
    ap.add_argument(
        "--empty-text-splits",
        type=str,
        nargs="*",
        default=["test"],
        help="Splits whose Findings_EN will be written as empty string (avoid token leakage into vocab).",
    )
    ap.add_argument(
        "--xlsx",
        type=str,
        default="",
        help="Output xlsx path (default: <out-root>/reports.xlsx)",
    )
    ap.add_argument(
        "--link-mode",
        type=str,
        choices=["symlink"],
        default="symlink",
        help="How to place volume files (only symlink supported for safety).",
    )
    ap.add_argument(
        "--max-records-per-split",
        type=int,
        default=0,
        help="Optional cap per split (smoke/debug). 0 means no cap.",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    _ensure_dir(out_root)
    for d in ["train", "valid", "test"]:
        _ensure_dir(out_root / d)

    xlsx_path = Path(args.xlsx) if str(args.xlsx) else (out_root / "reports.xlsx")

    empty_text_splits = {str(s).strip().lower() for s in (args.empty_text_splits or [])}

    xlsx_rows: List[Tuple[str, str]] = []
    seen_accessions: set[str] = set()

    for r in _iter_records(
        args.manifest,
        splits=list(args.splits),
        max_records_per_split=int(args.max_records_per_split),
    ):
        split = str(r.get("split", "")).strip().lower()
        split_dir = _split_to_ct2rep_dir(split)
        if split_dir is None:
            continue

        scan_hash = str(r.get("scan_hash", "")).strip()
        if not scan_hash:
            raise SystemExit("Record missing scan_hash")

        patient_id = str(r.get("patient_id", "")).strip() or "unknown_patient"
        vol_path = Path(str(r.get("volume_path", "")).strip())
        if not str(vol_path):
            raise SystemExit(f"Record missing volume_path (scan_hash={scan_hash})")
        if not vol_path.exists():
            raise SystemExit(f"Missing volume_path={vol_path} (scan_hash={scan_hash})")

        accession_id = scan_hash
        split_root = out_root / split_dir
        dst_dir = split_root / patient_id / accession_id
        _ensure_dir(dst_dir)
        dst_file = dst_dir / f"{scan_hash}.npz"
        if args.link_mode == "symlink":
            _safe_symlink(vol_path, dst_file)

        if accession_id not in seen_accessions:
            report_text = str(r.get("report_text", "") or "")
            # NOTE: pandas.read_excel turns empty cells into NaN (float), which breaks CT2Rep's tokenizer.
            # Use a single space so it stays a string but yields zero tokens on split().
            if split in empty_text_splits:
                report_text = " "
            elif report_text.strip() == "":
                report_text = " "
            xlsx_rows.append((accession_id, report_text))
            seen_accessions.add(accession_id)

    try:
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"pandas is required to write Excel: {e}") from e

    df = pd.DataFrame(xlsx_rows, columns=["AccessionNo", "Findings_EN"])
    _ensure_dir(xlsx_path.parent)
    df.to_excel(str(xlsx_path), index=False)

    print(
        json.dumps(
            {
                "out_root": str(out_root),
                "xlsx": str(xlsx_path),
                "n_accessions": int(len(xlsx_rows)),
                "splits": list(args.splits),
                "empty_text_splits": sorted(list(empty_text_splits)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
