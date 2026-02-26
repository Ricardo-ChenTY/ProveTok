from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

# Keep this script standalone so it can run even when package layout differs
# across environments (Windows local workspace vs Linux server clone).


@dataclass(frozen=True)
class ManifestRecord:
    dataset: str
    scan_hash: str
    split: str
    volume_path: str
    patient_id: str = ""
    study_date: str = ""
    series_uid: str = ""
    report_text: str = ""
    extra: Dict[str, Any] = None  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManifestRecord":
        dd = dict(d or {})
        dataset = str(dd.pop("dataset", "unknown"))
        scan_hash = str(dd.pop("scan_hash", ""))
        split = str(dd.pop("split", "test"))
        volume_path = str(dd.pop("volume_path", dd.pop("volume", "")))
        patient_id = str(dd.pop("patient_id", ""))
        study_date = str(dd.pop("study_date", ""))
        series_uid = str(dd.pop("series_uid", ""))
        report_text = str(dd.pop("report_text", dd.pop("report", "")))
        if not scan_hash:
            scan_hash = str(series_uid or _strip_known_suffixes(Path(volume_path).name) or "")
        return cls(
            dataset=dataset,
            scan_hash=scan_hash,
            split=split,
            volume_path=volume_path,
            patient_id=patient_id,
            study_date=study_date,
            series_uid=series_uid,
            report_text=report_text,
            extra=dd,
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "dataset": self.dataset,
            "scan_hash": self.scan_hash,
            "split": self.split,
            "patient_id": self.patient_id,
            "study_date": self.study_date,
            "series_uid": self.series_uid,
            "volume_path": self.volume_path,
            "report_text": self.report_text,
        }
        out.update(self.extra or {})
        return out


def save_manifest_jsonl(records: Sequence[ManifestRecord], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def compute_manifest_revision(records: Sequence[ManifestRecord]) -> str:
    lines = [json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for r in records]
    lines.sort()
    h = hashlib.sha1()
    for ln in lines:
        h.update(ln.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def find_exact_duplicate_reports(records: Sequence[ManifestRecord]) -> List[List[str]]:
    groups: Dict[str, List[str]] = {}
    for r in records:
        key = " ".join(str(r.report_text or "").strip().split())
        if not key:
            continue
        groups.setdefault(key, []).append(str(r.scan_hash))
    dupes: List[List[str]] = []
    for _, ids in groups.items():
        uniq = sorted({x for x in ids if x})
        if len(uniq) >= 2:
            dupes.append(uniq)
    dupes.sort(key=lambda xs: (-len(xs), xs[0] if xs else ""))
    return dupes


_SPLIT_ALIASES: Dict[str, str] = {
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


def _normalize_split(s: str) -> str:
    return _SPLIT_ALIASES.get(str(s or "").strip().lower(), "")


def _strip_known_suffixes(s: str) -> str:
    t = str(s or "").strip()
    for suf in (".nii.gz", ".nii", ".npz", ".npy"):
        if t.lower().endswith(suf):
            return t[: -len(suf)]
    return t


def _norm_id_keys(raw: str) -> Set[str]:
    """Return join keys for id matching across volume paths and report tables."""
    if not str(raw or "").strip():
        return set()
    x = str(raw).strip()
    p = Path(x)
    out = {
        x,
        x.lower(),
        _strip_known_suffixes(x),
        _strip_known_suffixes(x).lower(),
        p.name,
        p.name.lower(),
        _strip_known_suffixes(p.name),
        _strip_known_suffixes(p.name).lower(),
        p.stem,
        p.stem.lower(),
    }
    return {k for k in out if str(k).strip()}


def _iter_volume_paths(root: Path, patterns: Sequence[str]) -> Iterable[Path]:
    seen: Set[str] = set()
    for pat in patterns:
        for p in root.glob(str(pat)):
            if not p.is_file():
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            yield p.resolve()


def _infer_split_from_path(rel: Path) -> str:
    for part in rel.parts:
        sp = _normalize_split(str(part))
        if sp:
            return sp
    return ""


def _infer_split_from_prefix(name: str) -> str:
    stem = _strip_known_suffixes(name)
    pref = stem.split("_")[0] if "_" in stem else stem
    return _normalize_split(pref)


def _infer_ids_from_filename(name: str) -> Tuple[str, str]:
    """Infer patient_id/series_uid from CT-RATE-style names.

    Example:
    - train_1741_b_2.nii.gz -> patient_id=train_1741, series_uid=train_1741_b
    """
    stem = _strip_known_suffixes(name)
    parts = stem.split("_")
    if len(parts) >= 3:
        patient_id = "_".join(parts[:2])
        series_uid = "_".join(parts[:3])
        return patient_id, series_uid
    if len(parts) == 2:
        return stem, stem
    return stem, stem


def _iter_table_rows(path: Path) -> Iterable[Dict[str, Any]]:
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
        return

    if suf.endswith(".json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for row in obj:
                if isinstance(row, dict):
                    yield row
            return
        if isinstance(obj, dict):
            # Support {rows:[...]} / {data:[...]}
            for key in ("rows", "data", "records"):
                rows = obj.get(key)
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            yield row
                    return
        raise ValueError(f"Unsupported JSON report shape in {path}")

    if suf.endswith(".csv") or suf.endswith(".tsv"):
        delim = "\t" if suf.endswith(".tsv") else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f, delimiter=delim)
            for row in rd:
                if isinstance(row, dict):
                    yield dict(row)
        return

    if suf.endswith(".xlsx") or suf.endswith(".xls"):
        try:
            import pandas as pd  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Reading Excel requires pandas/openpyxl.") from e
        df = pd.read_excel(path)  # type: ignore[no-untyped-call]
        for row in df.to_dict(orient="records"):  # type: ignore[no-untyped-call]
            if isinstance(row, dict):
                yield row
        return

    raise ValueError(f"Unsupported report file format: {path}")


def _first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> str:
    def _canon(k: str) -> str:
        return str(k or "").strip().lstrip("\ufeff").lower()

    for k in keys:
        if k in row and str(row.get(k) or "").strip():
            return str(row.get(k) or "").strip()
    # Case-insensitive fallback
    low = {_canon(str(k)): k for k in row.keys()}
    for k in keys:
        kk = low.get(_canon(str(k)))
        if kk is not None and str(row.get(kk) or "").strip():
            return str(row.get(kk) or "").strip()
    return ""


def _build_report_index(
    rows: Iterable[Dict[str, Any]],
    *,
    id_keys: Sequence[str],
    text_keys: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, int]]:
    id_to_text: Dict[str, str] = {}
    n_rows = 0
    n_with_id = 0
    n_with_text = 0

    for row in rows:
        n_rows += 1
        text = _first_nonempty(row, text_keys)
        if text:
            n_with_text += 1
        rid = _first_nonempty(row, id_keys)
        if not rid:
            continue
        n_with_id += 1
        for k in _norm_id_keys(rid):
            # Keep first occurrence for deterministic behavior.
            if k not in id_to_text:
                id_to_text[k] = text
    return id_to_text, {
        "n_rows": int(n_rows),
        "n_with_id": int(n_with_id),
        "n_with_text": int(n_with_text),
        "n_index_keys": int(len(id_to_text)),
    }


def _sample_per_split(
    records: List[ManifestRecord],
    *,
    max_per_split: int,
    seed: int,
) -> List[ManifestRecord]:
    if int(max_per_split) <= 0:
        return records
    rng = np.random.RandomState(int(seed))
    out: List[ManifestRecord] = []
    for split in ("train", "val", "test"):
        rows = [r for r in records if str(r.split) == split]
        if len(rows) <= int(max_per_split):
            out.extend(rows)
            continue
        idx = rng.choice(len(rows), size=int(max_per_split), replace=False)
        out.extend([rows[int(i)] for i in sorted(idx.tolist())])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build CT-RATE manifest.jsonl from local volume files + optional report table.\n\n"
            "This script does not download data; it only scans local files and writes ProveTok manifest artifacts."
        )
    )
    ap.add_argument("--ct-root", type=str, required=True, help="Root folder containing CT volumes")
    ap.add_argument("--out-manifest", type=str, required=True, help="Output manifest.jsonl")
    ap.add_argument("--dataset-name", type=str, default="ct_rate_local")
    ap.add_argument("--volume-glob", type=str, nargs="+", default=["**/*.nii.gz", "**/*.nii", "**/*.npz"])

    ap.add_argument("--report-file", type=str, default="", help="Optional report table (.csv/.tsv/.xlsx/.json/.jsonl)")
    ap.add_argument(
        "--report-id-keys",
        type=str,
        nargs="+",
        default=["volume_name", "VolumeName", "volume", "file_name", "filename", "scan_hash", "sample_id", "AccessionNo"],
        help="Candidate id columns used to join reports with volume files",
    )
    ap.add_argument(
        "--report-text-keys",
        type=str,
        nargs="+",
        default=["report_text", "report", "Report", "findings", "Findings", "Findings_EN", "text"],
        help="Candidate text columns for report_text",
    )
    ap.add_argument("--report-default-text", type=str, default="", help="Fallback text when report row is missing")
    ap.add_argument("--require-report-text", action="store_true", help="Drop rows whose report_text is empty after matching")

    ap.add_argument("--split-from", type=str, choices=["path", "prefix", "none"], default="path")
    ap.add_argument("--set-split", type=str, default="", choices=["", "train", "val", "test"])
    ap.add_argument("--keep-splits", type=str, nargs="+", default=["train", "val", "test"])

    ap.add_argument("--max-records", type=int, default=0, help="Optional global cap after filtering (0 means all)")
    ap.add_argument("--max-per-split", type=int, default=0, help="Optional cap per split")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    ct_root = Path(args.ct_root).resolve()
    out_manifest = Path(args.out_manifest).resolve()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    keep_splits = {_normalize_split(s) for s in (args.keep_splits or [])}
    keep_splits = {s for s in keep_splits if s}
    if not ct_root.exists():
        raise SystemExit(f"--ct-root not found: {ct_root}")

    # 1) Load report index (optional).
    report_index: Dict[str, str] = {}
    report_stats: Dict[str, int] = {"n_rows": 0, "n_with_id": 0, "n_with_text": 0, "n_index_keys": 0}
    if str(args.report_file).strip():
        rp = Path(str(args.report_file)).resolve()
        if not rp.exists():
            raise SystemExit(f"--report-file not found: {rp}")
        report_index, report_stats = _build_report_index(
            _iter_table_rows(rp),
            id_keys=[str(x) for x in args.report_id_keys],
            text_keys=[str(x) for x in args.report_text_keys],
        )

    # 2) Scan volumes and build records.
    records: List[ManifestRecord] = []
    n_missing_text = 0
    n_report_matched = 0
    n_scan_hash_collisions = 0
    seen_scan_hash: Dict[str, int] = {}

    paths = sorted(_iter_volume_paths(ct_root, [str(x) for x in args.volume_glob]), key=lambda p: str(p))
    for p in paths:
        rel = p.relative_to(ct_root)
        stem = _strip_known_suffixes(p.name)

        if str(args.set_split):
            split = str(args.set_split)
        elif str(args.split_from) == "path":
            split = _infer_split_from_path(rel) or "test"
        elif str(args.split_from) == "prefix":
            split = _infer_split_from_prefix(p.name) or "test"
        else:
            split = "test"
        split = _normalize_split(split) or "test"
        if keep_splits and split not in keep_splits:
            continue

        patient_id, series_uid = _infer_ids_from_filename(p.name)
        # Prefer path hints when layout is .../<split>/<patient>/<series>/<file>.
        if len(rel.parts) >= 3:
            patient_hint = str(rel.parts[-3]).strip()
            series_hint = str(rel.parts[-2]).strip()
            if patient_hint and patient_hint.lower() not in {"dataset", split}:
                patient_id = patient_hint
            if series_hint:
                series_uid = series_hint

        # Join report text by several candidate ids.
        report_text = str(args.report_default_text or "")
        keys = set()
        keys.update(_norm_id_keys(p.name))
        keys.update(_norm_id_keys(stem))
        keys.update(_norm_id_keys(series_uid))
        keys.update(_norm_id_keys(patient_id))
        for k in keys:
            if k in report_index:
                report_text = str(report_index[k] or "")
                n_report_matched += 1
                break
        if not report_text.strip():
            n_missing_text += 1
            if bool(args.require_report_text):
                continue

        scan_hash = str(series_uid or stem)
        cur = seen_scan_hash.get(scan_hash, 0)
        seen_scan_hash[scan_hash] = cur + 1
        if cur > 0:
            n_scan_hash_collisions += 1
            scan_hash = f"{scan_hash}_{cur+1}"

        d: Dict[str, Any] = {
            "dataset": str(args.dataset_name),
            "scan_hash": scan_hash,
            "split": split,
            "patient_id": str(patient_id),
            "study_date": "",
            "series_uid": str(series_uid),
            "volume_path": str(p),
            "report_text": str(report_text),
            "volume_name": str(p.name),
            "volume_relpath": str(rel).replace("\\", "/"),
        }
        records.append(ManifestRecord.from_dict(d))

    if not records:
        raise SystemExit("No records found. Check --ct-root, --volume-glob, and split filters.")

    # Optional down-sampling.
    if int(args.max_per_split) > 0:
        records = _sample_per_split(records, max_per_split=int(args.max_per_split), seed=int(args.seed))
        records = sorted(records, key=lambda r: (str(r.split), str(r.scan_hash)))
    if int(args.max_records) > 0:
        records = records[: int(args.max_records)]

    save_manifest_jsonl(records, str(out_manifest))
    revision = compute_manifest_revision(records)
    dupes = find_exact_duplicate_reports(records)
    split_manifest = {
        "train": sorted([r.scan_hash for r in records if r.split == "train"]),
        "val": sorted([r.scan_hash for r in records if r.split == "val"]),
        "test": sorted([r.scan_hash for r in records if r.split == "test"]),
    }

    meta: Dict[str, Any] = {
        "ct_root": str(ct_root),
        "out_manifest": str(out_manifest),
        "dataset_name": str(args.dataset_name),
        "volume_glob": [str(x) for x in args.volume_glob],
        "report_file": str(args.report_file or ""),
        "report_id_keys": [str(x) for x in args.report_id_keys],
        "report_text_keys": [str(x) for x in args.report_text_keys],
        "report_stats": report_stats,
        "report_default_text": str(args.report_default_text or ""),
        "require_report_text": bool(args.require_report_text),
        "split_from": str(args.split_from),
        "set_split": str(args.set_split or ""),
        "keep_splits": sorted(list(keep_splits)),
        "max_records": int(args.max_records),
        "max_per_split": int(args.max_per_split),
        "seed": int(args.seed),
        "num_records": int(len(records)),
        "num_missing_text": int(n_missing_text),
        "num_report_matched": int(n_report_matched),
        "num_scan_hash_collisions_resolved": int(n_scan_hash_collisions),
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
                "num_records": int(len(records)),
                "split_counts": {k: len(v) for k, v in split_manifest.items()},
                "num_report_matched": int(n_report_matched),
                "num_missing_text": int(n_missing_text),
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
