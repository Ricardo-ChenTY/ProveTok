from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence


def compute_scan_hash(patient_id: str, study_date: str, series_uid: str) -> str:
    """Compute a strict, protocol-locked scan hash.

    This implements the ProtocolLock convention used by some manifest builders:
    SHA256(patient_id||study_date||series_uid) as a hex string.

    The intent is to produce a stable, split-safe identifier that does not
    depend on filesystem paths.
    """
    h = hashlib.sha256()
    h.update(str(patient_id or "").encode("utf-8"))
    h.update(b"||")
    h.update(str(study_date or "").encode("utf-8"))
    h.update(b"||")
    h.update(str(series_uid or "").encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class ManifestRecord:
    """A minimal, forward-compatible manifest record.

    The repo uses manifest-driven datasets for most experiments. Keep this schema
    permissive and store unknown fields under `extra` so external builders can
    evolve without breaking consumers.
    """

    dataset: str
    scan_hash: str
    split: str
    volume_path: str
    # Optional identifiers (commonly available in public CT datasets).
    patient_id: str = ""
    study_date: str = ""  # YYYYMMDD or provider-specific string
    series_uid: str = ""
    report_text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, default_dataset: str = "unknown") -> "ManifestRecord":
        dd = dict(d or {})
        dataset = str(dd.pop("dataset", default_dataset))
        split = str(dd.pop("split", "test"))
        patient_id = str(dd.pop("patient_id", ""))
        study_date = str(dd.pop("study_date", ""))
        series_uid = str(dd.pop("series_uid", ""))
        volume_path = str(dd.pop("volume_path", dd.pop("volume", "")))
        report_text = str(dd.pop("report_text", dd.pop("report", "")))
        # Keep common compatibility aliases under extra.
        if "mask_path" not in dd and "lesion_mask_path" in dd:
            dd["mask_path"] = dd.pop("lesion_mask_path")

        scan_hash = str(dd.pop("scan_hash", dd.pop("sample_id", "")))
        if not scan_hash and patient_id and study_date and series_uid:
            scan_hash = compute_scan_hash(patient_id, study_date, series_uid)
        if not scan_hash:
            # Prefer a stable per-series id when present.
            scan_hash = str(series_uid or "")
        if not scan_hash and volume_path:
            try:
                scan_hash = Path(volume_path).stem
            except Exception:
                scan_hash = ""
        if not scan_hash:
            # Last resort: content hash.
            h = hashlib.sha1()
            h.update(str(dataset).encode("utf-8"))
            h.update(b"\n")
            h.update(str(split).encode("utf-8"))
            h.update(b"\n")
            h.update(str(patient_id).encode("utf-8"))
            h.update(b"\n")
            h.update(str(series_uid).encode("utf-8"))
            h.update(b"\n")
            h.update(str(volume_path).encode("utf-8"))
            h.update(b"\n")
            h.update(str(report_text).encode("utf-8"))
            scan_hash = h.hexdigest()[:16]
        return cls(
            dataset=dataset,
            scan_hash=scan_hash,
            split=split,
            patient_id=patient_id,
            study_date=study_date,
            series_uid=series_uid,
            volume_path=volume_path,
            report_text=report_text,
            extra=dd,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
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


def iter_manifest(path: str) -> Iterator[ManifestRecord]:
    p = Path(str(path))
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not isinstance(d, dict):
                continue
            yield ManifestRecord.from_dict(d, default_dataset=str(p.parent.name))


def load_manifest(path: str) -> List[ManifestRecord]:
    return list(iter_manifest(path))


def split_records(records: List[ManifestRecord], *, split: str) -> List[ManifestRecord]:
    want = str(split)
    return [r for r in records if str(r.split) == want]


def get_record_mask_path(rec: ManifestRecord) -> Optional[str]:
    """Best-effort mask path resolution.

    Conventions:
    - `extra["mask_path"]` (single 3D mask or a 4D stack)
    - `extra["mask_paths"]` or `extra["lesion_mask_paths"]` (list)
    """

    extra = rec.extra or {}
    mp = extra.get("mask_path")
    if isinstance(mp, str) and mp:
        return mp
    mps = extra.get("mask_paths") or extra.get("lesion_mask_paths")
    if isinstance(mps, list) and mps:
        # Keep compatibility with code expecting a single path.
        if isinstance(mps[0], str):
            return str(mps[0])
    return None


def save_manifest_jsonl(records: Sequence[ManifestRecord], path: str) -> None:
    """Write a manifest JSONL in a stable, tool-friendly format."""
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            d = r.to_dict()
            f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")


def compute_manifest_revision(records: Sequence[ManifestRecord]) -> str:
    """Compute a stable content hash for a manifest.

    This is used in `scripts/data/build_*_manifest.py` meta artifacts so that
    experiment outputs can record the exact data version.
    """
    lines: List[str] = []
    for r in records:
        try:
            d = r.to_dict()
        except Exception:
            d = {}
        lines.append(json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    # Order-independent hash: sort lines.
    lines.sort()
    h = hashlib.sha1()
    for ln in lines:
        h.update(ln.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def find_exact_duplicate_reports(records: Sequence[ManifestRecord]) -> List[List[str]]:
    """Find groups of records with the exact same (normalized) report_text."""
    groups: Dict[str, List[str]] = {}
    for r in records:
        key = " ".join(str(getattr(r, "report_text", "") or "").strip().split())
        if not key:
            continue
        groups.setdefault(key, []).append(str(getattr(r, "scan_hash", "")))

    dupes: List[List[str]] = []
    for _, ids in groups.items():
        uniq = sorted({x for x in ids if x})
        if len(uniq) >= 2:
            dupes.append(uniq)
    dupes.sort(key=lambda xs: (-len(xs), xs[0] if xs else ""))
    return dupes



def summarize_manifest(records: Sequence[ManifestRecord]) -> Dict[str, Any]:
    """Return a lightweight JSON-serializable summary for sanity checks."""

    recs = list(records)

    ds = Counter()
    sp = Counter()
    patients = set()
    n_report = 0
    n_masks = 0

    for r in recs:
        ds[str(getattr(r, "dataset", ""))] += 1
        sp[str(getattr(r, "split", ""))] += 1
        pid = str(getattr(r, "patient_id", "") or "")
        if pid:
            patients.add(pid)
        if str(getattr(r, "report_text", "") or "").strip():
            n_report += 1
        if get_record_mask_path(r):
            n_masks += 1

    dupes = find_exact_duplicate_reports(recs)
    top_dupes = dupes[:3]
    max_dupe = max((len(g) for g in dupes), default=0)

    return {
        "num_records": int(len(recs)),
        "datasets": dict(ds),
        "splits": dict(sp),
        "num_patients": int(len(patients)),
        "num_with_report_text": int(n_report),
        "num_with_masks": int(n_masks),
        "manifest_revision": compute_manifest_revision(recs),
        "num_exact_duplicate_report_groups": int(len(dupes)),
        "max_exact_duplicate_report_group_size": int(max_dupe),
        "top_exact_duplicate_report_groups": top_dupes,
    }
