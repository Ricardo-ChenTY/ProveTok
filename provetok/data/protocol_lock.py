from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .manifest_schema import ManifestRecord, compute_scan_hash


class ProtocolViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class ProtocolLockConfig:
    forbid_splits: bool = True


class ProtocolLock:
    """Implements the "Protocol Lock" rules from the plan (minimal enforceable subset).

    This is intentionally strict and deterministic:
    - scan_hash must be SHA256(patient_id||study_date||series_uid)
    - patient_id is split-exclusive (train/val/test disjoint)
    - scan_hash must be unique
    """

    def __init__(self, cfg: Optional[ProtocolLockConfig] = None):
        self.cfg = cfg or ProtocolLockConfig()

    def validate_or_die(self, records: List[ManifestRecord]) -> None:
        self._validate_scan_hash(records)
        self._validate_unique_scan_hash(records)
        self._validate_patient_level_split(records)

    def _validate_scan_hash(self, records: Iterable[ManifestRecord]) -> None:
        bad: List[Dict[str, Any]] = []
        for r in records:
            expected = compute_scan_hash(r.patient_id, r.study_date, r.series_uid)
            if r.scan_hash != expected:
                bad.append(
                    {
                        "scan_hash": r.scan_hash,
                        "expected": expected,
                        "patient_id": r.patient_id,
                        "study_date": r.study_date,
                        "series_uid": r.series_uid,
                    }
                )
        if bad:
            raise ProtocolViolation(f"scan_hash mismatch for {len(bad)} record(s): {bad[:3]}")

    def _validate_unique_scan_hash(self, records: Iterable[ManifestRecord]) -> None:
        seen = set()
        dupes = []
        for r in records:
            if r.scan_hash in seen:
                dupes.append(r.scan_hash)
            seen.add(r.scan_hash)
        if dupes:
            raise ProtocolViolation(f"Duplicate scan_hash detected (n={len(dupes)}): {dupes[:3]}")

    def _validate_patient_level_split(self, records: Iterable[ManifestRecord]) -> None:
        patient_to_split: Dict[str, str] = {}
        violations: List[Dict[str, str]] = []
        for r in records:
            prev = patient_to_split.get(r.patient_id)
            if prev is None:
                patient_to_split[r.patient_id] = r.split
                continue
            if prev != r.split:
                violations.append({"patient_id": r.patient_id, "split_a": prev, "split_b": r.split})

        if violations:
            raise ProtocolViolation(f"Patient-level split violated for {len(violations)} patient(s): {violations[:3]}")

