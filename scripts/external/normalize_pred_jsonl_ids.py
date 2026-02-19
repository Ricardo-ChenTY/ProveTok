#!/usr/bin/env python3
from __future__ import annotations

"""Normalize external baseline `pred_jsonl` sample ids to manifest scan_hash.

Motivation (pp.md 5.3 / 6.3): external baselines often emit predictions keyed by
provider-specific ids (series_uid, filename stem, etc.). Our evaluation adapters
expect `sample_id` to match the manifest `scan_hash`.

Input JSONL rows (minimum):
  {"sample_id": "...", "method": "...", "pred_text": "..."}

This script:
- loads a manifest.jsonl;
- builds a best-effort id->scan_hash mapping;
- rewrites `sample_id` to scan_hash when possible;
- drops unmatched/ambiguous rows by default (so eval is explicit).

It is intentionally conservative: if an id maps to multiple scans, it is treated
as ambiguous and not rewritten.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# Ensure repo root is on sys.path when running as `python scripts/external/*.py ...`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.data.manifest_schema import compute_scan_hash, load_manifest


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict):
                yield d


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _stem(p: str) -> str:
    try:
        return Path(str(p)).stem
    except Exception:
        return ""


class _Ambiguous:
    pass


def _add_map(m: Dict[str, object], key: str, scan_hash: str) -> None:
    k = str(key or "").strip()
    if not k:
        return
    cur = m.get(k)
    if cur is None:
        m[k] = str(scan_hash)
        return
    if isinstance(cur, _Ambiguous):
        return
    if str(cur) != str(scan_hash):
        m[k] = _Ambiguous()


def _build_id_map(manifest_path: Path) -> Tuple[Dict[str, object], Dict[str, int]]:
    recs = load_manifest(str(manifest_path))
    id_to_hash: Dict[str, object] = {}

    stats: Dict[str, int] = {
        "n_records": int(len(recs)),
        "n_with_scan_hash": 0,
        "n_with_series_uid": 0,
        "n_with_patient_id": 0,
        "n_with_stem": 0,
    }

    for r in recs:
        sh = str(getattr(r, "scan_hash", "") or "").strip()
        if not sh:
            continue
        stats["n_with_scan_hash"] += 1
        _add_map(id_to_hash, sh, sh)

        su = str(getattr(r, "series_uid", "") or "").strip()
        if su:
            stats["n_with_series_uid"] += 1
            _add_map(id_to_hash, su, sh)

        pid = str(getattr(r, "patient_id", "") or "").strip()
        sd = str(getattr(r, "study_date", "") or "").strip()
        if pid:
            stats["n_with_patient_id"] += 1
        if pid and sd and su:
            _add_map(id_to_hash, compute_scan_hash(pid, sd, su), sh)

        vp = str(getattr(r, "volume_path", "") or "")
        st = _stem(vp)
        if st:
            stats["n_with_stem"] += 1
            _add_map(id_to_hash, st, sh)

    return id_to_hash, stats


def _get_sid(row: Dict[str, Any]) -> str:
    for k in ("sample_id", "scan_hash", "id"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _maybe_compute_hash_from_fields(row: Dict[str, Any]) -> str:
    pid = str(row.get("patient_id") or "").strip()
    sd = str(row.get("study_date") or "").strip()
    su = str(row.get("series_uid") or row.get("series") or "").strip()
    if pid and sd and su:
        return compute_scan_hash(pid, sd, su)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize pred_jsonl sample_id to manifest scan_hash.")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--in-pred-jsonl", type=str, required=True)
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--keep-unmatched", action="store_true", help="Keep rows that could not be mapped (default: drop)")
    args = ap.parse_args()

    manifest_path = Path(str(args.manifest)).expanduser().resolve()
    in_path = Path(str(args.in_pred_jsonl)).expanduser().resolve()
    out_path = Path(str(args.out_jsonl)).expanduser().resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    if not in_path.exists():
        raise SystemExit(f"Missing input JSONL: {in_path}")

    id_to_hash, mstats = _build_id_map(manifest_path)

    n_in = 0
    n_mapped = 0
    n_ambiguous = 0
    n_unmatched = 0
    n_kept_unmatched = 0
    n_dupe = 0

    seen: set[Tuple[str, str]] = set()
    out_rows: List[Dict[str, Any]] = []

    for row in _read_jsonl(in_path):
        n_in += 1
        sid = _get_sid(row)
        method = str(row.get("method") or "").strip()
        if not sid or not method:
            continue

        mapped: Optional[str] = None

        # 1) direct id map (scan_hash / series_uid / stem / computed hash)
        cur = id_to_hash.get(str(sid))
        if isinstance(cur, str) and cur:
            mapped = str(cur)
        elif isinstance(cur, _Ambiguous):
            n_ambiguous += 1

        # 2) compute from patient_id+study_date+series_uid if provided
        if mapped is None:
            h = _maybe_compute_hash_from_fields(row)
            cur2 = id_to_hash.get(h) if h else None
            if isinstance(cur2, str) and cur2:
                mapped = str(cur2)
            elif isinstance(cur2, _Ambiguous):
                n_ambiguous += 1

        if mapped is None:
            n_unmatched += 1
            if not bool(args.keep_unmatched):
                continue
            n_kept_unmatched += 1
            mapped = str(sid)

        key = (str(mapped), str(method))
        if key in seen:
            n_dupe += 1
            continue
        seen.add(key)

        out = dict(row)
        if str(mapped) != str(sid):
            out.setdefault("orig_sample_id", str(sid))
        out["sample_id"] = str(mapped)
        out_rows.append(out)
        if str(mapped) != str(sid):
            n_mapped += 1

    _write_jsonl(out_path, out_rows)

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "in_pred_jsonl": str(in_path),
                "out_jsonl": str(out_path),
                "manifest_stats": mstats,
                "n_in": int(n_in),
                "n_out": int(len(out_rows)),
                "n_mapped": int(n_mapped),
                "n_ambiguous": int(n_ambiguous),
                "n_unmatched": int(n_unmatched),
                "n_kept_unmatched": int(n_kept_unmatched),
                "n_dupe_dropped": int(n_dupe),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
