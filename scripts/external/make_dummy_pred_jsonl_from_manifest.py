#!/usr/bin/env python3
from __future__ import annotations

"""Generate a minimal `pred_jsonl` from a ProveTok manifest.

This is a utility for smoke-testing the external-baseline evaluation pipeline
(pp.md §6.3) before you have real external model outputs.

Output JSONL rows follow:
  {"sample_id": "<scan_hash>", "method": "dummy", "pred_text": "..."}
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple


def _iter_manifest_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if isinstance(d, dict):
                yield d


def _get_scan_hash(d: Dict[str, Any]) -> str:
    for k in ("scan_hash", "sample_id", "id"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Fallback: filename stem (keeps script usable for ad-hoc manifests).
    vp = d.get("volume_path") or d.get("volume")
    if isinstance(vp, str) and vp:
        try:
            return Path(vp).stem
        except Exception:
            return ""
    return ""


def _get_report_text(d: Dict[str, Any]) -> str:
    for k in ("report_text", "report"):
        v = d.get(k)
        if isinstance(v, str):
            return v
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a dummy pred_jsonl from a manifest JSONL.")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--method", type=str, default="dummy")
    ap.add_argument("--split", type=str, default="", help="Optional split filter: train|val|test")
    ap.add_argument("--max-records", type=int, default=0, help="0 means no limit")
    ap.add_argument("--pred-source", type=str, default="reference", choices=["reference", "constant"])
    ap.add_argument("--constant-text", type=str, default="No acute findings.")
    args = ap.parse_args()

    manifest_path = Path(str(args.manifest)).expanduser().resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    out_path = Path(str(args.out_jsonl)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    want_split = str(args.split).strip()
    limit = int(args.max_records)
    n = 0

    with out_path.open("w", encoding="utf-8") as f:
        for d in _iter_manifest_jsonl(manifest_path):
            if want_split:
                sp = str(d.get("split") or "").strip()
                if sp != want_split:
                    continue

            sid = _get_scan_hash(d)
            if not sid:
                continue

            if str(args.pred_source) == "reference":
                pred = _get_report_text(d) or str(args.constant_text)
            else:
                pred = str(args.constant_text)

            row = {"sample_id": sid, "method": str(args.method), "pred_text": str(pred)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if limit > 0 and n >= limit:
                break

    print(f"wrote: {out_path} (n={n})")


if __name__ == "__main__":
    main()
