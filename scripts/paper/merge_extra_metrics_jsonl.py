#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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


def _metrics_from_row(d: Dict[str, Any]) -> Dict[str, Any]:
    m = d.get("metrics")
    if isinstance(m, dict):
        return dict(m)
    # Fallback: treat remaining keys as metrics.
    out = {k: v for k, v in d.items() if k not in ("sample_id", "scan_hash", "method", "metrics")}
    return out


def _key(d: Dict[str, Any]) -> Tuple[str, str]:
    sid = str(d.get("sample_id") or d.get("scan_hash") or "").strip()
    method = str(d.get("method") or "").strip()
    return sid, method


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Merge two extra_metrics_jsonl files by (sample_id, method), unioning metric keys. "
            "By default, metrics from --b override duplicates from --a."
        )
    )
    ap.add_argument("--a", type=str, required=True, help="First JSONL (base)")
    ap.add_argument("--b", type=str, required=True, help="Second JSONL (overrides)")
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--prefer", type=str, default="b", choices=["a", "b"], help="Which side wins on key collisions")

    args = ap.parse_args()

    a_path = Path(str(args.a)).resolve()
    b_path = Path(str(args.b)).resolve()
    out_path = Path(str(args.out_jsonl)).resolve()

    if not a_path.exists():
        raise FileNotFoundError(str(a_path))
    if not b_path.exists():
        raise FileNotFoundError(str(b_path))

    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def ingest(path: Path, *, overwrite: bool) -> None:
        for d in _read_jsonl(path):
            sid, method = _key(d)
            if not sid or not method:
                continue
            m = _metrics_from_row(d)
            if (sid, method) not in merged:
                merged[(sid, method)] = {}
            if overwrite:
                merged[(sid, method)].update(m)
            else:
                for k, v in m.items():
                    if k not in merged[(sid, method)]:
                        merged[(sid, method)][k] = v

    if str(args.prefer) == "b":
        ingest(a_path, overwrite=True)
        ingest(b_path, overwrite=True)
    else:
        ingest(b_path, overwrite=True)
        ingest(a_path, overwrite=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for (sid, method) in sorted(merged.keys()):
            f.write(json.dumps({"sample_id": sid, "method": method, "metrics": merged[(sid, method)]}, ensure_ascii=False) + "\n")
            rows += 1

    print(f"wrote: {out_path} (rows={rows})")


if __name__ == "__main__":
    main()
