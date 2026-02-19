#!/usr/bin/env python3
from __future__ import annotations

"""Convert external baseline outputs into ProveTok `pred_jsonl`.

`provetok.experiments.eval_external_predictions` expects:
  {"sample_id": "<scan_hash>", "method": "<name>", "pred_text": "..."}

This script is intentionally generic: it supports common output shapes so you can
standardize evaluation without integrating the baseline itself.

Supported inputs:
- Directory of text files: each file is one prediction (sample_id = filename stem)
- JSONL: one dict per line
- JSON: list[dict] or dict
- CSV/TSV: header-based columns
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


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


def _detect_format(p: Path) -> str:
    if p.is_dir():
        return "dir"
    suf = "".join(p.suffixes).lower()
    if suf.endswith(".jsonl"):
        return "jsonl"
    if suf.endswith(".json"):
        return "json"
    if suf.endswith(".tsv"):
        return "tsv"
    if suf.endswith(".csv"):
        return "csv"
    return "jsonl"


def _get_sid(d: Dict[str, Any], *, id_key: str) -> str:
    for k in (id_key, "sample_id", "scan_hash", "id"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _get_pred(d: Dict[str, Any], *, pred_key: str) -> str:
    for k in (pred_key, "pred_text", "pred", "prediction", "hyp"):
        v = d.get(k)
        if isinstance(v, str):
            return v
    return ""


def _from_dir(p: Path, *, glob_pat: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for fp in sorted(p.glob(str(glob_pat))):
        if not fp.is_file():
            continue
        sid = fp.stem
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        out.append((sid, txt))
    return out


def _from_jsonl(p: Path, *, id_key: str, pred_key: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for d in _read_jsonl(p):
        sid = _get_sid(d, id_key=str(id_key))
        pred = _get_pred(d, pred_key=str(pred_key))
        if not sid:
            continue
        out.append((sid, pred))
    return out


def _from_json(p: Path, *, id_key: str, pred_key: str) -> List[Tuple[str, str]]:
    d = json.loads(p.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    if isinstance(d, list):
        for row in d:
            if not isinstance(row, dict):
                continue
            sid = _get_sid(row, id_key=str(id_key))
            pred = _get_pred(row, pred_key=str(pred_key))
            if not sid:
                continue
            out.append((sid, pred))
        return out
    if isinstance(d, dict):
        # Either dict[sid]=pred_text or dict with nested objects.
        for k, v in d.items():
            sid = str(k).strip()
            if not sid:
                continue
            if isinstance(v, str):
                out.append((sid, v))
            elif isinstance(v, dict):
                pred = _get_pred(v, pred_key=str(pred_key))
                out.append((sid, pred))
        return out
    raise ValueError("Unsupported JSON shape (expected list or dict)")


def _from_csv(
    p: Path,
    *,
    delimiter: str,
    id_key: str,
    pred_key: str,
    id_col: int,
    pred_col: int,
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        if id_col >= 0 and pred_col >= 0:
            reader = csv.reader(f, delimiter=str(delimiter))
            for row in reader:
                if not row:
                    continue
                if max(id_col, pred_col) >= len(row):
                    continue
                sid = str(row[id_col]).strip()
                pred = str(row[pred_col])
                if not sid:
                    continue
                out.append((sid, pred))
            return out

        reader = csv.DictReader(f, delimiter=str(delimiter))
        for row in reader:
            if not isinstance(row, dict):
                continue
            sid = str(row.get(str(id_key), row.get("sample_id", row.get("scan_hash", ""))) or "").strip()
            pred = str(row.get(str(pred_key), row.get("pred_text", row.get("pred", row.get("prediction", "")))) or "")
            if not sid:
                continue
            out.append((sid, pred))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert external baseline outputs to pred_jsonl for eval_external_predictions.")
    ap.add_argument("--in", dest="in_path", type=str, required=True, help="Input file or directory")
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--method", type=str, required=True, help="Method name to write into each row")
    ap.add_argument("--format", type=str, default="auto", choices=["auto", "dir", "jsonl", "json", "csv", "tsv"])
    ap.add_argument("--id-key", type=str, default="sample_id", help="Key/column for sample id")
    ap.add_argument("--pred-key", type=str, default="pred_text", help="Key/column for prediction text")

    # dir
    ap.add_argument("--glob", type=str, default="*.txt", help="When --format=dir: glob pattern")

    # csv/tsv
    ap.add_argument("--id-col", type=int, default=-1, help="When input has no header: 0-based id column")
    ap.add_argument("--pred-col", type=int, default=-1, help="When input has no header: 0-based pred column")

    ap.add_argument("--strip", action="store_true", help="Strip leading/trailing whitespace from pred_text")
    args = ap.parse_args()

    p = Path(str(args.in_path)).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"Missing input: {p}")

    fmt = str(args.format)
    if fmt == "auto":
        fmt = _detect_format(p)

    pairs: List[Tuple[str, str]]
    if fmt == "dir":
        pairs = _from_dir(p, glob_pat=str(args.glob))
    elif fmt == "json":
        pairs = _from_json(p, id_key=str(args.id_key), pred_key=str(args.pred_key))
    elif fmt == "jsonl":
        pairs = _from_jsonl(p, id_key=str(args.id_key), pred_key=str(args.pred_key))
    elif fmt in ("csv", "tsv"):
        delim = "\t" if fmt == "tsv" else ","
        pairs = _from_csv(
            p,
            delimiter=delim,
            id_key=str(args.id_key),
            pred_key=str(args.pred_key),
            id_col=int(args.id_col),
            pred_col=int(args.pred_col),
        )
    else:
        raise SystemExit(f"Unsupported format: {fmt}")

    if not pairs:
        raise SystemExit("No usable rows found")

    out_rows: List[Dict[str, Any]] = []
    for sid, pred in pairs:
        pred_text = str(pred)
        if bool(args.strip):
            pred_text = pred_text.strip()
        out_rows.append({"sample_id": str(sid), "method": str(args.method), "pred_text": pred_text})

    out_path = Path(str(args.out_jsonl)).expanduser().resolve()
    _write_jsonl(out_path, out_rows)
    print(f"wrote: {out_path} (n={len(out_rows)})")


if __name__ == "__main__":
    main()
