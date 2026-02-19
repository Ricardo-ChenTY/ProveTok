#!/usr/bin/env python3
from __future__ import annotations

"""Compute selected text metrics via RadEval and export `extra_metrics_jsonl`.

This script is meant to be run in a *separate* environment where RadEval and its
heavy dependencies are installed (RadEval currently targets Python 3.11).

Input format: a JSONL produced by `run_baselines --dump-text-pairs-jsonl`:
  {"sample_id": "...", "method": "...", "pred_text": "...", "ref_text": "..."}

Output format: JSONL suitable for `scripts/paper/compute_paper_metrics.py --extra-metrics-jsonl`:
  {"sample_id": "...", "method": "...", "metrics": { ... }}

Metrics supported (via RadEval):
- GREEN: key `green`
- RadCliQ v1: key `radcliq` (RadEval uses `radcliq-v1` internally)
- CheXbert: per-sample report-level F1 plus repeated dataset-level micro/macro
  summary keys for easy Table1 aggregation.

Notes:
- We run RadEval in `do_details=True` mode to obtain per-sample scores.
- CheXbert output structure in RadEval:
  - `scores["chexbert"]["sample_scores"]` is a dict with keys `all_labels` and `5_labels`.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


@dataclass(frozen=True)
class PairRow:
    sample_id: str
    method: str
    pred_text: str
    ref_text: str


def _load_pairs(path: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    for d in _read_jsonl(path):
        sid = str(d.get("sample_id") or d.get("scan_hash") or "").strip()
        method = str(d.get("method") or "").strip()
        pred = str(d.get("pred_text") or d.get("pred") or "")
        ref = str(d.get("ref_text") or d.get("ref") or "")
        if not sid or not method:
            continue
        rows.append(PairRow(sample_id=sid, method=method, pred_text=pred, ref_text=ref))
    if not rows:
        raise SystemExit(f"No valid rows found in {path}")
    return rows


def _try_import_radeval():
    try:
        # RadEval package name is capitalized in the upstream repo.
        from RadEval import RadEval  # type: ignore

        return RadEval
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Missing dependency: RadEval. "
            "Run this script inside a RadEval-capable environment (typically Python 3.11). "
            f"Import error: {type(e).__name__}: {e}"
        ) from e


def _f(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute GREEN/RadCliQ/CheXbert via RadEval and export extra_metrics_jsonl.")
    ap.add_argument("--text-pairs-jsonl", type=str, required=True)
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--methods", type=str, nargs="+", default=[], help="Optional subset of methods")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap on total rows (after filtering)")

    ap.add_argument("--skip-green", action="store_true")
    ap.add_argument("--skip-radcliq", action="store_true")
    ap.add_argument("--skip-chexbert", action="store_true")

    args = ap.parse_args()

    in_path = Path(str(args.text_pairs_jsonl)).resolve()
    out_path = Path(str(args.out_jsonl)).resolve()

    rows = _load_pairs(in_path)
    if args.methods:
        allow = {str(m) for m in args.methods}
        rows = [r for r in rows if r.method in allow]

    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]

    by_method: Dict[str, List[PairRow]] = {}
    for r in rows:
        by_method.setdefault(r.method, []).append(r)

    do_green = not bool(args.skip_green)
    do_radcliq = not bool(args.skip_radcliq)
    do_chexbert = not bool(args.skip_chexbert)

    if not do_green and not do_radcliq and not do_chexbert:
        raise SystemExit("Nothing to compute: all metrics skipped")

    RadEval = _try_import_radeval()

    evaluator = RadEval(
        do_green=bool(do_green),
        do_radcliq=bool(do_radcliq),
        do_chexbert=bool(do_chexbert),
        do_details=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for method, mr in sorted(by_method.items(), key=lambda kv: kv[0]):
            refs = [x.ref_text for x in mr]
            hyps = [x.pred_text for x in mr]

            scores: Dict[str, Any] = evaluator(refs=refs, hyps=hyps)

            green_scores: Optional[List[float]] = None
            if do_green:
                g = scores.get("green")
                if isinstance(g, dict) and isinstance(g.get("sample_scores"), list):
                    green_scores = [float(x) for x in g.get("sample_scores")]
                else:
                    raise RuntimeError(f"Unexpected RadEval green output for method={method!r}: {type(g).__name__}")

            radcliq_scores: Optional[List[float]] = None
            if do_radcliq:
                rc = scores.get("radcliq-v1")
                if isinstance(rc, dict) and isinstance(rc.get("sample_scores"), list):
                    radcliq_scores = [float(x) for x in rc.get("sample_scores")]
                else:
                    raise RuntimeError(f"Unexpected RadEval radcliq output for method={method!r}: {type(rc).__name__}")

            chex_all: Optional[List[float]] = None
            chex_5: Optional[List[float]] = None
            chex_summary: Dict[str, float] = {}
            if do_chexbert:
                cb = scores.get("chexbert")
                if not isinstance(cb, dict):
                    raise RuntimeError(f"Unexpected RadEval chexbert output for method={method!r}: {type(cb).__name__}")

                ss = cb.get("sample_scores")
                if not isinstance(ss, dict):
                    raise RuntimeError(f"Unexpected RadEval chexbert.sample_scores type for method={method!r}: {type(ss).__name__}")

                all_labels = ss.get("all_labels")
                five_labels = ss.get("5_labels")
                if not isinstance(all_labels, list) or not isinstance(five_labels, list):
                    raise RuntimeError("Unexpected RadEval chexbert sample_scores shape")

                chex_all = [float(x) for x in all_labels]
                chex_5 = [float(x) for x in five_labels]

                # Dataset-level summary numbers (repeat per sample so compute_paper_metrics can aggregate).
                mapping = {
                    "chexbert_all_micro_f1": "chexbert-all_micro avg_f1-score",
                    "chexbert_all_macro_f1": "chexbert-all_macro avg_f1-score",
                    "chexbert_all_weighted_f1": "chexbert-all_weighted_f1",
                    "chexbert_5_micro_f1": "chexbert-5_micro avg_f1-score",
                    "chexbert_5_macro_f1": "chexbert-5_macro avg_f1-score",
                    "chexbert_5_weighted_f1": "chexbert-5_weighted_f1",
                }
                for out_k, in_k in mapping.items():
                    v = _f(cb.get(in_k))
                    if v is not None:
                        chex_summary[out_k] = float(v)

            if green_scores is not None and len(green_scores) != len(mr):
                raise RuntimeError(f"GREEN length mismatch: {len(green_scores)} vs rows={len(mr)}")
            if radcliq_scores is not None and len(radcliq_scores) != len(mr):
                raise RuntimeError(f"RadCliQ length mismatch: {len(radcliq_scores)} vs rows={len(mr)}")
            if chex_all is not None and len(chex_all) != len(mr):
                raise RuntimeError(f"CheXbert(all) length mismatch: {len(chex_all)} vs rows={len(mr)}")
            if chex_5 is not None and len(chex_5) != len(mr):
                raise RuntimeError(f"CheXbert(5) length mismatch: {len(chex_5)} vs rows={len(mr)}")

            for i, r in enumerate(mr):
                metrics: Dict[str, Any] = {}
                if green_scores is not None:
                    metrics["green"] = float(green_scores[i])
                if radcliq_scores is not None:
                    metrics["radcliq"] = float(radcliq_scores[i])

                if chex_all is not None and chex_5 is not None:
                    metrics["chexbert_f1_all"] = float(chex_all[i])
                    metrics["chexbert_f1_5"] = float(chex_5[i])
                    metrics.update(chex_summary)

                rec = {
                    "sample_id": r.sample_id,
                    "method": r.method,
                    "metrics": metrics,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[ok] method={method} rows={len(mr)}")

    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
