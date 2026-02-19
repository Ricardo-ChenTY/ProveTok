#!/usr/bin/env python3
from __future__ import annotations

"""One-click Table1 pipeline on CT-RATE manifests (pp.md §6.4).

This script is glue. It does NOT run external baselines inside this repo.
Instead, it:
1) merges an existing ours `pairs.jsonl` (optional) with external `pred_jsonl` files;
2) optionally computes GREEN/RadCliQ/CheXbert via RadEval (Py3.11 env);
3) optionally computes proof metrics for *external* methods via post-hoc citations;
4) runs `scripts/paper/compute_paper_metrics.py` to produce paper-grade stats.

External predictions format (pred_jsonl):
  {"sample_id": "<scan_hash>", "method": "diallama", "pred_text": "..."}

Pairs format (pairs.jsonl):
  {"sample_id": "...", "method": "...", "pred_text": "...", "ref_text": "..."}

Outputs under --out-dir:
- pairs_external.jsonl
- pairs_all.jsonl
- (optional) extra_radeval.jsonl
- (optional) extra_proof_external.jsonl
- extra_merged.jsonl (if any extra inputs exist)
- paper_metrics.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.data.manifest_schema import load_manifest


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


def _atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _key(d: Dict[str, Any]) -> Tuple[str, str]:
    sid = str(d.get("sample_id") or d.get("scan_hash") or "").strip()
    method = str(d.get("method") or "").strip()
    return sid, method


@dataclass(frozen=True)
class _PredRow:
    sample_id: str
    method: str
    pred_text: str


def _load_pred_jsonl(paths: Sequence[Path]) -> List[_PredRow]:
    rows: List[_PredRow] = []
    for p in paths:
        for d in _read_jsonl(p):
            sid = str(d.get("sample_id") or d.get("scan_hash") or "").strip()
            method = str(d.get("method") or "").strip()
            pred = str(d.get("pred_text") or d.get("pred") or "")
            if not sid or not method:
                continue
            rows.append(_PredRow(sample_id=sid, method=method, pred_text=pred))
    if not rows:
        raise SystemExit(f"No usable rows found in pred_jsonl: {[str(p) for p in paths]}")
    return rows


def _load_pairs_jsonl(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        for d in _read_jsonl(p):
            sid = str(d.get("sample_id") or d.get("scan_hash") or "").strip()
            method = str(d.get("method") or "").strip()
            pred = str(d.get("pred_text") or d.get("pred") or "")
            ref = str(d.get("ref_text") or d.get("ref") or "")
            if not sid or not method:
                continue
            rows.append({"sample_id": sid, "method": method, "pred_text": pred, "ref_text": ref})
    if not rows:
        raise SystemExit(f"No usable rows found in pairs_jsonl: {[str(p) for p in paths]}")
    return rows


def _merge_extra_metrics_jsonl(paths: Sequence[Path], *, out_path: Path) -> Optional[Path]:
    # Merge by (sample_id, method). Later files override key collisions.
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def metrics_from_row(d: Dict[str, Any]) -> Dict[str, Any]:
        m = d.get("metrics")
        if isinstance(m, dict):
            return dict(m)
        return {k: v for k, v in d.items() if k not in ("sample_id", "scan_hash", "method", "metrics")}

    any_rows = 0
    for p in paths:
        for d in _read_jsonl(p):
            sid, method = _key(d)
            if not sid or not method:
                continue
            any_rows += 1
            merged.setdefault((sid, method), {}).update(metrics_from_row(d))

    if any_rows == 0:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for (sid, method) in sorted(merged.keys()):
            f.write(json.dumps({"sample_id": sid, "method": method, "metrics": merged[(sid, method)]}, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    return out_path


def _run(cmd: List[str], *, env: Optional[Dict[str, str]] = None) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _export_proof_extra(eval_json: Path, *, out_jsonl: Path, include: Optional[Sequence[str]] = None) -> None:
    rep = json.loads(eval_json.read_text(encoding="utf-8"))
    per = rep.get("per_sample") or []
    if not isinstance(per, list) or not per:
        raise SystemExit(f"No per_sample rows in {eval_json}")

    keys = list(include) if include else [
        "r1_no_citation",
        "r2_coarse_only",
        "r3_laterality_mismatch",
        "r4_bilateral_separation",
        "weighted_issue",
        "unsupported",
        "overclaim",
        "n_frames_pred",
        "n_citations",
        "warm_time_s",
    ]

    rows: List[Dict[str, Any]] = []
    for d in per:
        if not isinstance(d, dict):
            continue
        sid = str(d.get("sample_id") or "").strip()
        method = str(d.get("method") or "").strip()
        if not sid or not method:
            continue
        m: Dict[str, Any] = {}
        for k in keys:
            if k in d:
                v = d.get(k)
                if isinstance(v, (int, float)):
                    m[k] = float(v)
        rows.append({"sample_id": sid, "method": method, "metrics": m})

    _atomic_write_jsonl(out_jsonl, rows)




def _find_eval_external_predictions_json(proof_out_dir: Path) -> Path:
    """Locate eval_external_predictions.json produced by make_output_dir().

    Newer experiment entrypoints create a timestamped subdirectory like:
      <proof_out_dir>/eval_external_predictions_YYYYMMDD_HHMMSS/eval_external_predictions.json

    Older code may write directly under <proof_out_dir>.
    """

    direct = proof_out_dir / "eval_external_predictions.json"
    if direct.exists():
        return direct

    candidates = sorted(proof_out_dir.glob("eval_external_predictions_*/eval_external_predictions.json"))
    if candidates:
        return candidates[-1]

    deep = sorted(proof_out_dir.glob("**/eval_external_predictions.json"))
    if deep:
        return deep[-1]

    raise SystemExit(
        "Missing expected output under "
        + str(proof_out_dir)
        + " (expected eval_external_predictions.json or eval_external_predictions_*/eval_external_predictions.json)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CT-RATE Table1 pipeline (pairs + optional RadEval + optional proof extra + paper metrics).")

    ap.add_argument("--manifest", type=str, required=True, help="Manifest jsonl (must contain report_text)")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap on number of samples (debug/smoke)")

    ap.add_argument("--base-pairs-jsonl", type=str, nargs="*", default=[], help="Optional existing pairs.jsonl (e.g., ours from run_baselines)")
    ap.add_argument("--external-pred-jsonl", type=str, nargs="*", default=[], help="Optional external predictions pred_jsonl files")

    ap.add_argument("--out-dir", type=str, default="", help="Output directory (default: outputs/table1_ct_rate_<timestamp>)")

    ap.add_argument("--allow-partial", action="store_true", help="Allow methods with incomplete coverage (NOT recommended for SOTA claims)")

    # Optional RadEval
    ap.add_argument("--run-radeval", action="store_true", help="Compute GREEN/RadCliQ/CheXbert via RadEval (Py3.11 env)")
    ap.add_argument("--radeval-env", type=str, default="/data/conda_envs/radeval311", help="Conda env path for RadEval")
    ap.add_argument("--radeval-skip-green", action="store_true")
    ap.add_argument("--radeval-skip-radcliq", action="store_true")
    ap.add_argument("--radeval-skip-chexbert", action="store_true")

    # Optional proof metrics for external methods
    ap.add_argument("--run-proof-external", action="store_true", help="Run eval_external_predictions on external methods and export proof extra_metrics_jsonl")
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--tokenizer", type=str, default="fixed_grid", choices=["fixed_grid", "bet_alg"])
    ap.add_argument("--budget-tokens", type=int, default=256)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--l-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)

    # compute_paper_metrics args (core)
    ap.add_argument("--baseline-method", type=str, default="", help="Baseline method name for paired comparisons")
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--holm-family", type=str, default="all", choices=["all", "per_metric"])

    # compute_paper_metrics args (optional heavy)
    ap.add_argument("--radgraph-model-type", type=str, default="")
    ap.add_argument("--radgraph-cuda", type=int, default=None)
    ap.add_argument("--ratescore", action="store_true")
    ap.add_argument("--ratescore-use-gpu", action="store_true")

    # Extra metrics input
    ap.add_argument("--extra-metrics-jsonl", type=str, nargs="*", default=[], help="Optional extra_metrics_jsonl files to merge")

    args = ap.parse_args()

    manifest_path = Path(str(args.manifest)).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"--manifest not found: {manifest_path}")

    recs = [r for r in load_manifest(str(manifest_path)) if str(r.split) == str(args.split)]
    recs = sorted(recs, key=lambda r: str(r.scan_hash))
    if int(args.max_samples) > 0:
        recs = recs[: int(args.max_samples)]

    if not recs:
        raise SystemExit(f"No records found for split={args.split!r} in {manifest_path}")

    sample_ids = [str(r.scan_hash) for r in recs]
    ref_map = {str(r.scan_hash): str(r.report_text or "") for r in recs}

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (ROOT / "outputs" / f"table1_ct_rate_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load base pairs (ours)
    base_pairs_paths = [Path(p).resolve() for p in (args.base_pairs_jsonl or []) if str(p).strip()]
    base_pairs: List[Dict[str, Any]] = []
    if base_pairs_paths:
        for p in base_pairs_paths:
            if not p.exists():
                raise SystemExit(f"--base-pairs-jsonl not found: {p}")
        base_pairs = _load_pairs_jsonl(base_pairs_paths)
        # Filter to current split sample_ids for safety.
        allow = set(sample_ids)
        base_pairs = [r for r in base_pairs if str(r.get("sample_id")) in allow]

    # 2) Build external pairs from pred_jsonl
    ext_pred_paths = [Path(p).resolve() for p in (args.external_pred_jsonl or []) if str(p).strip()]
    ext_pairs_rows: List[Dict[str, Any]] = []
    merged_external_pred_jsonl: Optional[Path] = None

    if ext_pred_paths:
        for p in ext_pred_paths:
            if not p.exists():
                raise SystemExit(f"--external-pred-jsonl not found: {p}")
        pred_rows = _load_pred_jsonl(ext_pred_paths)

        # Merge preds (keep first by (sid, method)).
        pred_map: Dict[Tuple[str, str], str] = {}
        for r in pred_rows:
            k = (r.sample_id, r.method)
            if k not in pred_map:
                pred_map[k] = r.pred_text

        # Coverage check per method.
        methods = sorted({m for (_, m) in pred_map.keys()})
        sid_set = set(sample_ids)
        for m in methods:
            have = {sid for (sid, mm) in pred_map.keys() if mm == m and sid in sid_set}
            if (not bool(args.allow_partial)) and len(have) != len(sample_ids):
                missing = len(sample_ids) - len(have)
                raise SystemExit(
                    f"External method={m!r} incomplete coverage on split={args.split!r}: "
                    f"have={len(have)} expected={len(sample_ids)} missing={missing}. "
                    f"(hint: run normalize_pred_jsonl_ids.py; or pass --allow-partial for debug)"
                )

        # Write merged external pred jsonl (useful for eval_external_predictions).
        merged_external_pred_jsonl = out_dir / "preds_external_merged.jsonl"
        pred_dump = [
            {"sample_id": sid, "method": method, "pred_text": pred}
            for (sid, method), pred in sorted(pred_map.items(), key=lambda kv: (kv[0][1], kv[0][0]))
            if sid in sid_set
        ]
        _atomic_write_jsonl(merged_external_pred_jsonl, pred_dump)

        # Create pairs_external.jsonl aligned to manifest.
        for m in methods:
            for sid in sample_ids:
                pred = pred_map.get((sid, m), "")
                ext_pairs_rows.append(
                    {
                        "sample_id": sid,
                        "method": m,
                        "pred_text": pred,
                        "ref_text": ref_map.get(sid, ""),
                    }
                )

    pairs_external_path = out_dir / "pairs_external.jsonl"
    if ext_pairs_rows:
        _atomic_write_jsonl(pairs_external_path, ext_pairs_rows)

    # 3) Merge pairs (base wins on collisions)
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in (ext_pairs_rows or []):
        sid = str(r.get("sample_id") or "")
        method = str(r.get("method") or "")
        if sid and method:
            merged[(sid, method)] = dict(r)

    for r in (base_pairs or []):
        sid = str(r.get("sample_id") or "")
        method = str(r.get("method") or "")
        if sid and method:
            merged[(sid, method)] = dict(r)

    pairs_all = [merged[k] for k in sorted(merged.keys(), key=lambda t: (t[1], t[0]))]
    pairs_all_path = out_dir / "pairs_all.jsonl"
    _atomic_write_jsonl(pairs_all_path, pairs_all)

    # 4) Optional: compute RadEval extra metrics
    extra_paths: List[Path] = [Path(p).resolve() for p in (args.extra_metrics_jsonl or []) if str(p).strip()]
    for p in extra_paths:
        if not p.exists():
            raise SystemExit(f"--extra-metrics-jsonl not found: {p}")

    radeval_out = out_dir / "extra_radeval.jsonl"
    if bool(args.run_radeval):
        env_path = Path(str(args.radeval_env)).resolve()
        if not env_path.exists():
            raise SystemExit(f"--radeval-env not found: {env_path}")
        cmd = [
            "conda",
            "run",
            "-p",
            str(env_path),
            "python",
            str(ROOT / "scripts" / "external" / "compute_radeval_metrics_jsonl.py"),
            "--text-pairs-jsonl",
            str(pairs_all_path),
            "--out-jsonl",
            str(radeval_out),
        ]
        if bool(args.radeval_skip_green):
            cmd.append("--skip-green")
        if bool(args.radeval_skip_radcliq):
            cmd.append("--skip-radcliq")
        if bool(args.radeval_skip_chexbert):
            cmd.append("--skip-chexbert")
        _run(cmd)
        extra_paths.append(radeval_out)

    # 5) Optional: compute proof extra metrics for external methods
    proof_extra_out = out_dir / "extra_proof_external.jsonl"
    if bool(args.run_proof_external):
        if merged_external_pred_jsonl is None or not merged_external_pred_jsonl.exists():
            raise SystemExit("--run-proof-external requires --external-pred-jsonl")

        proof_out_dir = out_dir / "eval_external_predictions"
        cmd = [
            sys.executable,
            "-m",
            "provetok.experiments.eval_external_predictions",
            "--manifest",
            str(manifest_path),
            "--split",
            str(args.split),
            "--resize-shape",
            str(int(args.resize_shape[0])),
            str(int(args.resize_shape[1])),
            str(int(args.resize_shape[2])),
            "--max-samples",
            str(int(args.max_samples)),
            "--pred-jsonl",
            str(merged_external_pred_jsonl),
            "--tokenizer",
            str(args.tokenizer),
            "--max-depth",
            str(int(args.max_depth)),
            "--budget-tokens",
            str(int(args.budget_tokens)),
            "--emb-dim",
            str(int(args.emb_dim)),
            "--l-min",
            str(int(args.l_min)),
            "--k-max",
            str(int(args.k_max)),
            "--no-text-metrics",
            "--output-dir",
            str(proof_out_dir),
        ]
        _run(cmd)

        eval_json = _find_eval_external_predictions_json(proof_out_dir)
        _export_proof_extra(eval_json, out_jsonl=proof_extra_out)
        extra_paths.append(proof_extra_out)

    # 6) Merge extra metrics (if any)
    merged_extra_path = out_dir / "extra_merged.jsonl"
    merged_extra = _merge_extra_metrics_jsonl(extra_paths, out_path=merged_extra_path) if extra_paths else None

    # 7) Run paper metrics
    paper_out = out_dir / "paper_metrics.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "paper" / "compute_paper_metrics.py"),
        "--text-pairs-jsonl",
        str(pairs_all_path),
        "--out",
        str(paper_out),
        "--holm-family",
        str(args.holm_family),
        "--n-bootstrap",
        str(int(args.n_bootstrap)),
    ]
    if str(args.baseline_method).strip():
        cmd += ["--baseline-method", str(args.baseline_method).strip()]
    if merged_extra is not None and merged_extra.exists():
        cmd += ["--extra-metrics-jsonl", str(merged_extra)]

    # Optional heavy metrics
    if str(args.radgraph_model_type).strip():
        cmd += ["--radgraph-model-type", str(args.radgraph_model_type).strip()]
        if args.radgraph_cuda is not None:
            cmd += ["--radgraph-cuda", str(int(args.radgraph_cuda))]
    if bool(args.ratescore):
        cmd.append("--ratescore")
    if bool(args.ratescore_use_gpu):
        cmd.append("--ratescore-use-gpu")

    _run(cmd)

    print(json.dumps({"out_dir": str(out_dir), "pairs_all": str(pairs_all_path), "paper_metrics": str(paper_out)}, indent=2))


if __name__ == "__main__":
    main()
