"""Baselines multi-budget + multi-seed runner (paper-grade scaffold).

Runs `run_baselines` for multiple FLOPs budgets and multiple seeds, then aggregates
per-method/per-budget metrics with hierarchical bootstrap CIs:
1) average over seeds per sample
2) bootstrap across samples

Outputs:
- <output_dir>/budget_<B>/seed_<seed>/baselines.json (per seed)
- <output_dir>/baselines_curve_multiseed.json (aggregated curve + CI)
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ..eval.stats import bootstrap_mean_ci, bootstrap_quantile_ci
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .run_baselines import BaselineRunConfig, run_baselines
from .utils import save_results_json


def _mean_and_ci_from_seed_sample_matrix(
    x: np.ndarray, *, n_boot: int, seed: int, ci: float
) -> Dict[str, float]:
    """Hierarchical mean CI: mean over seeds per sample, bootstrap across samples."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (S,N), got shape={x.shape}")
    if x.shape[1] == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    # Some metrics are conditional (e.g., critical-present proxies). Represent
    # undefined samples as NaN and drop them consistently across seeds.
    if np.isnan(x).any():
        keep = ~np.isnan(x).any(axis=0)
        x = x[:, keep]
        if x.shape[1] == 0:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    per_sample = x.mean(axis=0)
    res = bootstrap_mean_ci(per_sample.tolist(), n_boot=n_boot, seed=seed, ci=ci)
    return {"mean": float(res.mean), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}


def _p95_and_ci_from_seed_sample_matrix(
    x: np.ndarray, *, n_boot: int, seed: int, ci: float
) -> Dict[str, float]:
    """Hierarchical P95 CI: mean over seeds per sample, bootstrap across samples."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (S,N), got shape={x.shape}")
    if x.shape[1] == 0:
        return {"p95_s": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if np.isnan(x).any():
        keep = ~np.isnan(x).any(axis=0)
        x = x[:, keep]
        if x.shape[1] == 0:
            return {"p95_s": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    per_sample = x.mean(axis=0)
    res = bootstrap_quantile_ci(per_sample.tolist(), q=0.95, n_boot=n_boot, seed=seed, ci=ci)
    return {"p95_s": float(res.value), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}


def _run_one(cfg_dict: Dict[str, Any], out_path: str) -> None:
    # Reduce OpenMP oversubscription when running many processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    rep = run_baselines(BaselineRunConfig(**cfg_dict))
    save_results_json(rep, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baselines across multiple FLOPs budgets + seeds with bootstrap CI.")
    ap.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest"])
    ap.add_argument("--manifest", type=str, default="", help="Manifest path when dataset-type=manifest")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64], help="Resize (D,H,W) for manifest volumes")
    ap.add_argument("--pcg", type=str, default="toy", choices=["toy", "llama2"], help="PCG backend")
    ap.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])
    ap.add_argument("--budgets", type=float, nargs="+", required=True, help="FLOPs total budgets (budget caps).")
    ap.add_argument("--b-gen", type=int, default=128, help="Decoder token budget for matched accounting (toy).")
    ap.add_argument("--n-verify", type=int, default=1, help="Verifier call count for matched accounting (toy).")
    ap.add_argument("--costs-json", type=str, default="", help="Optional JSON with ComputeUnitCosts.")
    ap.add_argument("--selector-ratio", type=float, default=0.1, help="ROI selector cost as a fraction of enc-token FLOPs.")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--topk-citations", type=int, default=3, help="Number of citations per finding frame.")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--nlg-weight", type=float, default=0.5)
    ap.add_argument("--grounding-weight", type=float, default=0.5)
    ap.add_argument("--lesionness-weights", type=str, default="", help="Optional lesionness_head.pt for provetok_lesionness.")
    ap.add_argument("--lesionness-device", type=str, default="cpu")
    ap.add_argument(
        "--lesionness-score-level-power",
        type=float,
        default=0.0,
        help="Multiply lesionness scores by (level+1)^p to favor finer cells when selecting citations.",
    )
    ap.add_argument("--ct2rep-strong-weights", type=str, default="", help="Optional ct2rep_strong.pt (paper-grade baseline).")
    ap.add_argument("--ct2rep-strong-device", type=str, default="cpu")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers for per-budget/per-seed runs.")
    ap.add_argument("--resume", action="store_true", help="Reuse existing baselines.json if present (skip rerun).")
    ap.add_argument("--output-dir", type=str, default="./outputs/baselines_curve_multiseed")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    budgets = [float(b) for b in args.budgets]
    seeds = [int(s) for s in args.seeds]

    # Run per-budget per-seed (optionally parallel + resumable).
    per_budget: Dict[str, Dict[int, Dict[str, Any]]] = {}
    per_budget_dirs: Dict[str, Dict[int, str]] = {}
    pending: List[Dict[str, Any]] = []
    for b in budgets:
        b_key = f"{b:g}"
        b_dir = b_key.replace(".", "_")
        per_budget[b_key] = {}
        per_budget_dirs[b_key] = {}
        for s in seeds:
            seed_dir = out_root / f"budget_{b_dir}" / f"seed_{s}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            cfg = BaselineRunConfig(
                dataset_type=str(args.dataset_type),
                manifest_path=str(args.manifest),
                split=str(args.split),
                n_samples=int(args.n_samples),
                seed=int(s),
                output_dir=str(seed_dir),
                flops_total=float(b),
                b_gen=int(args.b_gen),
                n_verify=int(args.n_verify),
                costs_json=str(args.costs_json),
                selector_ratio=float(args.selector_ratio),
                resize_shape=tuple(int(x) for x in args.resize_shape),
                pcg_backend=str(args.pcg),
                llama2_path=str(args.llama2_path),
                llama2_quant=str(args.llama2_quant),
                nlg_weight=float(args.nlg_weight),
                grounding_weight=float(args.grounding_weight),
                lesionness_weights=str(args.lesionness_weights),
                lesionness_device=str(args.lesionness_device),
                lesionness_score_level_power=float(args.lesionness_score_level_power),
                topk_citations=int(args.topk_citations),
                ct2rep_strong_weights=str(args.ct2rep_strong_weights),
                ct2rep_strong_device=str(args.ct2rep_strong_device),
            )
            out_path = seed_dir / "baselines.json"

            if args.resume and out_path.exists():
                try:
                    import json

                    rep = json.loads(out_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    rep = None
                if isinstance(rep, dict) and rep:
                    per_budget[b_key][int(s)] = rep
                    per_budget_dirs[b_key][int(s)] = str(seed_dir)
                    continue

            pending.append(
                {
                    "b_key": b_key,
                    "seed": int(s),
                    "seed_dir": str(seed_dir),
                    "cfg_dict": cfg.__dict__,
                    "out_path": str(out_path),
                }
            )

    if pending:
        workers = max(1, int(args.workers))
        if workers == 1:
            for job in pending:
                _run_one(job["cfg_dict"], job["out_path"])
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_one, job["cfg_dict"], job["out_path"]) for job in pending]
                for fut in as_completed(futs):
                    fut.result()

        # Load completed jobs
        import json

        for job in pending:
            out_path = Path(str(job["out_path"]))
            rep = json.loads(out_path.read_text(encoding="utf-8"))
            per_budget[str(job["b_key"])][int(job["seed"])] = rep
            per_budget_dirs[str(job["b_key"])][int(job["seed"])] = str(job["seed_dir"])

    # Aggregate curve with hierarchical bootstrap per budget/method/metric.
    any_budget = f"{budgets[0]:g}"
    any_seed = seeds[0]
    methods = sorted(per_budget[any_budget][any_seed].get("raw", {}).keys())
    metric_keys = sorted(per_budget[any_budget][any_seed]["raw"][methods[0]].keys())

    metrics_out: Dict[str, Dict[str, List[Dict[str, float]]]] = {k: {m: [] for m in methods} for k in metric_keys}
    latency_mean_out: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
    latency_p95_out: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}

    for bidx, b in enumerate(budgets):
        b_key = f"{b:g}"
        for m in methods:
            for k in metric_keys:
                mats = []
                for s in seeds:
                    mats.append(per_budget[b_key][int(s)]["raw"][m][k])
                arr = np.asarray(mats, dtype=np.float64)  # (S,N)
                stable = hashlib.sha1(f"{m}:{k}".encode("utf-8")).digest()
                stable_seed = int.from_bytes(stable[:4], "little", signed=False)
                ci_rec = _mean_and_ci_from_seed_sample_matrix(
                    arr,
                    n_boot=int(args.n_bootstrap),
                    seed=int(seeds[0]) + 1000 * bidx + (stable_seed % 997),
                    ci=float(args.ci),
                )
                metrics_out[k][m].append(ci_rec)

            # Paper-grade tail latency: compute warm P95 CI (and also keep mean latency in a stable location).
            # We use the per-sample mean over seeds, then bootstrap across samples.
            mats_lat = []
            for s in seeds:
                mats_lat.append(per_budget[b_key][int(s)]["raw"][m].get("warm_time_s", []))
            if mats_lat and mats_lat[0]:
                min_n = min(len(x) for x in mats_lat)
                arr_lat = np.asarray([x[:min_n] for x in mats_lat], dtype=np.float64)
                stable = hashlib.sha1(f"{m}:warm_time_s".encode("utf-8")).digest()
                stable_seed = int.from_bytes(stable[:4], "little", signed=False)
                mean_ci = _mean_and_ci_from_seed_sample_matrix(
                    arr_lat,
                    n_boot=int(args.n_bootstrap),
                    seed=int(seeds[0]) + 3000 * bidx + (stable_seed % 997),
                    ci=float(args.ci),
                )
                p95_ci = _p95_and_ci_from_seed_sample_matrix(
                    arr_lat,
                    n_boot=int(args.n_bootstrap),
                    seed=int(seeds[0]) + 4000 * bidx + (stable_seed % 997),
                    ci=float(args.ci),
                )
            else:
                mean_ci = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
                p95_ci = {"p95_s": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            latency_mean_out[m].append(mean_ci)
            latency_p95_out[m].append(p95_ci)

    # Artifact meta for aggregated report
    repo_root = Path(__file__).resolve().parents[2]
    data_revision = "synthetic"
    split_manifest_path = ""
    if args.dataset_type == "manifest":
        data_revision, split_manifest_path = try_manifest_revision(str(args.manifest))
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(seeds[0]),
        config={
            "dataset_type": args.dataset_type,
            "manifest_path": args.manifest,
            "split": args.split,
            "resize_shape": list(args.resize_shape),
            "pcg": args.pcg,
            "llama2_path": args.llama2_path,
            "llama2_quant": args.llama2_quant,
            "budgets": budgets,
            "b_gen": int(args.b_gen),
            "n_verify": int(args.n_verify),
            "costs_json": args.costs_json,
            "selector_ratio": float(args.selector_ratio),
            "n_samples": int(args.n_samples),
            "topk_citations": int(args.topk_citations),
            "seeds": seeds,
            "n_bootstrap": int(args.n_bootstrap),
            "ci": float(args.ci),
            "nlg_weight": float(args.nlg_weight),
            "grounding_weight": float(args.grounding_weight),
            "lesionness_weights": str(args.lesionness_weights),
            "lesionness_device": str(args.lesionness_device),
            "ct2rep_strong_weights": str(args.ct2rep_strong_weights),
            "ct2rep_strong_device": str(args.ct2rep_strong_device),
        },
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    # Keep per-budget baseline accounting snapshots (from the first seed; deterministic).
    budgets_by_method: Dict[str, Dict[str, Any]] = {}
    for b in budgets:
        b_key = f"{b:g}"
        budgets_by_method[b_key] = per_budget[b_key][any_seed].get("budgets", {})

    report: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "budget_mode": "flops",
        "budgets": budgets,
        "seeds": seeds,
        "n_samples": int(args.n_samples),
        "n_bootstrap": int(args.n_bootstrap),
        "ci": float(args.ci),
        "methods": methods,
        "metrics": metrics_out,
        "latency": {
            "warm_time_mean_s": latency_mean_out,
            "warm_time_p95_s": latency_p95_out,
        },
        "budget_targets": [{"flops_total": float(b), "b_gen": float(args.b_gen), "n_verify": float(args.n_verify)} for b in budgets],
        "budgets_by_method": budgets_by_method,
        "per_budget_seed_dirs": {bk: {str(s): d for s, d in dirs.items()} for bk, dirs in per_budget_dirs.items()},
    }

    out_path = out_root / "baselines_curve_multiseed.json"
    save_results_json(report, str(out_path))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
