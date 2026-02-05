"""Fig 3 (C0002): Scaling-law allocation model with dev→test regret on real pipeline.

This experiment treats each *method* (tokenization/protocol variant) as a candidate
configuration and fits a per-method scaling law on the dev split (AIC/BIC model
selection). It then predicts the best method per budget and reports regret on the
test split relative to the oracle best method.

Inputs:
- dev baselines curve artifact: baselines_curve_multiseed.json (split=val)
- test baselines curve artifact: baselines_curve_multiseed.json (split=test)

Output:
- <output_dir>/fig3_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..eval.scaling import fit_scaling_law
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .utils import save_results_json


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CurveData:
    path: str
    split: str
    budgets: List[float]
    methods: List[str]
    metric: str
    means: Dict[str, List[float]]
    raw: Dict[str, Any]


def _load_curve(path: str, *, metric: str) -> CurveData:
    p = Path(path)
    d = _read_json(p)
    budgets = [float(x) for x in d.get("budgets", [])]
    methods = [str(m) for m in (d.get("methods") or [])]
    meta = d.get("meta") or {}
    split = str((meta.get("config") or {}).get("split", ""))

    metrics = d.get("metrics") or {}
    metric_dict = metrics.get(metric) or {}
    if not isinstance(metric_dict, dict):
        raise ValueError(f"Invalid curve metrics[{metric!r}] at {path!r}")

    means: Dict[str, List[float]] = {}
    for m in methods:
        rows = metric_dict.get(m) or []
        means[m] = [float(r.get("mean", 0.0)) for r in rows]

    return CurveData(path=str(p), split=split, budgets=budgets, methods=methods, metric=str(metric), means=means, raw=d)


def _align_dev_test(dev: CurveData, test: CurveData) -> Tuple[List[float], List[str], List[int], List[int]]:
    """Return (budgets, methods, dev_idx, test_idx) aligned on shared budgets/methods."""
    shared_methods = sorted(set(dev.methods).intersection(test.methods))
    if not shared_methods:
        raise ValueError("No shared methods between dev and test curves.")

    shared_budgets = sorted(set(float(b) for b in dev.budgets).intersection(float(b) for b in test.budgets))
    if not shared_budgets:
        raise ValueError("No shared budgets between dev and test curves.")

    dev_idx = [dev.budgets.index(b) for b in shared_budgets]
    test_idx = [test.budgets.index(b) for b in shared_budgets]
    return shared_budgets, shared_methods, dev_idx, test_idx


def _per_sample_means_for_test_curve(
    test: CurveData, *, budget: float, method: str, metric: str
) -> List[float]:
    """Load per-sample metric arrays from baselines.json and average over seeds.

    The baselines curve stores pointers to per-budget/seed dirs; each contains
    baselines.json with raw per-sample arrays. For paper-grade regret CIs, we
    bootstrap across samples using these arrays.
    """
    raw = test.raw or {}
    per_budget_seed_dirs = raw.get("per_budget_seed_dirs") or {}
    seeds = [int(s) for s in (raw.get("seeds") or [])]
    b_key = f"{float(budget):g}"
    seed_dirs = per_budget_seed_dirs.get(b_key) or {}
    mats: List[List[float]] = []
    for s in seeds:
        dpath = Path(str(seed_dirs.get(str(s), ""))) / "baselines.json"
        if not dpath.exists():
            continue
        d = _read_json(dpath)
        xs = (((d.get("raw") or {}).get(method) or {}).get(metric)) or []
        mats.append([float(x) for x in xs])
    if not mats:
        raise FileNotFoundError(f"Missing baselines.json raw arrays for budget={b_key} method={method} metric={metric}")
    n = min(len(x) for x in mats)
    if n <= 0:
        return []
    arr = np.asarray([x[:n] for x in mats], dtype=np.float64)  # (S,N)
    return arr.mean(axis=0).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig3 allocation/regret on real pipeline (dev→test).")
    ap.add_argument("--dev-curve", type=str, required=True, help="Path to dev baselines_curve_multiseed.json (split=val).")
    ap.add_argument("--test-curve", type=str, required=True, help="Path to test baselines_curve_multiseed.json (split=test).")
    ap.add_argument("--metric", type=str, default="combined", help="Metric key in baselines curve (e.g., combined/iou).")
    ap.add_argument("--criterion", type=str, default="bic", choices=["aic", "bic"], help="Model selection criterion on dev.")
    ap.add_argument("--n-bootstrap", type=int, default=20_000, help="Bootstrap replicates for regret CI (paper-grade).")
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default="./outputs/fig3_allocation_regret_real")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dev = _load_curve(str(args.dev_curve), metric=str(args.metric))
    test = _load_curve(str(args.test_curve), metric=str(args.metric))
    budgets, methods, dev_idx, test_idx = _align_dev_test(dev, test)

    # Fit per-method scaling laws on dev.
    dev_fits: Dict[str, Any] = {}
    dev_pred: Dict[str, List[float]] = {}
    for m in methods:
        y = [float(dev.means[m][i]) for i in dev_idx]
        fit, reason = fit_scaling_law(budgets, y, criterion=str(args.criterion))
        dev_fits[m] = {
            "model_type": fit.model_type,
            "params": fit.params,
            "r_squared": float(fit.r_squared),
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "residual_std": float(fit.residual_std),
            "reason": str(reason),
        }
        dev_pred[m] = [float(fit.predict(float(b))) for b in budgets]

    # Oracle + predicted-best + regret on test.
    rows: List[Dict[str, Any]] = []
    regrets: List[float] = []
    norm_regrets: List[float] = []
    predicted_by_budget: Dict[float, str] = {}
    oracle_by_budget: Dict[float, str] = {}
    for bi, b in enumerate(budgets):
        test_vals = {m: float(test.means[m][test_idx[bi]]) for m in methods}
        oracle_method = max(test_vals.items(), key=lambda kv: kv[1])[0]

        pred_vals = {m: float(dev_pred[m][bi]) for m in methods}
        predicted_method = max(pred_vals.items(), key=lambda kv: kv[1])[0]

        oracle = float(test_vals[oracle_method])
        worst = float(min(test_vals.values())) if test_vals else oracle
        achieved = float(test_vals[predicted_method])
        regret = float(oracle - achieved)
        denom = float(max(oracle - worst, 1e-12))
        norm_regret = float(regret / denom)
        regrets.append(regret)
        norm_regrets.append(norm_regret)
        predicted_by_budget[float(b)] = str(predicted_method)
        oracle_by_budget[float(b)] = str(oracle_method)

        rows.append(
            {
                "budget": float(b),
                "oracle_method": str(oracle_method),
                "oracle_metric": oracle,
                "predicted_method": str(predicted_method),
                "predicted_metric_dev_fit": float(pred_vals[predicted_method]),
                "achieved_metric_test": achieved,
                "regret": regret,
                "normalized_regret": norm_regret,
            }
        )

    mean_regret = float(np.mean(np.asarray(regrets, dtype=np.float64))) if regrets else 0.0
    mean_norm_regret = float(np.mean(np.asarray(norm_regrets, dtype=np.float64))) if norm_regrets else 0.0

    # Naive policies (paper-grade sanity baselines for allocation).
    fixed_grid = "fixed_grid" if "fixed_grid" in methods else methods[0]
    best_dev_max_budget = None
    if dev_idx:
        max_i = dev_idx[-1]
        best_dev_max_budget = max(methods, key=lambda m: float(dev.means[m][max_i]))
    else:
        best_dev_max_budget = methods[0]
    naive_policies: Dict[str, Dict[float, str]] = {
        "always_fixed_grid": {float(b): str(fixed_grid) for b in budgets},
        "best_dev_at_max_budget": {float(b): str(best_dev_max_budget) for b in budgets},
    }

    def _policy_point_estimates(policy: Dict[float, str]) -> Dict[str, Any]:
        rs = []
        nrs = []
        for bi, b in enumerate(budgets):
            test_vals = {m: float(test.means[m][test_idx[bi]]) for m in methods}
            oracle = float(max(test_vals.values())) if test_vals else 0.0
            worst = float(min(test_vals.values())) if test_vals else oracle
            ach = float(test_vals.get(str(policy[float(b)]), 0.0))
            r = float(oracle - ach)
            denom = float(max(oracle - worst, 1e-12))
            nr = float(r / denom)
            rs.append(r)
            nrs.append(nr)
        return {
            "mean_regret": float(np.mean(np.asarray(rs, dtype=np.float64))) if rs else 0.0,
            "mean_normalized_regret": float(np.mean(np.asarray(nrs, dtype=np.float64))) if nrs else 0.0,
            "per_budget": [
                {"budget": float(b), "method": str(policy[float(b)]), "regret": float(rs[i]), "normalized_regret": float(nrs[i])}
                for i, b in enumerate(budgets)
            ],
        }

    naive_point: Dict[str, Any] = {name: _policy_point_estimates(pol) for name, pol in naive_policies.items()}

    # Bootstrap CI for mean normalized regret (across samples; dev fits/predicted methods fixed).
    n_boot = int(args.n_bootstrap)
    ci = float(args.ci)
    alpha = 1.0 - ci
    rng = np.random.RandomState(int(args.seed))

    # Preload per-sample arrays for test curve: budget -> method -> [N]
    per_sample: Dict[float, Dict[str, List[float]]] = {}
    n_samples = None
    for b in budgets:
        per_sample[float(b)] = {}
        for m in methods:
            xs = _per_sample_means_for_test_curve(test, budget=float(b), method=str(m), metric=str(args.metric))
            per_sample[float(b)][str(m)] = xs
            if n_samples is None:
                n_samples = len(xs)
            else:
                n_samples = min(int(n_samples), len(xs))
    n_samples = int(n_samples or 0)

    def _boot_mean_norm_regret(policy: Dict[float, str]) -> float:
        if n_samples <= 0:
            return 0.0
        idx = rng.randint(0, n_samples, size=n_samples)
        vals = []
        for b in budgets:
            b = float(b)
            means_b = {m: float(np.asarray(per_sample[b][m][:n_samples], dtype=np.float64)[idx].mean()) for m in methods}
            oracle = max(means_b.values()) if means_b else 0.0
            worst = min(means_b.values()) if means_b else oracle
            chosen = str(policy[b])
            achieved = float(means_b.get(chosen, 0.0))
            regret = float(oracle - achieved)
            denom = float(max(oracle - worst, 1e-12))
            vals.append(float(regret / denom))
        return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else 0.0

    # Boot distributions
    boot_main = []
    boot_naive: Dict[str, List[float]] = {k: [] for k in naive_policies.keys()}
    main_policy = {float(b): str(predicted_by_budget[float(b)]) for b in budgets}
    for _ in range(int(n_boot)):
        boot_main.append(_boot_mean_norm_regret(main_policy))
        for name, pol in naive_policies.items():
            boot_naive[name].append(_boot_mean_norm_regret(pol))

    boot_main_arr = np.asarray(boot_main, dtype=np.float64) if boot_main else np.asarray([0.0], dtype=np.float64)
    boot_main_ci_low = float(np.quantile(boot_main_arr, alpha / 2.0))
    boot_main_ci_high = float(np.quantile(boot_main_arr, 1.0 - alpha / 2.0))

    boot_naive_out: Dict[str, Any] = {}
    for name, xs in boot_naive.items():
        arr = np.asarray(xs, dtype=np.float64) if xs else np.asarray([0.0], dtype=np.float64)
        boot_naive_out[name] = {
            "mean_normalized_regret_ci_low": float(np.quantile(arr, alpha / 2.0)),
            "mean_normalized_regret_ci_high": float(np.quantile(arr, 1.0 - alpha / 2.0)),
        }

    # Meta: reuse version locks from curve artifacts when present.
    repo_root = Path(__file__).resolve().parents[2]
    meta_src = dev.raw.get("meta") or {}
    data_revision = str(meta_src.get("data_revision", ""))
    split_manifest_path = str(meta_src.get("split_manifest_path", ""))
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=0,
        config={
            "dataset_type": str((meta_src.get("config") or {}).get("dataset_type", "")),
            "dev_curve": str(dev.path),
            "dev_split": str(dev.split),
            "test_curve": str(test.path),
            "test_split": str(test.split),
            "budgets": budgets,
            "methods": methods,
            "metric": str(args.metric),
            "criterion": str(args.criterion),
            "n_bootstrap": int(args.n_bootstrap),
            "ci": float(args.ci),
            "seed": int(args.seed),
        },
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    out: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "config": {
            "dev_curve": str(dev.path),
            "test_curve": str(test.path),
            "metric": str(args.metric),
            "criterion": str(args.criterion),
            "budgets": budgets,
            "methods": methods,
            "dev_split": str(dev.split),
            "test_split": str(test.split),
        },
        "dev": {"curve_path": str(dev.path), "split": str(dev.split), "fits": dev_fits},
        "test": {"curve_path": str(test.path), "split": str(test.split)},
        "rows": rows,
        "regret": {
            "mean_regret": mean_regret,
            "mean_normalized_regret": mean_norm_regret,
            "per_budget": regrets,
            "per_budget_normalized": norm_regrets,
            "bootstrap": {
                "n_bootstrap": int(args.n_bootstrap),
                "ci": float(args.ci),
                "seed": int(args.seed),
                "mean_normalized_regret_ci_low": boot_main_ci_low,
                "mean_normalized_regret_ci_high": boot_main_ci_high,
                "naive_policies": boot_naive_out,
            },
        },
        "naive_policies": {
            "definitions": {name: {str(k): str(v) for k, v in pol.items()} for name, pol in naive_policies.items()},
            "point_estimates": naive_point,
        },
    }

    out_path = Path(args.output_dir) / "fig3_results.json"
    save_results_json(out, str(out_path))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
