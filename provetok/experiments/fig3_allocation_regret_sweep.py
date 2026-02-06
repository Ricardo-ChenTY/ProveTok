"""Fig 3 (oral-strength): Allocation regret over an expanded candidate set (multi-curve sweep).

Motivation:
`fig3_allocation_regret_real.py` treats each *method* as one candidate. For an
oral-ready story, we often need a *non-trivial* allocation space where curves
cross (e.g., varying b_gen/n_verify/topk across runs). This script supports that
by consuming multiple dev/test baselines curves and treating (tag, method) as a
distinct candidate.

Inputs:
- Multiple dev curves (split=val) produced by baselines_curve_multiseed.json under
  different global configs (e.g. b_gen/n_verify/topk).
- Multiple test curves (split=test) matching the same tags.

Output:
- <output_dir>/fig3_regret_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..eval.scaling import fit_scaling_law
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .utils import save_results_json


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_tagged_paths(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in str(it):
            raise ValueError(f"Expected TAG=PATH, got {it!r}")
        tag, path = str(it).split("=", 1)
        tag = tag.strip()
        path = path.strip()
        if not tag:
            raise ValueError(f"Empty tag in {it!r}")
        if not path:
            raise ValueError(f"Empty path in {it!r}")
        if tag in out:
            raise ValueError(f"Duplicate tag={tag!r}")
        out[tag] = path
    return out


@dataclass(frozen=True)
class CurveData:
    tag: str
    path: str
    split: str
    budgets: List[float]
    methods: List[str]
    metric: str
    means: Dict[str, List[float]]  # method -> mean per budget
    raw: Dict[str, Any]


def _load_curve(path: str, *, tag: str, metric: str) -> CurveData:
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

    return CurveData(tag=str(tag), path=str(p), split=split, budgets=budgets, methods=methods, metric=str(metric), means=means, raw=d)


def _shared_budgets(curves: Sequence[CurveData]) -> List[float]:
    shared: Optional[set[float]] = None
    for c in curves:
        s = set(float(b) for b in c.budgets)
        shared = s if shared is None else shared.intersection(s)
    return sorted(shared or set())


def _budget_indices(curve: CurveData, *, budgets: Sequence[float]) -> List[int]:
    idx = []
    for b in budgets:
        if float(b) not in curve.budgets:
            raise ValueError(f"Curve {curve.path!r} missing budget={b}")
        idx.append(curve.budgets.index(float(b)))
    return idx


def _per_sample_means_for_curve(
    curve: CurveData, *, budget: float, method: str, metric: str
) -> List[float]:
    raw = curve.raw or {}
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
        raise FileNotFoundError(
            f"Missing baselines.json raw arrays for tag={curve.tag} budget={b_key} method={method} metric={metric}"
        )
    n = min(len(x) for x in mats)
    if n <= 0:
        return []
    arr = np.asarray([x[:n] for x in mats], dtype=np.float64)  # (S,N)
    return arr.mean(axis=0).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig3 allocation/regret on an expanded candidate space (multi-curve sweep).")
    ap.add_argument(
        "--dev-curves",
        type=str,
        nargs="+",
        required=True,
        help="One or more TAG=path entries for dev (val) baselines_curve_multiseed.json.",
    )
    ap.add_argument(
        "--test-curves",
        type=str,
        nargs="+",
        required=True,
        help="One or more TAG=path entries for test baselines_curve_multiseed.json (must match tags).",
    )
    ap.add_argument("--metric", type=str, default="combined", help="Metric key in baselines curve (e.g., combined/iou).")
    ap.add_argument("--criterion", type=str, default="bic", choices=["aic", "bic"], help="Model selection criterion on dev.")
    ap.add_argument("--n-bootstrap", type=int, default=20_000, help="Bootstrap replicates for regret CI (paper-grade).")
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default="./outputs/fig3_regret_sweep")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dev_paths = _parse_tagged_paths(args.dev_curves)
    test_paths = _parse_tagged_paths(args.test_curves)

    tags = sorted(set(dev_paths.keys()).intersection(test_paths.keys()))
    if not tags:
        raise ValueError("No shared tags between --dev-curves and --test-curves.")
    missing_dev = sorted(set(test_paths.keys()) - set(dev_paths.keys()))
    missing_test = sorted(set(dev_paths.keys()) - set(test_paths.keys()))
    if missing_dev or missing_test:
        raise ValueError(f"Tag mismatch: missing_dev={missing_dev}, missing_test={missing_test}")

    dev_curves = [_load_curve(dev_paths[t], tag=t, metric=str(args.metric)) for t in tags]
    test_curves = [_load_curve(test_paths[t], tag=t, metric=str(args.metric)) for t in tags]

    budgets = _shared_budgets(dev_curves + test_curves)
    if not budgets:
        raise ValueError("No shared budgets across all curves. Re-run curves with the same --budgets list.")

    dev_idx = {c.tag: _budget_indices(c, budgets=budgets) for c in dev_curves}
    test_idx = {c.tag: _budget_indices(c, budgets=budgets) for c in test_curves}

    # Candidate set: (tag, method) that exists in both dev+test for that tag.
    candidates: List[Tuple[str, str]] = []
    for t in tags:
        dev_methods = set(next(c for c in dev_curves if c.tag == t).methods)
        test_methods = set(next(c for c in test_curves if c.tag == t).methods)
        shared = sorted(dev_methods.intersection(test_methods))
        for m in shared:
            candidates.append((t, m))
    if not candidates:
        raise ValueError("No shared methods across dev/test curves for any tag.")

    def cid(tag: str, method: str) -> str:
        return f"{tag}::{method}"

    # Fit per-candidate scaling laws on dev.
    dev_fits: Dict[str, Any] = {}
    dev_pred: Dict[str, List[float]] = {}
    for tag, method in candidates:
        dev_curve = next(c for c in dev_curves if c.tag == tag)
        y = [float(dev_curve.means[method][i]) for i in dev_idx[tag]]
        fit, reason = fit_scaling_law(budgets, y, criterion=str(args.criterion))
        key = cid(tag, method)
        dev_fits[key] = {
            "tag": str(tag),
            "method": str(method),
            "model_type": fit.model_type,
            "params": fit.params,
            "r_squared": float(fit.r_squared),
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "residual_std": float(fit.residual_std),
            "reason": str(reason),
        }
        dev_pred[key] = [float(fit.predict(float(b))) for b in budgets]

    # Point-estimate oracle + predicted-best + regret on test.
    rows: List[Dict[str, Any]] = []
    predicted_by_budget: Dict[float, str] = {}
    oracle_by_budget: Dict[float, str] = {}
    regrets: List[float] = []
    norm_regrets: List[float] = []
    for bi, b in enumerate(budgets):
        test_vals: Dict[str, float] = {}
        for tag, method in candidates:
            test_curve = next(c for c in test_curves if c.tag == tag)
            test_vals[cid(tag, method)] = float(test_curve.means[method][test_idx[tag][bi]])
        oracle_id = max(test_vals.items(), key=lambda kv: kv[1])[0]

        pred_vals = {k: float(v[bi]) for k, v in dev_pred.items()}
        predicted_id = max(pred_vals.items(), key=lambda kv: kv[1])[0]

        oracle = float(test_vals[oracle_id])
        worst = float(min(test_vals.values())) if test_vals else oracle
        achieved = float(test_vals[predicted_id])
        regret = float(oracle - achieved)
        denom = float(max(oracle - worst, 1e-12))
        norm_regret = float(regret / denom)
        regrets.append(regret)
        norm_regrets.append(norm_regret)

        predicted_by_budget[float(b)] = str(predicted_id)
        oracle_by_budget[float(b)] = str(oracle_id)
        rows.append(
            {
                "budget": float(b),
                "oracle_candidate": str(oracle_id),
                "oracle_metric": oracle,
                "predicted_candidate": str(predicted_id),
                "predicted_metric_dev_fit": float(pred_vals[predicted_id]),
                "achieved_metric_test": achieved,
                "regret": regret,
                "normalized_regret": norm_regret,
            }
        )

    mean_regret = float(np.mean(np.asarray(regrets, dtype=np.float64))) if regrets else 0.0
    mean_norm_regret = float(np.mean(np.asarray(norm_regrets, dtype=np.float64))) if norm_regrets else 0.0

    # Naive policies (for oral: one simple baseline is enough).
    tag0 = tags[0]
    fixed_grid_id = cid(tag0, "fixed_grid") if (tag0, "fixed_grid") in candidates else cid(*candidates[0])
    best_dev_max_budget = max(dev_pred.keys(), key=lambda k: float(dev_pred[k][-1])) if dev_pred else fixed_grid_id
    naive_policies: Dict[str, Dict[float, str]] = {
        "always_fixed_grid": {float(b): str(fixed_grid_id) for b in budgets},
        "best_dev_at_max_budget": {float(b): str(best_dev_max_budget) for b in budgets},
    }

    def _policy_point_estimates(policy: Dict[float, str]) -> Dict[str, Any]:
        rs = []
        nrs = []
        for bi, b in enumerate(budgets):
            test_vals: Dict[str, float] = {}
            for tag, method in candidates:
                test_curve = next(c for c in test_curves if c.tag == tag)
                test_vals[cid(tag, method)] = float(test_curve.means[method][test_idx[tag][bi]])
            oracle = float(max(test_vals.values())) if test_vals else 0.0
            worst = float(min(test_vals.values())) if test_vals else oracle
            chosen = str(policy[float(b)])
            achieved = float(test_vals.get(chosen, 0.0))
            r = float(oracle - achieved)
            denom = float(max(oracle - worst, 1e-12))
            nr = float(r / denom)
            rs.append(r)
            nrs.append(nr)
        return {
            "mean_regret": float(np.mean(np.asarray(rs, dtype=np.float64))) if rs else 0.0,
            "mean_normalized_regret": float(np.mean(np.asarray(nrs, dtype=np.float64))) if nrs else 0.0,
        }

    naive_point: Dict[str, Any] = {name: _policy_point_estimates(pol) for name, pol in naive_policies.items()}

    # Bootstrap CI for mean normalized regret (paired across candidates by sample index).
    n_boot = int(args.n_bootstrap)
    ci = float(args.ci)
    alpha = 1.0 - ci
    rng = np.random.RandomState(int(args.seed))

    # Preload per-sample arrays: budget -> candidate -> [N]
    per_sample: Dict[float, Dict[str, List[float]]] = {}
    n_samples: Optional[int] = None
    for b in budgets:
        per_sample[float(b)] = {}
        for tag, method in candidates:
            test_curve = next(c for c in test_curves if c.tag == tag)
            xs = _per_sample_means_for_curve(test_curve, budget=float(b), method=str(method), metric=str(args.metric))
            per_sample[float(b)][cid(tag, method)] = xs
            n_samples = len(xs) if n_samples is None else min(int(n_samples), len(xs))
    n_samples = int(n_samples or 0)

    def _boot_mean_norm_regret(policy: Dict[float, str]) -> float:
        if n_samples <= 0:
            return 0.0
        idx = rng.randint(0, n_samples, size=n_samples)
        vals = []
        for b in budgets:
            b = float(b)
            means_b: Dict[str, float] = {}
            for cand_id, xs in per_sample[b].items():
                arr = np.asarray(xs[:n_samples], dtype=np.float64)
                means_b[cand_id] = float(arr[idx].mean()) if arr.size else 0.0
            oracle = float(max(means_b.values())) if means_b else 0.0
            worst = float(min(means_b.values())) if means_b else oracle
            chosen = str(policy[b])
            achieved = float(means_b.get(chosen, 0.0))
            regret = float(oracle - achieved)
            denom = float(max(oracle - worst, 1e-12))
            vals.append(float(regret / denom))
        return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else 0.0

    # Policies evaluated: learned (predicted_by_budget) + naive ones.
    learned_policy = {float(b): str(predicted_by_budget[float(b)]) for b in budgets}
    policies = {"learned": learned_policy, **naive_policies}

    boot_samples: Dict[str, List[float]] = {k: [] for k in policies.keys()}
    for _ in range(n_boot):
        for name, pol in policies.items():
            boot_samples[name].append(_boot_mean_norm_regret(pol))

    def _ci_from_boot(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "ci_low": float(np.quantile(arr, alpha / 2.0)),
            "ci_high": float(np.quantile(arr, 1.0 - alpha / 2.0)),
        }

    boot_summary = {name: _ci_from_boot(xs) for name, xs in boot_samples.items()}
    learned_ci = boot_summary.get("learned") or {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    boot_naive_out: Dict[str, Any] = {}
    for name, rec in boot_summary.items():
        if name == "learned":
            continue
        boot_naive_out[str(name)] = {
            "mean_normalized_regret_ci_low": float(rec.get("ci_low", 0.0)),
            "mean_normalized_regret_ci_high": float(rec.get("ci_high", 0.0)),
        }

    oracle_candidates = sorted({str(r.get("oracle_candidate", "")) for r in rows if r.get("oracle_candidate")})
    is_nontrivial_case = len(oracle_candidates) > 1

    repo_root = Path(__file__).resolve().parents[2]
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(args.seed),
        config={
            "dev_curves": dict(dev_paths),
            "test_curves": dict(test_paths),
            "tags": tags,
            "metric": str(args.metric),
            "criterion": str(args.criterion),
            "budgets": budgets,
            "num_candidates": int(len(candidates)),
            "n_bootstrap": int(n_boot),
            "ci": float(ci),
        },
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision="multi_curve",
        split_manifest_path="",
    )

    report: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "config": meta.to_dict().get("config", {}),
        "budgets": budgets,
        "candidates": [cid(t, m) for t, m in candidates],
        "dev": {
            "fits": dev_fits,
        },
        "test": {
            "rows": rows,
            "predicted_by_budget": predicted_by_budget,
            "oracle_by_budget": oracle_by_budget,
        },
        "regret": {
            "mean_regret": mean_regret,
            "mean_normalized_regret": mean_norm_regret,
            "per_budget": regrets,
            "per_budget_normalized": norm_regrets,
            "is_nontrivial_case": bool(is_nontrivial_case),
            "bootstrap": {
                "n_bootstrap": int(n_boot),
                "ci": float(ci),
                "seed": int(args.seed),
                "mean_normalized_regret_ci_low": float(learned_ci.get("ci_low", 0.0)),
                "mean_normalized_regret_ci_high": float(learned_ci.get("ci_high", 0.0)),
                "naive_policies": boot_naive_out,
                "policies": boot_summary,
                "naive_point": naive_point,
            },
        },
        "naive_policies": {
            "definitions": {name: {str(k): str(v) for k, v in pol.items()} for name, pol in naive_policies.items()},
            "point_estimates": naive_point,
        },
    }

    out_path = os.path.join(args.output_dir, "fig3_regret_sweep.json")
    save_results_json(report, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
