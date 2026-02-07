#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = list(enumerate(float(p) for p in p_values))
    indexed.sort(key=lambda x: x[1])
    adjusted: Dict[int, float] = {}
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * float(p))
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return [adjusted[i] for i in range(m)]


def _paired_bootstrap_mean_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int,
    seed: int,
    ci: float,
    alternative: str,
) -> Dict[str, float]:
    """Paired bootstrap for mean difference (a - b)."""
    a_arr = np.asarray(list(a), dtype=np.float64)
    b_arr = np.asarray(list(b), dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Shape mismatch: a={a_arr.shape}, b={b_arr.shape}")
    n = int(a_arr.shape[0])
    if n <= 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": 0}
    diffs = a_arr - b_arr
    mean_diff = float(diffs.mean())
    rng = np.random.RandomState(int(seed))
    idx = rng.randint(0, n, size=(int(n_boot), n))
    boot = diffs[idx].mean(axis=1)
    alpha = 1.0 - float(ci)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    p_pos = float(np.mean(boot >= 0.0))
    p_neg = float(np.mean(boot <= 0.0))
    alt = str(alternative)
    if alt == "two-sided":
        p_val = min(1.0, 2.0 * min(p_pos, p_neg))
    elif alt == "greater":
        # H1: mean_diff > 0
        p_val = p_neg
    elif alt == "less":
        # H1: mean_diff < 0
        p_val = p_pos
    else:
        raise ValueError(f"unknown alternative={alternative!r}")
    return {
        "mean_diff": float(mean_diff),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_val),
        "n": int(n),
    }


@dataclass(frozen=True)
class MetricSpec:
    key: str
    higher_is_better: bool


METRICS: Dict[str, MetricSpec] = {
    "combined": MetricSpec(key="combined", higher_is_better=True),
    "iou": MetricSpec(key="iou", higher_is_better=True),
    "unsupported": MetricSpec(key="unsupported", higher_is_better=False),
}


def _budget_key(per_budget_seed_dirs: Dict[str, Any], budget: float) -> Optional[str]:
    for k in per_budget_seed_dirs.keys():
        try:
            if float(k) == float(budget):
                return str(k)
        except Exception:
            continue
    return None


def _load_per_budget_seed_dirs(curve: Dict[str, Any]) -> Dict[str, Any]:
    raw = curve
    per_budget_seed_dirs = raw.get("per_budget_seed_dirs") or {}
    if not isinstance(per_budget_seed_dirs, dict):
        return {}
    return per_budget_seed_dirs


def _load_curve(curve_path: Path) -> Dict[str, Any]:
    d = _read_json(curve_path)
    if "budgets" not in d or "metrics" not in d:
        raise ValueError(f"Not a baselines_curve_multiseed.json: {curve_path}")
    return d


def _extract_curve_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    budgets = [float(x) for x in (d.get("budgets") or [])]
    seeds = [int(x) for x in (d.get("seeds") or [])]
    n_boot = int(d.get("n_bootstrap", 10_000))
    ci = float(d.get("ci", 0.95))
    return {"budgets": budgets, "seeds": seeds, "n_bootstrap": n_boot, "ci": ci}


def compute_curve_tests(
    *,
    curve_path: Path,
    method: str,
    baseline: str,
    metric_names: Sequence[str],
    n_bootstrap_override: int,
    seed: int,
) -> Dict[str, Any]:
    d = _load_curve(curve_path)
    meta = _extract_curve_meta(d)
    budgets = meta["budgets"]
    seeds = meta["seeds"]
    n_boot = int(n_bootstrap_override) if int(n_bootstrap_override) > 0 else int(meta["n_bootstrap"])
    ci = float(meta["ci"])

    per_budget_seed_dirs = _load_per_budget_seed_dirs(d)
    if not per_budget_seed_dirs:
        raise ValueError(f"Missing per_budget_seed_dirs in {curve_path} (need paper-style baselines_curve runner).")

    rows_by_metric: Dict[str, List[Dict[str, Any]]] = {m: [] for m in metric_names}
    pvals_by_metric: Dict[str, List[float]] = {m: [] for m in metric_names}

    for bidx, budget in enumerate(budgets):
        bk = _budget_key(per_budget_seed_dirs, float(budget))
        if bk is None:
            continue
        seed_dirs = per_budget_seed_dirs.get(bk) or {}
        shared_seeds = [s for s in seeds if str(s) in seed_dirs]
        if not shared_seeds:
            continue

        # Load raw arrays per seed (S,N).
        raw_by_metric = {m: {"method": [], "baseline": []} for m in metric_names}
        used_seeds: List[int] = []
        for s in shared_seeds:
            dpath = Path(str(seed_dirs[str(s)])) / "baselines.json"
            if not dpath.exists():
                continue
            try:
                sd = _read_json(dpath)
            except Exception:
                continue
            used_seeds.append(int(s))
            for m in metric_names:
                spec = METRICS.get(m)
                if spec is None:
                    raise ValueError(f"Unknown metric name={m!r} (supported={sorted(METRICS.keys())})")
                raw_m = ((sd.get("raw") or {}).get(method) or {}).get(spec.key) or []
                raw_b = ((sd.get("raw") or {}).get(baseline) or {}).get(spec.key) or []
                raw_by_metric[m]["method"].append(raw_m)
                raw_by_metric[m]["baseline"].append(raw_b)

        if not used_seeds:
            continue

        for m in metric_names:
            mats_m = raw_by_metric[m]["method"]
            mats_b = raw_by_metric[m]["baseline"]
            if not mats_m or not mats_b:
                continue
            arr_m = np.asarray(mats_m, dtype=np.float64)
            arr_b = np.asarray(mats_b, dtype=np.float64)
            n = int(min(arr_m.shape[1], arr_b.shape[1]))
            if n <= 0:
                continue
            per_sample_m = arr_m[:, :n].mean(axis=0)
            per_sample_b = arr_b[:, :n].mean(axis=0)

            spec = METRICS[m]
            # For "lower is better" metrics, define improvement as (baseline - method).
            if spec.higher_is_better:
                a = per_sample_m.tolist()
                b = per_sample_b.tolist()
            else:
                a = per_sample_b.tolist()
                b = per_sample_m.tolist()

            res = _paired_bootstrap_mean_diff(
                a,
                b,
                n_boot=int(n_boot),
                seed=int(seed) + 1000 * bidx + 17 * (hash(m) % 997),
                ci=float(ci),
                alternative="greater",
            )
            pvals_by_metric[m].append(float(res["p_value"]))
            rows_by_metric[m].append(
                {
                    "budget": float(budget),
                    "n_samples": int(res["n"]),
                    "used_seeds": used_seeds,
                    "mean_diff": float(res["mean_diff"]),
                    "ci_low": float(res["ci_low"]),
                    "ci_high": float(res["ci_high"]),
                    "p_value": float(res["p_value"]),
                    "improvement_definition": (
                        f"{method}-{baseline}" if spec.higher_is_better else f"{baseline}-{method}"
                    ),
                }
            )

    # Holm per metric family across budgets.
    for m in metric_names:
        p_holm = _holm_bonferroni(pvals_by_metric[m])
        for row, ph in zip(rows_by_metric[m], p_holm):
            row["p_holm"] = float(ph)

    return {
        "curve_path": str(curve_path),
        "method": str(method),
        "baseline": str(baseline),
        "meta": {
            "budgets": budgets,
            "seeds": seeds,
            "n_bootstrap_used": int(n_boot),
            "ci": float(ci),
        },
        "rows": rows_by_metric,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Backbone transfer report (paired bootstrap + Holm over budgets).")
    ap.add_argument("--toy-curve", type=str, required=True, help="baselines_curve_multiseed.json for backbone A")
    ap.add_argument("--llama2-curve", type=str, required=True, help="baselines_curve_multiseed.json for backbone B")
    ap.add_argument("--method", type=str, default="provetok_lesionness")
    ap.add_argument("--baseline", type=str, default="fixed_grid")
    ap.add_argument("--metrics", type=str, nargs="+", default=["combined", "iou", "unsupported"])
    ap.add_argument("--n-bootstrap", type=int, default=0, help="Override bootstrap resamples (0=use artifact meta).")
    ap.add_argument("--seed", type=int, default=0, help="Seed for the report bootstrap.")
    ap.add_argument("--out-dir", type=str, default="outputs/V0004-backbone-transfer")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    toy = compute_curve_tests(
        curve_path=Path(args.toy_curve),
        method=str(args.method),
        baseline=str(args.baseline),
        metric_names=list(args.metrics),
        n_bootstrap_override=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    llama2 = compute_curve_tests(
        curve_path=Path(args.llama2_curve),
        method=str(args.method),
        baseline=str(args.baseline),
        metric_names=list(args.metrics),
        n_bootstrap_override=int(args.n_bootstrap),
        seed=int(args.seed) + 1,
    )

    report = {
        "toy": toy,
        "llama2": llama2,
        "notes": {
            "paired_bootstrap": "avg over seeds per sample, bootstrap over samples",
            "holm_family": "per-metric across budgets",
            "direction": "positive mean_diff indicates improvement",
        },
    }
    out_path = out_dir / "backbone_transfer_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Simple markdown summary for quick copy-paste.
    md_lines: List[str] = []
    md_lines.append("# Backbone Transfer Report (V0004)\n")
    md_lines.append(f"- method: `{args.method}`\n")
    md_lines.append(f"- baseline: `{args.baseline}`\n")
    md_lines.append(f"- metrics: `{', '.join(args.metrics)}`\n")
    md_lines.append(f"- toy curve: `{args.toy_curve}`\n")
    md_lines.append(f"- llama2 curve: `{args.llama2_curve}`\n")
    md_lines.append("\n")
    for name, block in (("toy", toy), ("llama2", llama2)):
        md_lines.append(f"## {name}\n\n")
        md_lines.append(f"- budgets: {block['meta']['budgets']}\n")
        md_lines.append(f"- seeds: {block['meta']['seeds']}\n")
        md_lines.append(f"- n_bootstrap: {block['meta']['n_bootstrap_used']}\n\n")
        for m in args.metrics:
            md_lines.append(f"### {m}\n\n")
            md_lines.append("| budget | mean_diff | 95% CI | p | p_holm | n |\n")
            md_lines.append("|---:|---:|---:|---:|---:|---:|\n")
            for row in block["rows"].get(m, []):
                md_lines.append(
                    f"| {row['budget']:.0f} | {row['mean_diff']:.6f} | [{row['ci_low']:.6f},{row['ci_high']:.6f}] | {row['p_value']:.4g} | {row.get('p_holm',1.0):.4g} | {row['n_samples']} |\n"
                )
            md_lines.append("\n")
    (out_dir / "backbone_transfer_report.md").write_text("".join(md_lines), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "out_json": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

