#!/usr/bin/env python3
from __future__ import annotations

# Allow running as a script (`python scripts/...`) without requiring editable install.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_mean_ci(values: Sequence[float], *, n_boot: int, seed: int, ci: float) -> Dict[str, float]:
    from provetok.eval.stats import bootstrap_mean_ci

    res = bootstrap_mean_ci(values, n_boot=int(n_boot), seed=int(seed), ci=float(ci))
    return {"mean": float(res.mean), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}


def _bootstrap_p95_ci(values: Sequence[float], *, n_boot: int, seed: int, ci: float) -> Dict[str, float]:
    from provetok.eval.stats import bootstrap_quantile_ci

    res = bootstrap_quantile_ci(values, q=0.95, n_boot=int(n_boot), seed=int(seed), ci=float(ci))
    return {"p95_s": float(res.value), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}


def _stack_seed_sample(raws: Dict[str, Dict[str, Any]], *, sample_key: str, bidx: int) -> np.ndarray:
    mats: List[List[float]] = []
    for _, d in raws.items():
        samples = (d.get("samples") or {}).get(sample_key) or []
        if len(samples) <= int(bidx):
            mats.append([])
            continue
        mats.append(list(samples[int(bidx)]))
    min_n = min((len(x) for x in mats if x), default=0)
    if min_n <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    arr = np.asarray([x[:min_n] for x in mats], dtype=np.float64)  # (S,N)
    return arr


def _mean_over_seeds_per_sample(arr: np.ndarray) -> List[float]:
    if arr.ndim != 2 or arr.shape[1] == 0:
        return []
    # Drop samples that are NaN in any seed to keep pairing stable.
    if np.isnan(arr).any():
        keep = ~np.isnan(arr).any(axis=0)
        arr = arr[:, keep]
    if arr.shape[1] == 0:
        return []
    return arr.mean(axis=0).astype(np.float64).tolist()


def _find_fig2_raws(fig2_root: Path) -> Dict[str, Dict[str, Any]]:
    raws: Dict[str, Dict[str, Any]] = {}
    for p in sorted(fig2_root.glob("seed_*/fig2_raw_data.json")):
        seed_name = p.parent.name
        raws[seed_name] = _load_json(p)
    return raws


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Table Sx. Budget Accounting Snapshot (auditable)")
    lines.append("")
    lines.append("| Budget | evidence tokens | mean token level | p95 token level | verifier calls | refine steps | warm latency P95 (s) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        b = r["budget"]
        tok = r["tokens_used"]["mean"]
        lvl = r["token_level_mean"]["mean"]
        lvl95 = r["token_level_p95"]["mean"]
        vc = r["verifier_calls"]["mean"]
        st = r["steps_used"]["mean"]
        p95 = r["warm_time_p95_s"]["p95_s"]
        lines.append(f"| {b:g} | {tok:.2f} | {lvl:.3f} | {lvl95:.3f} | {vc:.2f} | {st:.2f} | {p95:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a budget-accounting table (Table Sx) from Fig2 multi-seed outputs.\n\n"
            "Inputs are `seed_*/fig2_raw_data.json` under --fig2-root."
        )
    )
    ap.add_argument("--fig2-root", type=str, required=True, help="Output dir of fig2_scaling_multiseed (contains seed_*/fig2_raw_data.json).")
    ap.add_argument("--out-dir", type=str, default="", help="Output directory (default: <fig2-root>/budget_accounting).")
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-docs-table", type=str, default="", help="Optional path to write a .md table for paper assets.")
    args = ap.parse_args()

    fig2_root = Path(args.fig2_root).resolve()
    raws = _find_fig2_raws(fig2_root)
    if not raws:
        raise SystemExit(f"No seed_*/fig2_raw_data.json found under: {fig2_root}")

    any_raw = next(iter(raws.values()))
    budgets = any_raw.get("budgets") or []
    if not budgets:
        raise SystemExit("Missing budgets in fig2_raw_data.json")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (fig2_root / "budget_accounting")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for bidx, b in enumerate(budgets):
        tokens_arr = _stack_seed_sample(raws, sample_key="tokens_used", bidx=bidx)
        steps_arr = _stack_seed_sample(raws, sample_key="steps_used", bidx=bidx)
        verifier_arr = _stack_seed_sample(raws, sample_key="verifier_calls", bidx=bidx)
        lvl_arr = _stack_seed_sample(raws, sample_key="token_level_mean", bidx=bidx)
        lvl95_arr = _stack_seed_sample(raws, sample_key="token_level_p95", bidx=bidx)
        warm_arr = _stack_seed_sample(raws, sample_key="warm_time_s", bidx=bidx)

        per_sample_tokens = _mean_over_seeds_per_sample(tokens_arr)
        per_sample_steps = _mean_over_seeds_per_sample(steps_arr)
        per_sample_verifier = _mean_over_seeds_per_sample(verifier_arr)
        per_sample_lvl = _mean_over_seeds_per_sample(lvl_arr)
        per_sample_lvl95 = _mean_over_seeds_per_sample(lvl95_arr)
        per_sample_warm = _mean_over_seeds_per_sample(warm_arr)

        rows.append(
            {
                "budget": float(b),
                "tokens_used": _bootstrap_mean_ci(per_sample_tokens, n_boot=args.n_bootstrap, seed=args.seed + 10 * bidx, ci=args.ci),
                "steps_used": _bootstrap_mean_ci(per_sample_steps, n_boot=args.n_bootstrap, seed=args.seed + 100 + 10 * bidx, ci=args.ci),
                "verifier_calls": _bootstrap_mean_ci(per_sample_verifier, n_boot=args.n_bootstrap, seed=args.seed + 200 + 10 * bidx, ci=args.ci),
                "token_level_mean": _bootstrap_mean_ci(per_sample_lvl, n_boot=args.n_bootstrap, seed=args.seed + 300 + 10 * bidx, ci=args.ci),
                "token_level_p95": _bootstrap_mean_ci(per_sample_lvl95, n_boot=args.n_bootstrap, seed=args.seed + 400 + 10 * bidx, ci=args.ci),
                "warm_time_p95_s": _bootstrap_p95_ci(per_sample_warm, n_boot=args.n_bootstrap, seed=args.seed + 500 + 10 * bidx, ci=args.ci),
            }
        )

    md = _markdown_table(rows)
    out_json = out_dir / "tableS_budget_accounting.json"
    out_md = out_dir / "tableS_budget_accounting.md"

    report = {
        "root": str(ROOT),
        "fig2_root": str(fig2_root),
        "seed_raw_paths": {k: str(fig2_root / k / "fig2_raw_data.json") for k in raws.keys()},
        "budgets": [float(x) for x in budgets],
        "n_bootstrap": int(args.n_bootstrap),
        "ci": float(args.ci),
        "rows": rows,
        "out_md": str(out_md),
    }
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(md + "\n", encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md)}, ensure_ascii=False))

    if args.write_docs_table:
        p = Path(args.write_docs_table).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md + "\n", encoding="utf-8")
        print(json.dumps({"write_docs_table": str(p)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
