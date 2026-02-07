#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

def _holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = list(enumerate(float(p) for p in p_values))
    indexed.sort(key=lambda x: x[1])
    adjusted: Dict[int, float] = {}
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * p)
        running_max = max(running_max, float(adj))
        adjusted[idx] = running_max
    return [float(adjusted[i]) for i in range(m)]


@dataclass(frozen=True)
class BootResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_one_sided: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bootstrap_hierarchical_mean_diff(
    diffs_by_seed: Sequence[np.ndarray],
    *,
    n_boot: int,
    seed: int,
    ci: float,
) -> BootResult:
    if not diffs_by_seed:
        return BootResult(mean_diff=0.0, ci_low=0.0, ci_high=0.0, p_value=1.0, p_value_one_sided=1.0)

    # Equal seed weighting: first average within seed, then average across seeds.
    per_seed_means = np.asarray([float(np.mean(d)) if d.size else 0.0 for d in diffs_by_seed], dtype=np.float64)
    mean_diff = float(np.mean(per_seed_means))

    rng = np.random.RandomState(int(seed))
    s = len(diffs_by_seed)
    boot = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        sampled_seed_idx = rng.randint(0, s, size=s)
        seed_means = np.empty(s, dtype=np.float64)
        for j, sid in enumerate(sampled_seed_idx):
            arr = diffs_by_seed[int(sid)]
            n = int(arr.size)
            if n == 0:
                seed_means[j] = 0.0
                continue
            idx = rng.randint(0, n, size=n)
            seed_means[j] = float(np.mean(arr[idx]))
        boot[i] = float(np.mean(seed_means))

    alpha = 1.0 - float(ci)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))

    # Two-sided p-value (bootstrap sign test style), consistent with repo style.
    p_pos = float(np.mean(boot >= 0.0))
    p_neg = float(np.mean(boot <= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_pos, p_neg)))

    # One-sided for H1: mean_diff > 0
    p_one = float(np.mean(boot <= 0.0))

    return BootResult(
        mean_diff=mean_diff,
        ci_low=lo,
        ci_high=hi,
        p_value=p_two,
        p_value_one_sided=p_one,
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_inputs(inputs: Sequence[str], patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for p in inputs:
        path = Path(p).expanduser().resolve()
        if path.exists() and path.is_file():
            paths.append(path)
    for patt in patterns:
        for p in glob.glob(patt):
            path = Path(p).expanduser().resolve()
            if path.exists() and path.is_file():
                paths.append(path)
    # unique + stable
    uniq = sorted({str(p): p for p in paths}.values(), key=lambda x: str(x))
    return uniq


def _build_markdown(
    *,
    created_at: str,
    metric: str,
    primary_key: str,
    alpha: float,
    n_bootstrap: int,
    files: Sequence[Path],
    per_seed_rows: Sequence[dict],
    family_rows: Dict[str, dict],
    primary_row: dict,
) -> str:
    lines: List[str] = []
    lines.append("# Omega-Perm Power Report")
    lines.append("")
    lines.append(f"- created_at: `{created_at}`")
    lines.append(f"- metric: `{metric}`")
    lines.append(f"- primary_key: `{primary_key}`")
    lines.append(f"- alpha: `{alpha}`")
    lines.append(f"- n_bootstrap: `{n_bootstrap}`")
    lines.append(f"- files: `{len(files)}`")
    lines.append("")

    lines.append("## Primary (pooled)")
    lines.append("")
    lines.append("| key | mean_diff | ci_low | ci_high | p_one_sided | p_two_sided | p_holm_secondary | positive_seeds |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| "
        + " | ".join(
            [
                str(primary_key),
                f"{float(primary_row.get('mean_diff', 0.0)):.6f}",
                f"{float(primary_row.get('ci_low', 0.0)):.6f}",
                f"{float(primary_row.get('ci_high', 0.0)):.6f}",
                f"{float(primary_row.get('p_value_one_sided', 1.0)):.6g}",
                f"{float(primary_row.get('p_value', 1.0)):.6g}",
                f"{float(primary_row.get('p_value_holm_secondary', 1.0)):.6g}",
                f"{int(primary_row.get('positive_seed_count', 0))}/{int(primary_row.get('seed_count', 0))}",
            ]
        )
        + " |"
    )
    lines.append("")

    lines.append("## Per-Seed")
    lines.append("")
    lines.append("| seed | mean_diff | ci_low | ci_high | p_value | p_holm | n_samples | path |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted(per_seed_rows, key=lambda r: int(r.get("seed", 0))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row.get("seed", 0))),
                    f"{float(row.get('mean_diff', 0.0)):.6f}",
                    f"{float(row.get('ci_low', 0.0)):.6f}",
                    f"{float(row.get('ci_high', 0.0)):.6f}",
                    f"{float(row.get('p_value', 1.0)):.6g}",
                    f"{float(row.get('p_value_holm', 1.0)):.6g}",
                    str(int(row.get("n_samples", 0))),
                    str(row.get("path", "")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Secondary Family (pooled + Holm)")
    lines.append("")
    lines.append("| key | mean_diff | ci_low | ci_high | p_two_sided | p_holm |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in sorted(family_rows.keys()):
        row = family_rows[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    f"{float(row.get('mean_diff', 0.0)):.6f}",
                    f"{float(row.get('ci_low', 0.0)):.6f}",
                    f"{float(row.get('ci_high', 0.0)):.6f}",
                    f"{float(row.get('p_value', 1.0)):.6g}",
                    f"{float(row.get('p_value_holm_secondary', 1.0)):.6g}",
                ]
            )
            + " |"
        )

    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate counterfactual omega_perm power across seeds.")
    ap.add_argument("--inputs", nargs="*", default=[], help="Explicit figX_counterfactual.json files.")
    ap.add_argument("--glob", action="append", default=[], help="Glob pattern(s) for figX_counterfactual.json files.")
    ap.add_argument("--metric", type=str, default="grounding_iou_union_orig_minus_cf", help="paired_bootstrap metric key")
    ap.add_argument("--primary-key", type=str, default="omega_perm", help="Primary counterfactual key")
    ap.add_argument("--n-bootstrap", type=int, default=20_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    in_files = _collect_inputs(args.inputs, args.glob)
    if not in_files:
        raise SystemExit("No input files found. Provide --inputs or --glob.")

    raw_by_seed: Dict[int, dict] = {}
    for p in in_files:
        d = _load_json(p)
        seed = int((d.get("meta") or {}).get("seed", -1))
        if seed < 0:
            continue
        # Keep latest file per seed by timestamp_utc if duplicates exist.
        prev = raw_by_seed.get(seed)
        if prev is None:
            raw_by_seed[seed] = {"path": p, "obj": d}
            continue
        ts_prev = ((prev["obj"].get("meta") or {}).get("timestamp_utc", ""))
        ts_new = ((d.get("meta") or {}).get("timestamp_utc", ""))
        if str(ts_new) >= str(ts_prev):
            raw_by_seed[seed] = {"path": p, "obj": d}

    if not raw_by_seed:
        raise SystemExit("No valid counterfactual files with meta.seed found.")

    seeds = sorted(raw_by_seed.keys())
    first_obj = raw_by_seed[seeds[0]]["obj"]
    score_keys = sorted([k for k in ((first_obj.get("scores") or {}).keys()) if k != "orig"])
    if args.primary_key not in score_keys:
        raise SystemExit(f"primary key {args.primary_key!r} not in available score keys: {score_keys}")

    diffs_by_key_seed: Dict[str, List[np.ndarray]] = {k: [] for k in score_keys}
    per_seed_rows: List[dict] = []

    for seed in seeds:
        rec = raw_by_seed[seed]
        obj = rec["obj"]
        path = rec["path"]

        pb_metric = ((obj.get("paired_bootstrap") or {}).get(args.metric) or {})
        pb_primary = pb_metric.get(args.primary_key) or {}
        scores = obj.get("scores") or {}
        orig = np.asarray(scores.get("orig") or [], dtype=np.float64)

        for key in score_keys:
            cf = np.asarray(scores.get(key) or [], dtype=np.float64)
            if orig.shape != cf.shape:
                raise SystemExit(f"shape mismatch in seed={seed}, key={key}: orig={orig.shape}, cf={cf.shape}")
            diffs_by_key_seed[key].append(orig - cf)

        per_seed_rows.append(
            {
                "seed": int(seed),
                "path": str(path),
                "n_samples": int(orig.size),
                "mean_diff": float(pb_primary.get("mean_diff", float(np.mean(diffs_by_key_seed[args.primary_key][-1]) if orig.size else 0.0))),
                "ci_low": float(pb_primary.get("ci_low", math.nan)),
                "ci_high": float(pb_primary.get("ci_high", math.nan)),
                "p_value": float(pb_primary.get("p_value", 1.0)),
                "p_value_holm": float(pb_primary.get("p_value_holm", 1.0)),
            }
        )

    pooled_family: Dict[str, dict] = {}
    pvals_two: List[float] = []
    family_keys: List[str] = []
    for i, key in enumerate(score_keys):
        r = _bootstrap_hierarchical_mean_diff(
            diffs_by_key_seed[key],
            n_boot=int(args.n_bootstrap),
            seed=int(args.seed) + 1000 * i,
            ci=float(args.ci),
        )
        pooled_family[key] = {
            "mean_diff": float(r.mean_diff),
            "ci_low": float(r.ci_low),
            "ci_high": float(r.ci_high),
            "p_value": float(r.p_value),
            "p_value_one_sided": float(r.p_value_one_sided),
        }
        family_keys.append(key)
        pvals_two.append(float(r.p_value))

    p_holm = _holm_bonferroni(pvals_two)
    for key, ph in zip(family_keys, p_holm):
        pooled_family[key]["p_value_holm_secondary"] = float(ph)

    primary = dict(pooled_family[args.primary_key])
    primary["positive_seed_count"] = int(sum(1 for row in per_seed_rows if float(row["mean_diff"]) > 0.0))
    primary["seed_count"] = int(len(per_seed_rows))
    primary["passed_primary_one_sided"] = bool(
        float(primary["mean_diff"]) > 0.0 and float(primary["p_value_one_sided"]) < float(args.alpha)
    )
    primary["passed_secondary_holm"] = bool(
        float(primary["mean_diff"]) > 0.0 and float(primary["p_value_holm_secondary"]) < float(args.alpha)
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    created_at = _now_iso()
    report = {
        "meta": {
            "created_at": created_at,
            "input_files": [str(p) for p in in_files],
            "selected_files_by_seed": {str(s): str(raw_by_seed[s]["path"]) for s in seeds},
            "seed_count": int(len(seeds)),
            "seeds": [int(s) for s in seeds],
            "metric": str(args.metric),
            "primary_key": str(args.primary_key),
            "n_bootstrap": int(args.n_bootstrap),
            "ci": float(args.ci),
            "alpha": float(args.alpha),
            "holm_scope": "counterfactual family keys (pooled) for the selected metric",
        },
        "primary": primary,
        "per_seed_primary": sorted(per_seed_rows, key=lambda r: int(r["seed"])),
        "pooled_family": {k: pooled_family[k] for k in sorted(pooled_family.keys())},
    }

    out_json = out_dir / "omega_perm_power_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_md = out_dir / "omega_perm_power_table.md"
    out_md.write_text(
        _build_markdown(
            created_at=created_at,
            metric=str(args.metric),
            primary_key=str(args.primary_key),
            alpha=float(args.alpha),
            n_bootstrap=int(args.n_bootstrap),
            files=in_files,
            per_seed_rows=per_seed_rows,
            family_rows=pooled_family,
            primary_row=primary,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"report": str(out_json), "table": str(out_md), "primary": primary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
