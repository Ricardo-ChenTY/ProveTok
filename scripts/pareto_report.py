#!/usr/bin/env python3
"""Generate a multi-objective Pareto/latency report from baselines_curve_multiseed.json.

This is a lightweight, audit-friendly "oral vNext" artifact generator: it does not
run new experiments. It turns an existing baselines curve artifact into:
- per-budget method table (quality / trust / latency)
- Pareto frontier membership (all methods, and optionally a gated subset)

It is intentionally conservative: the input artifact is the source of truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Objective:
    name: str
    direction: str  # "max" | "min"
    source: str  # "metrics" | "latency"
    field: str  # e.g. "mean" or "p95_s"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_objective(spec: str) -> Objective:
    # Format: "<name>:<max|min>"
    if ":" not in spec:
        raise ValueError(f"invalid objective '{spec}', expected '<name>:<max|min>'")
    name, direction = spec.rsplit(":", 1)
    direction = direction.strip().lower()
    name = name.strip()
    if direction not in ("max", "min"):
        raise ValueError(f"invalid objective '{spec}', direction must be max|min")

    # Infer source + field from name.
    # - metrics: metrics[name][method][i] has {"mean","ci_low","ci_high"}
    # - latency: latency[name][method][i] has {"mean","ci_low","ci_high"} or {"p95_s","ci_low","ci_high"}
    if name.startswith("warm_time_") and name.endswith("_s"):
        source = "latency"
        field = "p95_s" if name.endswith("_p95_s") else "mean"
    else:
        source = "metrics"
        field = "mean"

    return Objective(name=name, direction=direction, source=source, field=field)


def _get_scalar(rec: Any, field: str) -> Optional[float]:
    if not isinstance(rec, dict):
        return None
    v = rec.get(field)
    if v is None and field == "mean":
        v = rec.get("mean_s")
    if v is None and field == "p95_s":
        v = rec.get("p95")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _get_value(d: Dict[str, Any], obj: Objective, method: str, budget_index: int) -> Optional[float]:
    if obj.source == "metrics":
        metric_table = (d.get("metrics") or {}).get(obj.name) or {}
        rows = metric_table.get(method)
        if not isinstance(rows, list) or budget_index >= len(rows):
            return None
        return _get_scalar(rows[budget_index], obj.field)

    if obj.source == "latency":
        latency_table = d.get("latency") or {}
        rows = (latency_table.get(obj.name) or {}).get(method)
        if not isinstance(rows, list) or budget_index >= len(rows):
            return None
        return _get_scalar(rows[budget_index], obj.field)

    raise ValueError(f"unknown objective source: {obj.source}")


def _dominates(a: Dict[str, float], b: Dict[str, float], objectives: List[Objective], eps: float = 1e-12) -> bool:
    strictly_better = False
    for obj in objectives:
        va = a[obj.name]
        vb = b[obj.name]
        if obj.direction == "max":
            if va + eps < vb:
                return False
            if va > vb + eps:
                strictly_better = True
        else:  # min
            if va > vb + eps:
                return False
            if va + eps < vb:
                strictly_better = True
    return strictly_better


def _pareto_frontier(items: List[Tuple[str, Dict[str, float]]], objectives: List[Objective]) -> List[str]:
    front: List[str] = []
    for i, (name_i, vec_i) in enumerate(items):
        dominated = False
        for j, (name_j, vec_j) in enumerate(items):
            if i == j:
                continue
            if _dominates(vec_j, vec_i, objectives):
                dominated = True
                break
        if not dominated:
            front.append(name_i)
    return sorted(front)


def _fmt(v: Optional[float], *, ndigits: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{v:.{ndigits}f}"


def _build_markdown_table(
    rows: List[Dict[str, Any]],
    objectives: List[Objective],
    baseline_method: str,
    frontier_gated: List[str],
    frontier_all: List[str],
) -> str:
    # Keep the table compact: show objective values + gate flags + frontier membership.
    headers = ["method"] + [obj.name for obj in objectives] + ["gate_ok", "pareto_gated", "pareto_all"]
    out_lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    def _sort_key(r: Dict[str, Any]) -> Tuple[int, str]:
        m = r["method"]
        if m == baseline_method:
            return (0, m)
        if m == "provetok_lesionness":
            return (1, m)
        return (2, m)

    for r in sorted(rows, key=_sort_key):
        m = r["method"]
        vals = r["values"]
        gate_ok = bool((r.get("gates") or {}).get("passed", False))
        is_gated = m in frontier_gated
        is_all = m in frontier_all
        line = [m] + [_fmt(vals.get(obj.name)) for obj in objectives] + [str(gate_ok), str(is_gated), str(is_all)]
        out_lines.append("| " + " | ".join(line) + " |")
    return "\n".join(out_lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Pareto + latency/trust report from baselines_curve_multiseed.json.")
    ap.add_argument("--in", dest="in_path", type=str, required=True, help="Path to baselines_curve_multiseed.json")
    ap.add_argument("--out", dest="out_dir", type=str, required=True, help="Output directory (will be created)")
    ap.add_argument("--tag", type=str, default="", help="Optional tag to include in outputs (e.g., 'real_profile').")
    ap.add_argument("--baseline-method", type=str, default="fixed_grid", help="Baseline method for delta/gates (default: fixed_grid)")
    ap.add_argument(
        "--objectives",
        type=str,
        nargs="*",
        default=["combined:max", "iou:max", "unsupported:min", "warm_time_p95_s:min"],
        help="Objectives, format '<name>:<max|min>' (default: combined,max; iou,max; unsupported,min; warm_time_p95_s,min)",
    )
    ap.add_argument("--gate-unsupported-delta-abs", type=float, default=0.05, help="Gate: unsupported(method)-unsupported(baseline) <= this")
    ap.add_argument(
        "--gate-warm-p95-delta-ratio",
        type=float,
        default=0.05,
        help="Gate: (warm_p95(method)-warm_p95(baseline))/warm_p95(baseline) <= this",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    tag = args.tag.strip()

    d = _load_json(in_path)
    budgets = d.get("budgets") or []
    methods = d.get("methods") or []
    if not isinstance(budgets, list) or not budgets:
        raise SystemExit("input missing 'budgets'")
    if not isinstance(methods, list) or not methods:
        raise SystemExit("input missing 'methods'")

    objectives = [_parse_objective(x) for x in args.objectives]
    baseline_method = args.baseline_method
    if baseline_method not in methods:
        raise SystemExit(f"baseline method '{baseline_method}' not present in methods={methods}")

    # Precompute baseline series needed for gates.
    unsupported_name = "unsupported"
    warm_p95_name = "warm_time_p95_s"
    if unsupported_name not in (d.get("metrics") or {}):
        raise SystemExit(f"input missing metrics['{unsupported_name}']")
    if warm_p95_name not in (d.get("latency") or {}):
        raise SystemExit(f"input missing latency['{warm_p95_name}']")

    per_budget: List[Dict[str, Any]] = []
    frontier_counts_all: Dict[str, int] = {m: 0 for m in methods}
    frontier_counts_gated: Dict[str, int] = {m: 0 for m in methods}

    for bi, b in enumerate(budgets):
        # Collect vectors.
        items_all: List[Tuple[str, Dict[str, float]]] = []
        items_gated: List[Tuple[str, Dict[str, float]]] = []
        rows_md: List[Dict[str, Any]] = []

        base_unsupported = _get_value(d, Objective(unsupported_name, "min", "metrics", "mean"), baseline_method, bi)
        base_warm_p95 = _get_value(d, Objective(warm_p95_name, "min", "latency", "p95_s"), baseline_method, bi)

        for m in methods:
            values: Dict[str, float] = {}
            missing = False
            for obj in objectives:
                v = _get_value(d, obj, m, bi)
                if v is None:
                    missing = True
                    break
                values[obj.name] = v
            if missing:
                continue

            gate: Dict[str, Any] = {"passed": False}
            unsupported_v = _get_value(d, Objective(unsupported_name, "min", "metrics", "mean"), m, bi)
            warm_p95_v = _get_value(d, Objective(warm_p95_name, "min", "latency", "p95_s"), m, bi)
            if base_unsupported is not None and unsupported_v is not None:
                gate["unsupported_delta_abs"] = float(unsupported_v - base_unsupported)
                gate["unsupported_tol_abs"] = float(args.gate_unsupported_delta_abs)
            if base_warm_p95 is not None and warm_p95_v is not None and base_warm_p95 > 0:
                gate["warm_p95_delta_ratio"] = float((warm_p95_v - base_warm_p95) / base_warm_p95)
                gate["warm_p95_tol_ratio"] = float(args.gate_warm_p95_delta_ratio)

            passed = True
            if "unsupported_delta_abs" in gate:
                passed = passed and (gate["unsupported_delta_abs"] <= gate["unsupported_tol_abs"])
            if "warm_p95_delta_ratio" in gate:
                passed = passed and (gate["warm_p95_delta_ratio"] <= gate["warm_p95_tol_ratio"])
            gate["passed"] = bool(passed)

            items_all.append((m, values))
            if gate["passed"]:
                items_gated.append((m, values))

            rows_md.append({"method": m, "values": values, "gates": gate})

        frontier_all = _pareto_frontier(items_all, objectives)
        frontier_gated = _pareto_frontier(items_gated, objectives) if items_gated else []
        for m in frontier_all:
            frontier_counts_all[m] += 1
        for m in frontier_gated:
            frontier_counts_gated[m] += 1

        for r in rows_md:
            m = r["method"]
            r["pareto_all"] = bool(m in frontier_all)
            r["pareto_gated"] = bool(m in frontier_gated)

        per_budget.append(
            {
                "budget": float(b),
                "frontier_all": frontier_all,
                "frontier_gated": frontier_gated,
                "rows": rows_md,
            }
        )

    report = {
        "generated_at_utc": _utc_now(),
        "input": str(in_path),
        "tag": tag,
        "baseline_method": baseline_method,
        "objectives": [obj.__dict__ for obj in objectives],
        "gates": {
            "unsupported_delta_abs": float(args.gate_unsupported_delta_abs),
            "warm_p95_delta_ratio": float(args.gate_warm_p95_delta_ratio),
            "baseline_method": baseline_method,
        },
        "meta": d.get("meta") or {},
        "budget_mode": d.get("budget_mode"),
        "budgets": [float(x) for x in budgets],
        "methods": methods,
        "summary": {
            "frontier_counts_all": frontier_counts_all,
            "frontier_counts_gated": frontier_counts_gated,
        },
        "per_budget": per_budget,
    }

    suffix = f"-{tag}" if tag else ""
    json_path = out_dir / f"pareto_report{suffix}.json"
    _write_json(json_path, report)

    md_lines: List[str] = []
    md_lines.append("# Pareto Report\n")
    md_lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`\n")
    md_lines.append(f"- input: `{in_path}`\n")
    if tag:
        md_lines.append(f"- tag: `{tag}`\n")
    md_lines.append(f"- baseline_method: `{baseline_method}`\n")
    md_lines.append("- objectives:\n")
    for obj in objectives:
        md_lines.append(f"  - `{obj.name}` ({obj.direction}, source={obj.source}, field={obj.field})\n")
    md_lines.append("- gates:\n")
    md_lines.append(f"  - unsupported_delta_abs <= {args.gate_unsupported_delta_abs}\n")
    md_lines.append(f"  - warm_p95_delta_ratio <= {args.gate_warm_p95_delta_ratio}\n")
    md_lines.append("\n## Summary\n")
    md_lines.append(f"- frontier_counts_gated: `{frontier_counts_gated}`\n")
    md_lines.append(f"- frontier_counts_all: `{frontier_counts_all}`\n")

    md_lines.append("\n## Per Budget\n")
    for rec in per_budget:
        b = rec["budget"]
        md_lines.append(f"\n### Budget {int(b)}\n")
        md_lines.append(f"- frontier_gated: `{rec['frontier_gated']}`\n")
        md_lines.append(f"- frontier_all: `{rec['frontier_all']}`\n\n")
        md_lines.append(
            _build_markdown_table(
                rec["rows"],
                objectives=objectives,
                baseline_method=baseline_method,
                frontier_gated=rec["frontier_gated"],
                frontier_all=rec["frontier_all"],
            )
        )

    md_path = out_dir / f"pareto_table{suffix}.md"
    _write_text(md_path, "".join(md_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

