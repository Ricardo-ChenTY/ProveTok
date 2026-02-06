#!/usr/bin/env python3
"""Generate an effect-size oriented summary from scripts/proof_check.py JSON.

This is intentionally not a new proof rule. It helps answer:
- "How big are the deltas that passed the proof gates?"
- "Which budgets/baselines are carrying the claim?"

Outputs:
- <out>/effect_size_report.json
- <out>/effect_size_table.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_proof_check(profile: str) -> Dict[str, Any]:
    out = subprocess.check_output([sys.executable, str(ROOT / "scripts" / "proof_check.py"), "--profile", profile], cwd=ROOT)
    return json.loads(out.decode("utf-8"))


def _load_proof(path: Optional[str], profile: str) -> Dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return _run_proof_check(profile)


def _index_checks(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c in report.get("checks") or []:
        cid = c.get("claim_id")
        if isinstance(cid, str):
            out[cid] = c
    return out


def _c0001_effect(check: Dict[str, Any]) -> Dict[str, Any]:
    det = check.get("details") or {}
    rows = det.get("rows") or []
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        budget = r.get("budget")
        combined = r.get("combined") or {}
        iou = r.get("iou_union") or {}
        latency = r.get("latency") or {}
        unsupported = r.get("unsupported") or {}
        p95_baseline = latency.get("p95_baseline")
        p95_delta = latency.get("p95_delta")
        p95_delta_ratio = None
        if isinstance(p95_baseline, (int, float)) and isinstance(p95_delta, (int, float)) and float(p95_baseline) > 0:
            p95_delta_ratio = float(p95_delta) / float(p95_baseline)
        out_rows.append(
            {
                "budget": budget,
                "combined_mean_diff": combined.get("mean_diff"),
                "combined_p_holm": combined.get("p_holm"),
                "combined_passed": combined.get("passed"),
                "iou_mean_diff": iou.get("mean_diff"),
                "iou_p_holm": iou.get("p_holm"),
                "iou_passed": iou.get("passed"),
                "latency_p95_delta_ratio": p95_delta_ratio,
                "latency_passed_p95": latency.get("passed_p95"),
                "unsupported_mean_diff": unsupported.get("mean_diff"),
                "unsupported_passed": unsupported.get("passed"),
            }
        )
    return {
        "proved": bool(check.get("proved", False)),
        "artifact": det.get("baselines"),
        "method": det.get("method"),
        "baseline": det.get("baseline"),
        "rows": out_rows,
    }


def _c0002_effect(check: Dict[str, Any]) -> Dict[str, Any]:
    det = check.get("details") or {}
    paper = det.get("paper") or {}
    return {
        "proved": bool(check.get("proved", False)),
        "artifact": det.get("path"),
        "mean_normalized_regret_ci_high": paper.get("mean_normalized_regret_ci_high"),
        "naive_always_fixed_grid_ci_low": paper.get("naive_always_fixed_grid_ci_low"),
        "passed": paper.get("passed"),
    }


def _c0003_effect(check: Dict[str, Any]) -> Dict[str, Any]:
    det = check.get("details") or {}
    return {
        "proved": bool(check.get("proved", False)),
        "artifact": det.get("path"),
        "notes": "Counterfactual is better explained via the JSON artifact + appendix; this report focuses on effect-size tables for multi-budget claims.",
    }


def _c0004_effect(check: Dict[str, Any]) -> Dict[str, Any]:
    det = check.get("details") or {}
    per = det.get("per_budget") or {}
    out: Dict[str, Any] = {"proved": bool(check.get("proved", False)), "artifact": det.get("path"), "per_baseline": {}}
    for baseline, rows in per.items():
        if not isinstance(rows, list):
            continue
        out_rows: List[Dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            out_rows.append(
                {
                    "budget": r.get("budget"),
                    "mean_diff": r.get("mean_diff"),
                    "p_holm": r.get("p_holm"),
                    "passed": r.get("passed"),
                }
            )
        out["per_baseline"][baseline] = out_rows
    return out


def _c0005_effect(check: Dict[str, Any]) -> Dict[str, Any]:
    det = check.get("details") or {}
    per = det.get("per_budget") or []
    out_rows: List[Dict[str, Any]] = []
    for r in per:
        if not isinstance(r, dict):
            continue
        out_rows.append(
            {
                "budget": r.get("budget"),
                "unsupported_delta": r.get("unsupported_delta"),
                "critical_miss_rate": r.get("critical_miss_rate"),
                "refusal_rate": r.get("refusal_rate"),
                "refusal_ece": r.get("refusal_ece"),
                "passed_miss": r.get("passed_miss"),
                "passed_ece": r.get("passed_ece"),
                "passed_refusal_rate": r.get("passed_refusal_rate"),
                "passed_unsupported": r.get("passed_unsupported"),
            }
        )
    return {
        "proved": bool(check.get("proved", False)),
        "artifact": det.get("path"),
        "tau_refuse": det.get("tau_refuse"),
        "rows": out_rows,
    }


def _c0006_effect(check: Dict[str, Any]) -> Dict[str, Any]:
    det = check.get("details") or {}
    return {
        "proved": bool(check.get("proved", False)),
        "artifact": det.get("path"),
        "methods": det.get("methods"),
        "has_budget_accounting": det.get("has_budget_accounting"),
        "ct2rep_strong": det.get("ct2rep_strong"),
    }


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "true" if v else "false"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:.{nd}f}"


def _md_table_c0001(rows: List[Dict[str, Any]]) -> str:
    hdr = [
        "budget",
        "d_combined",
        "p_holm_combined",
        "d_iou",
        "p_holm_iou",
        "p95_delta_ratio",
        "d_unsupported",
        "pass",
    ]
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        passed = bool(r.get("combined_passed")) and bool(r.get("iou_passed")) and bool(r.get("latency_passed_p95")) and bool(r.get("unsupported_passed"))
        out.append(
            "| "
            + " | ".join(
                [
                    str(int(float(r.get("budget") or 0))),
                    _fmt(r.get("combined_mean_diff")),
                    _fmt(r.get("combined_p_holm")),
                    _fmt(r.get("iou_mean_diff")),
                    _fmt(r.get("iou_p_holm")),
                    _fmt(r.get("latency_p95_delta_ratio")),
                    _fmt(r.get("unsupported_mean_diff")),
                    "true" if passed else "false",
                ]
            )
            + " |"
        )
    return "\n".join(out) + "\n"


def _md_table_c0004(per_baseline: Dict[str, List[Dict[str, Any]]]) -> str:
    parts: List[str] = []
    for baseline, rows in per_baseline.items():
        parts.append(f"\n### Baseline: `{baseline}`\n")
        hdr = ["budget", "mean_diff", "p_holm", "passed"]
        out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
        for r in rows:
            out.append(
                "| "
                + " | ".join(
                    [
                        str(int(float(r.get("budget") or 0))),
                        _fmt(r.get("mean_diff")),
                        _fmt(r.get("p_holm")),
                        "true" if bool(r.get("passed")) else "false",
                    ]
                )
                + " |"
            )
        parts.append("\n".join(out) + "\n")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Effect-size oriented report from proof_check JSON.")
    ap.add_argument("--out", type=str, default="outputs/V0002-effect", help="Output directory")
    ap.add_argument("--profile", type=str, default="default,real", help="Comma-separated profiles to run (default: default,real)")
    ap.add_argument("--proof-json", type=str, default="", help="Optional path to an existing proof_check JSON (applies to single-profile use)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    profiles = [p.strip() for p in args.profile.split(",") if p.strip()]
    if not profiles:
        raise SystemExit("no profiles provided")

    report: Dict[str, Any] = {"generated_at_utc": _utc_now(), "profiles": {}}
    md_parts: List[str] = []
    md_parts.append("# Effect Size Report (From proof_check)\n")
    md_parts.append(f"- generated_at_utc: `{report['generated_at_utc']}`\n\n")

    for profile in profiles:
        raw = _load_proof(args.proof_json or None, profile=profile)
        idx = _index_checks(raw)
        effects = {
            "C0001": _c0001_effect(idx.get("C0001", {})),
            "C0002": _c0002_effect(idx.get("C0002", {})),
            "C0003": _c0003_effect(idx.get("C0003", {})),
            "C0004": _c0004_effect(idx.get("C0004", {})),
            "C0005": _c0005_effect(idx.get("C0005", {})),
            "C0006": _c0006_effect(idx.get("C0006", {})),
        }
        report["profiles"][profile] = {
            "ok": all(bool(effects[c].get("proved", False)) for c in ["C0001", "C0002", "C0003", "C0004", "C0005", "C0006"]),
            "effects": effects,
        }

        md_parts.append(f"## Profile: `{profile}`\n\n")
        md_parts.append(f"- C0001 artifact: `{effects['C0001'].get('artifact')}`\n")
        md_parts.append(f"- C0004 artifact: `{effects['C0004'].get('artifact')}`\n")
        md_parts.append(f"- C0005 artifact: `{effects['C0005'].get('artifact')}`\n\n")

        md_parts.append("### C0001 (Per Budget)\n\n")
        md_parts.append(_md_table_c0001(effects["C0001"].get("rows") or []))

        md_parts.append("\n### C0002 (Regret)\n\n")
        md_parts.append(
            "| mean_regret_ci_high | naive_fixed_grid_ci_low | passed |\n|---|---|---|\n"
            + f"| {_fmt(effects['C0002'].get('mean_normalized_regret_ci_high'))} | {_fmt(effects['C0002'].get('naive_always_fixed_grid_ci_low'))} | {str(bool(effects['C0002'].get('passed'))).lower()} |\n"
        )

        md_parts.append("\n### C0004 (Per Budget, Per Baseline)\n")
        md_parts.append(_md_table_c0004((effects["C0004"].get("per_baseline") or {})))

        md_parts.append("\n### C0005 (Per Budget)\n\n")
        hdr = ["budget", "unsupported_delta", "critical_miss_rate", "refusal_rate", "refusal_ece", "passed"]
        lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
        for r in effects["C0005"].get("rows") or []:
            passed = bool(r.get("passed_miss")) and bool(r.get("passed_ece")) and bool(r.get("passed_refusal_rate")) and bool(r.get("passed_unsupported"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(int(float(r.get("budget") or 0))),
                        _fmt(r.get("unsupported_delta")),
                        _fmt(r.get("critical_miss_rate")),
                        _fmt(r.get("refusal_rate")),
                        _fmt(r.get("refusal_ece")),
                        "true" if passed else "false",
                    ]
                )
                + " |"
            )
        md_parts.append("\n".join(lines) + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "effect_size_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "effect_size_table.md").write_text("".join(md_parts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

