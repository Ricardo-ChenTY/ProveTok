#!/usr/bin/env python3
"""Heuristic 'strength' grader on top of scripts/proof_check.py.

This does NOT change proof rules. It helps answer questions like:
- Which claims are only barely passing?
- Which ones are medium / weak (close to thresholds)?
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STRENGTH_ORDER = ["failed", "barely", "medium", "strong"]


@dataclass(frozen=True)
class Strength:
    level: str  # failed|barely|medium|strong
    reasons: List[str]


def _run_proof_check() -> Dict[str, Any]:
    out = subprocess.check_output([sys.executable, "scripts/proof_check.py"], cwd=Path(__file__).resolve().parents[1])
    return json.loads(out.decode("utf-8"))


def _load_report(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return _run_proof_check()
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _count_passed(rows: List[Dict[str, Any]], key: str) -> int:
    n = 0
    for r in rows:
        rec = r.get(key) or {}
        if isinstance(rec, dict) and bool(rec.get("passed", False)):
            n += 1
    return n


def _c0001_strength(check: Dict[str, Any]) -> Strength:
    if not check.get("proved", False):
        return Strength("failed", ["proof_check: proved=false"])
    det = check.get("details") or {}
    rows = det.get("rows") or []
    rule = det.get("rule") or {}
    need = int(rule.get("need_passed_budgets", 4))

    passed_combined = _count_passed(rows, "combined")
    passed_iou = _count_passed(rows, "iou_union")
    reasons = [f"combined_pass={passed_combined}/{len(rows)} (need {need})", f"iou_pass={passed_iou}/{len(rows)} (need {need})"]

    failing_iou = [
        int(r.get("budget", -1))
        for r in rows
        if isinstance(r.get("iou_union"), dict)
        and (not bool(r["iou_union"].get("passed", False)))
        and float(r["iou_union"].get("mean_diff", 0.0)) > 0.0
    ]
    if failing_iou:
        reasons.append(f"iou positive but not significant at budgets={sorted(failing_iou)}")

    if passed_combined <= need or passed_iou <= need:
        return Strength("barely", reasons)
    if passed_combined == need + 1 or passed_iou == need + 1:
        return Strength("medium", reasons)
    return Strength("strong", reasons)


def _c0004_strength(check: Dict[str, Any]) -> Strength:
    if not check.get("proved", False):
        return Strength("failed", ["proof_check: proved=false"])
    det = check.get("details") or {}
    per = det.get("per_budget") or {}
    rule = det.get("rule") or {}
    need = int(rule.get("need_passed_budgets", 4))

    reasons: List[str] = []
    levels: List[str] = []
    for base, rows in per.items():
        if not isinstance(rows, list):
            continue
        passed = sum(1 for r in rows if bool((r.get("passed") if isinstance(r, dict) else False)))
        reasons.append(f"{base}_pass={passed}/{len(rows)} (need {need})")
        if passed == need:
            levels.append("barely")
        elif passed == need + 1:
            levels.append("medium")
        elif passed > need + 1:
            levels.append("strong")
        else:
            levels.append("failed")

        failing = [
            int(r.get("budget", -1))
            for r in rows
            if isinstance(r, dict) and (not bool(r.get("passed", False))) and float(r.get("mean_diff", 0.0)) > 0.0
        ]
        if failing:
            reasons.append(f"{base}: positive-but-not-sig budgets={sorted(failing)}")

    # weakest baseline determines overall strength
    overall = min(levels, key=lambda x: STRENGTH_ORDER.index(x)) if levels else "failed"
    return Strength(overall, reasons)


def _c0005_strength(check: Dict[str, Any]) -> Strength:
    if not check.get("proved", False):
        return Strength("failed", ["proof_check: proved=false"])
    det = check.get("details") or {}
    rows = det.get("per_budget") or []
    tau = det.get("best_tau")
    max_rr = float(det.get("max_refusal_rate", 0.2))
    if not isinstance(rows, list) or not rows:
        return Strength("failed", ["missing per_budget rows"])

    max_refusal = max(float(r.get("refusal_rate", 0.0)) for r in rows)
    max_miss = max(float(r.get("critical_miss_rate", 0.0)) for r in rows)
    max_ece = max(float(r.get("refusal_ece", 0.0)) for r in rows)
    improved = sum(1 for r in rows if bool(r.get("passed_unsupported", False)))
    need = int(det.get("rule", {}).get("need_passed_budgets", 4))

    reasons = [
        f"tau_refuse={tau}",
        f"unsupported_improved={improved}/{len(rows)} (need {need})",
        f"max_refusal_rate={max_refusal:.4f} (limit {max_rr:.2f})",
        f"max_critical_miss_rate={max_miss:.4f} (limit {float(det.get('max_critical_miss_rate', 0.05)):.2f})",
        f"max_refusal_ece={max_ece:.4f} (limit {float(det.get('max_refusal_ece', 0.15)):.2f})",
    ]

    # Strength by distance to refusal-rate cap (primary "anti-gaming" safeguard).
    if max_refusal >= max_rr - 1e-12:
        return Strength("barely", reasons + ["refusal_rate hits cap (anti-gaming margin is zero)"])
    if max_refusal >= max_rr - 0.03:
        return Strength("medium", reasons + ["refusal_rate close to cap"])
    return Strength("strong", reasons)


def _default_strength(check: Dict[str, Any]) -> Strength:
    return Strength("strong" if bool(check.get("proved", False)) else "failed", [])


def grade_strength(report: Dict[str, Any]) -> Dict[str, Any]:
    checks = report.get("checks") or []
    out: List[Dict[str, Any]] = []
    for c in checks:
        cid = str(c.get("claim_id", ""))
        if cid == "C0001":
            s = _c0001_strength(c)
        elif cid == "C0004":
            s = _c0004_strength(c)
        elif cid == "C0005":
            s = _c0005_strength(c)
        else:
            s = _default_strength(c)
        out.append({"claim_id": cid, "proved": bool(c.get("proved", False)), "strength": s.level, "reasons": s.reasons})

    # sort weakest -> strongest
    out.sort(key=lambda r: STRENGTH_ORDER.index(str(r["strength"])))
    return {"graded": out, "order": STRENGTH_ORDER}


def _to_md(graded: Dict[str, Any]) -> str:
    rows = graded["graded"]
    lines = []
    lines.append("# Proof Strength (Heuristic)\n")
    lines.append("| claim_id | proved | strength | reasons |\n")
    lines.append("|---|---:|---:|---|\n")
    for r in rows:
        reasons = "; ".join(str(x) for x in r.get("reasons") or [])
        lines.append(f"| {r['claim_id']} | {'yes' if r['proved'] else 'no'} | {r['strength']} | {reasons} |\n")
    lines.append("\n")
    lines.append("Notes: this is a heuristic layer; `scripts/proof_check.py` remains the final judge.\n")
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, default="", help="Optional proof_check JSON path. If empty, runs scripts/proof_check.py.")
    ap.add_argument("--format", type=str, choices=["json", "md"], default="md")
    args = ap.parse_args()

    report = _load_report(args.in_path)
    graded = grade_strength(report)
    if args.format == "json":
        print(json.dumps(graded, indent=2, ensure_ascii=False))
    else:
        print(_to_md(graded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

