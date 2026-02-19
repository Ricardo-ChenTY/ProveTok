from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def _run_cmd(cmd: List[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)


def _run_proof_check(profile: str) -> Dict[str, Any]:
    proc = _run_cmd([sys.executable, str(ROOT / "scripts" / "proof_check.py"), "--profile", profile], cwd=ROOT)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return {
            "profile": profile,
            "ok": False,
            "error": f"proof_check failed rc={proc.returncode}",
            "stdout": out[-2000:],
            "stderr": (proc.stderr or "").strip()[-2000:],
        }
    try:
        report = json.loads(out)
    except Exception as exc:  # noqa: BLE001
        return {
            "profile": profile,
            "ok": False,
            "error": f"failed to parse proof_check json: {type(exc).__name__}",
            "stdout": out[-2000:],
        }

    checks = report.get("checks") or []
    claim_map = {}
    all_proved = True
    for row in checks:
        cid = str(row.get("claim_id", ""))
        proved = bool(row.get("proved"))
        summary = str(row.get("summary", ""))
        if cid:
            claim_map[cid] = {"proved": proved, "summary": summary}
        all_proved = all_proved and proved
    return {
        "profile": profile,
        "ok": True,
        "all_proved": bool(all_proved),
        "claims": claim_map,
        "raw": report,
    }


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _classify_invalid_gate_artifact(profile: str, claim_id: str, summary: str, details: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return invalid-artifact metadata when a failed claim is due to non-gate evidence.

    We currently enforce this for `real::C0001` because oral gate relies on
    paper-grade evidence (6 budgets, 5 seeds, 20k bootstrap, split=test).
    """
    if str(profile) != "real" or str(claim_id) != "C0001":
        return None

    budgets = details.get("budgets") if isinstance(details, dict) else None
    declared_budgets = details.get("declared_budgets") if isinstance(details, dict) else None
    evaluable_budgets = details.get("evaluable_budgets") if isinstance(details, dict) else None
    seeds = details.get("seeds") if isinstance(details, dict) else None
    per_budget_seed_counts = details.get("per_budget_seed_counts") if isinstance(details, dict) else None
    n_boot = details.get("n_bootstrap") if isinstance(details, dict) else None

    if isinstance(budgets, list) and len(budgets) < 6:
        return {"reason": "insufficient_budgets", "required": 6, "found": len(budgets)}
    if isinstance(declared_budgets, list) and len(declared_budgets) < 6:
        return {"reason": "insufficient_declared_budgets", "required": 6, "found": len(declared_budgets)}
    if isinstance(evaluable_budgets, list) and len(evaluable_budgets) < 6:
        return {"reason": "insufficient_evaluable_budgets", "required": 6, "found": len(evaluable_budgets)}
    if isinstance(seeds, list) and len(seeds) < 5:
        return {"reason": "insufficient_seeds", "required": 5, "found": len(seeds)}
    if isinstance(per_budget_seed_counts, list):
        for row in per_budget_seed_counts:
            if not isinstance(row, dict):
                continue
            n_eval = _to_int(row.get("n_evaluable_seeds", 0), 0)
            if n_eval < 5:
                return {
                    "reason": "insufficient_evaluable_seeds",
                    "required": 5,
                    "found": n_eval,
                    "budget": row.get("budget"),
                }
    if n_boot is not None and _to_int(n_boot, 0) < 20_000:
        return {"reason": "insufficient_bootstrap", "required": 20000, "found": _to_int(n_boot, 0)}

    s = str(summary).lower()
    if "paper-grade c0001" in s and "need >=" in s:
        return {"reason": "paper_grade_requirement_not_met"}
    return None


def _collect_claim_gaps(profile_report: Dict[str, Any]) -> tuple[List[str], List[Dict[str, Any]]]:
    gaps: List[str] = []
    invalid_gate_artifacts: List[Dict[str, Any]] = []
    profile = str(profile_report.get("profile", ""))

    if not profile_report.get("ok"):
        gaps.append(f"{profile}: proof_check execution failed")
        return gaps, invalid_gate_artifacts

    claims = profile_report.get("claims") or {}
    raw_checks = {}
    raw = profile_report.get("raw") or {}
    for row in raw.get("checks") or []:
        if isinstance(row, dict) and row.get("claim_id"):
            raw_checks[str(row["claim_id"])] = row

    for cid, rec in claims.items():
        if bool((rec or {}).get("proved")):
            continue
        summary = str((rec or {}).get("summary", ""))
        row = raw_checks.get(str(cid)) or {}
        details = row.get("details") if isinstance(row, dict) and isinstance(row.get("details"), dict) else {}
        invalid = _classify_invalid_gate_artifact(profile, str(cid), summary, details)
        if invalid is not None:
            artifact = str(details.get("baselines", ""))
            invalid_gate_artifacts.append(
                {
                    "profile": profile,
                    "claim_id": str(cid),
                    "summary": summary,
                    "artifact": artifact,
                    **invalid,
                }
            )
            gaps.append(f"{profile}::{cid}: artifact_invalid_for_gate: {summary}")
        else:
            gaps.append(f"{profile}::{cid}: {summary}")
    return gaps, invalid_gate_artifacts


def _claim_raw_row(profile_report: Dict[str, Any], claim_id: str) -> Dict[str, Any]:
    raw = profile_report.get("raw") or {}
    for row in raw.get("checks") or []:
        if isinstance(row, dict) and str(row.get("claim_id", "")) == str(claim_id):
            return row
    return {}


def _artifact_status(default_report: Dict[str, Any], real_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    c1 = _claim_raw_row(real_report, "C0001")
    c2 = _claim_raw_row(real_report, "C0002")
    c3 = _claim_raw_row(real_report, "C0003")
    c4 = _claim_raw_row(real_report, "C0004")
    c5 = _claim_raw_row(default_report, "C0005") or _claim_raw_row(real_report, "C0005")
    c6 = _claim_raw_row(real_report, "C0006")

    checks = [
        ("C0001(real)", str((c1.get("details") or {}).get("baselines", ROOT / "outputs" / "E0164-full" / "baselines_curve_multiseed.json"))),
        ("C0002(real)", str((c2.get("details") or {}).get("path", ROOT / "outputs" / "E0161-full" / "fig3_regret_sweep.json"))),
        ("C0003(real)", str((c3.get("details") or {}).get("path", ROOT / "outputs" / "E0162-full"))),
        ("C0004(real)", str((c4.get("details") or {}).get("path", ROOT / "outputs" / "E0156-grounding_proof_100g_saliency_seed20" / "figX_grounding_proof.json"))),
        ("C0005(default/real)", str((c5.get("details") or {}).get("path", ROOT / "outputs" / "E0144-full" / "figX_refusal_calibration.json"))),
        ("C0006(real)", str((c6.get("details") or {}).get("path", ROOT / "outputs" / "E0164-full" / "baselines_curve_multiseed.json"))),
    ]

    out: List[Dict[str, Any]] = []
    for tag, path_str in checks:
        p = Path(path_str)
        exists = False
        resolved = path_str
        if p.is_dir():
            found = sorted(p.glob("**/*.json"))
            if found:
                exists = True
                resolved = str(found[0])
        elif p.exists():
            exists = True
            resolved = str(p)
        out.append({"target": tag, "exists": bool(exists), "path": resolved})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="One-shot oral readiness audit (proof_check default+real + key artifact index).")
    ap.add_argument("--sync", action="store_true", help="Run `python scripts/rd_queue.py sync` before audit.")
    ap.add_argument("--out", type=str, default="", help="Optional JSON output path.")
    ap.add_argument("--strict", action="store_true", help="Return non-zero when any gap exists.")
    args = ap.parse_args()

    sync_result: Dict[str, Any] | None = None
    if args.sync:
        proc = _run_cmd([sys.executable, str(ROOT / "scripts" / "rd_queue.py"), "sync"], cwd=ROOT)
        sync_result = {
            "returncode": int(proc.returncode),
            "stdout_tail": (proc.stdout or "").strip()[-2000:],
            "stderr_tail": (proc.stderr or "").strip()[-2000:],
        }

    default_report = _run_proof_check("default")
    real_report = _run_proof_check("real")
    artifacts = _artifact_status(default_report, real_report)

    gaps: List[str] = []
    invalid_gate_artifacts: List[Dict[str, Any]] = []
    for p in [default_report, real_report]:
        claim_gaps, invalids = _collect_claim_gaps(p)
        gaps.extend(claim_gaps)
        invalid_gate_artifacts.extend(invalids)
    for art in artifacts:
        if not bool(art.get("exists")):
            gaps.append(f"missing artifact: {art.get('target')} -> {art.get('path')}")

    report: Dict[str, Any] = {
        "root": str(ROOT),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sync": sync_result,
        "profiles": {
            "default": default_report,
            "real": real_report,
        },
        "artifacts": artifacts,
        "invalid_gate_artifacts": invalid_gate_artifacts,
        "gaps": gaps,
        "ready_for_oral_gate": bool(not gaps),
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote -> {out}")

    if args.strict and gaps:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
