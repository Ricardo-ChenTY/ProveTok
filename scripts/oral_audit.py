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


def _artifact_status() -> List[Dict[str, Any]]:
    checks = [
        ("C0001(real)", [ROOT / "outputs" / "E0164-full" / "baselines_curve_multiseed.json"]),
        ("C0002(real)", [ROOT / "outputs" / "E0161-full" / "fig3_regret_sweep.json"]),
        ("C0003(real)", [ROOT / "outputs" / "E0162-full"]),
        ("C0004(real)", [ROOT / "outputs" / "E0156-grounding_proof_100g_saliency_full" / "figX_grounding_proof.json"]),
        ("C0005(default/real)", [ROOT / "outputs" / "E0144-full" / "figX_refusal_calibration.json"]),
        ("C0006(real)", [ROOT / "outputs" / "E0164-full" / "baselines_curve_multiseed.json"]),
    ]

    out: List[Dict[str, Any]] = []
    for tag, candidates in checks:
        exists = False
        first_existing = ""
        for c in candidates:
            if c.is_dir():
                found = sorted(c.glob("**/*.json"))
                if found:
                    exists = True
                    first_existing = str(found[0])
                    break
            elif c.exists():
                exists = True
                first_existing = str(c)
                break
        out.append(
            {
                "target": tag,
                "exists": bool(exists),
                "path": first_existing if exists else str(candidates[0]),
            }
        )
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
    artifacts = _artifact_status()

    gaps: List[str] = []
    for p in [default_report, real_report]:
        if not p.get("ok"):
            gaps.append(f"{p.get('profile')}: proof_check execution failed")
            continue
        if not p.get("all_proved"):
            for cid, rec in (p.get("claims") or {}).items():
                if not bool(rec.get("proved")):
                    gaps.append(f"{p.get('profile')}::{cid}: {rec.get('summary')}")
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
