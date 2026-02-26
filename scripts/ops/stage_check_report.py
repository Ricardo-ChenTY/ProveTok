#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def _load_latest_run_json(stage_dir: Path) -> Path:
    candidates = sorted(
        list(stage_dir.rglob("baselines.json")) + list(stage_dir.rglob("baselines_multiseed.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No baselines json found under: {stage_dir}")
    return candidates[0]


def _method_checks(
    raw_method: Dict[str, Any],
    *,
    max_parse_failure_rate: float,
    min_citation_nonempty_rate: float,
    max_abnormal_output_rate: float,
) -> Dict[str, Any]:
    pred_total = [float(x) for x in raw_method.get("n_frames_pred_total", [])]
    pred_pos = [float(x) for x in raw_method.get("n_frames_pred_pos", [])]
    pos_with_cite = [float(x) for x in raw_method.get("n_frames_pos_with_citations", [])]
    unsupported = [float(x) for x in raw_method.get("unsupported", [])]
    overclaim = [float(x) for x in raw_method.get("overclaim", [])]
    warm = [float(x) for x in raw_method.get("warm_time_s", [])]

    n = len(pred_total)
    parse_fail_n = sum(1 for v in pred_total if v <= 0.0)
    parse_failure_rate = (parse_fail_n / n) if n > 0 else 1.0

    n_pos_samples = len(pred_pos)
    cite_ok_n = 0
    for p, c in zip(pred_pos, pos_with_cite):
        if p <= 0.0:
            cite_ok_n += 1
        elif c > 0.0:
            cite_ok_n += 1
    citation_nonempty_rate = (cite_ok_n / n_pos_samples) if n_pos_samples > 0 else 0.0

    abnormal_output_rate = _safe_mean(unsupported) + _safe_mean(overclaim)
    warm_time_s = _safe_mean(warm)

    pass_parse = parse_failure_rate <= max_parse_failure_rate
    pass_cite = citation_nonempty_rate >= min_citation_nonempty_rate
    pass_abnormal = abnormal_output_rate <= max_abnormal_output_rate
    passed = bool(pass_parse and pass_cite and pass_abnormal)

    return {
        "n_samples": int(n),
        "parse_failure_rate": float(parse_failure_rate),
        "citation_nonempty_rate": float(citation_nonempty_rate),
        "abnormal_output_rate": float(abnormal_output_rate),
        "warm_time_s_mean": float(warm_time_s),
        "pass_parse": bool(pass_parse),
        "pass_citation": bool(pass_cite),
        "pass_abnormal": bool(pass_abnormal),
        "pass": bool(passed),
    }


def _render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Stage Check Report: {report['stage_name']}")
    lines.append("")
    lines.append(f"- Overall pass: `{report['overall_pass']}`")
    lines.append(f"- Stage dir: `{report['stage_dir']}`")
    lines.append(f"- Run json: `{report['run_json']}`")
    lines.append("")
    lines.append("## Method Checks")
    lines.append("")
    lines.append("| method | n | parse_fail | cite_nonempty | abnormal | warm_s | pass |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for name, m in sorted(report.get("methods", {}).items()):
        lines.append(
            f"| {name} | {m['n_samples']} | {m['parse_failure_rate']:.4f} | "
            f"{m['citation_nonempty_rate']:.4f} | {m['abnormal_output_rate']:.4f} | "
            f"{m['warm_time_s_mean']:.4f} | `{m['pass']}` |"
        )
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    t = report.get("thresholds", {})
    lines.append(f"- max_parse_failure_rate: `{t.get('max_parse_failure_rate')}`")
    lines.append(f"- min_citation_nonempty_rate: `{t.get('min_citation_nonempty_rate')}`")
    lines.append(f"- max_abnormal_output_rate: `{t.get('max_abnormal_output_rate')}`")
    lines.append("")
    return "\n".join(lines)


def build_report(
    *,
    stage_name: str,
    stage_dir: Path,
    run_json: Path | None,
    max_parse_failure_rate: float,
    min_citation_nonempty_rate: float,
    max_abnormal_output_rate: float,
) -> Tuple[Dict[str, Any], Path]:
    stage_dir = stage_dir.resolve()
    if run_json is None:
        run_json = _load_latest_run_json(stage_dir)
    run_json = run_json.resolve()

    payload = json.loads(run_json.read_text(encoding="utf-8"))
    raw = payload.get("raw", {})
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid run json, missing non-empty `raw`: {run_json}")

    method_reports: Dict[str, Any] = {}
    for method, raw_method in raw.items():
        if not isinstance(raw_method, dict):
            continue
        method_reports[str(method)] = _method_checks(
            raw_method,
            max_parse_failure_rate=max_parse_failure_rate,
            min_citation_nonempty_rate=min_citation_nonempty_rate,
            max_abnormal_output_rate=max_abnormal_output_rate,
        )

    if not method_reports:
        raise ValueError(f"No valid method entries found in raw: {run_json}")

    overall_pass = all(bool(v.get("pass")) for v in method_reports.values())
    report = {
        "stage_name": stage_name,
        "stage_dir": str(stage_dir),
        "run_json": str(run_json),
        "overall_pass": bool(overall_pass),
        "thresholds": {
            "max_parse_failure_rate": float(max_parse_failure_rate),
            "min_citation_nonempty_rate": float(min_citation_nonempty_rate),
            "max_abnormal_output_rate": float(max_abnormal_output_rate),
        },
        "methods": method_reports,
    }
    return report, run_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate per-stage integrity/quality check report.")
    ap.add_argument("--stage-name", type=str, required=True)
    ap.add_argument("--stage-dir", type=str, required=True)
    ap.add_argument("--run-json", type=str, default="", help="Optional explicit baselines json path.")
    ap.add_argument("--max-parse-failure-rate", type=float, default=0.20)
    ap.add_argument("--min-citation-nonempty-rate", type=float, default=0.80)
    ap.add_argument("--max-abnormal-output-rate", type=float, default=0.75)
    ap.add_argument("--out-json", type=str, default="", help="Default: <stage-dir>/stage_check_report.json")
    ap.add_argument("--out-md", type=str, default="", help="Default: <stage-dir>/stage_check_report.md")
    args = ap.parse_args()

    stage_dir = Path(args.stage_dir).expanduser()
    run_json = Path(args.run_json).expanduser() if str(args.run_json).strip() else None

    report, resolved_run_json = build_report(
        stage_name=str(args.stage_name),
        stage_dir=stage_dir,
        run_json=run_json,
        max_parse_failure_rate=float(args.max_parse_failure_rate),
        min_citation_nonempty_rate=float(args.min_citation_nonempty_rate),
        max_abnormal_output_rate=float(args.max_abnormal_output_rate),
    )

    out_json = Path(args.out_json).expanduser() if str(args.out_json).strip() else stage_dir / "stage_check_report.json"
    out_md = Path(args.out_md).expanduser() if str(args.out_md).strip() else stage_dir / "stage_check_report.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")

    print(f"[OK] run_json={resolved_run_json}")
    print(f"[OK] stage_report_json={out_json}")
    print(f"[OK] stage_report_md={out_md}")
    print(f"[OK] overall_pass={report['overall_pass']}")

    if not bool(report["overall_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
