#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MetricCI:
    mean: float
    ci_low: float
    ci_high: float


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_ci(curve: dict, *, metric: str, method: str, bidx: int) -> Optional[MetricCI]:
    metrics = curve.get("metrics") or {}
    m = (metrics.get(metric) or {}).get(method)
    if not isinstance(m, list) or bidx >= len(m):
        return None
    row = m[bidx]
    if not isinstance(row, dict):
        return None
    return MetricCI(
        mean=float(row.get("mean", 0.0) or 0.0),
        ci_low=float(row.get("ci_low", 0.0) or 0.0),
        ci_high=float(row.get("ci_high", 0.0) or 0.0),
    )


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{float(v):.{digits}f}"


def _fmt_int(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return str(int(round(float(v))))


def _ratio(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None:
        return None
    d = float(d)
    if d <= 0:
        return None
    return float(n) / d


def build_report(curve: dict) -> Dict[str, Any]:
    budgets: List[float] = list(curve.get("budgets") or [])
    methods: List[str] = list(curve.get("methods") or [])
    seeds: List[int] = list(curve.get("seeds") or [])
    n_samples = int(curve.get("n_samples", 0) or 0)
    n_boot = int(curve.get("n_bootstrap", 0) or 0)
    ci = float(curve.get("ci", 0.95) or 0.95)

    rows: List[Dict[str, Any]] = []
    for bidx, budget in enumerate(budgets):
        row: Dict[str, Any] = {"budget": float(budget), "methods": {}}
        for method in methods:
            # Diagnostics: polarity/citations (added in run_baselines.py).
            n_pred_total = _get_ci(curve, metric="n_frames_pred_total", method=method, bidx=bidx)
            n_pred_pos = _get_ci(curve, metric="n_frames_pred_pos", method=method, bidx=bidx)
            n_pred_abs = _get_ci(curve, metric="n_frames_pred_absent", method=method, bidx=bidx)
            n_cite_total = _get_ci(curve, metric="n_citations_total", method=method, bidx=bidx)
            n_cite_pos = _get_ci(curve, metric="n_citations_pos", method=method, bidx=bidx)
            n_cite_nonpos = _get_ci(curve, metric="n_citations_nonpos", method=method, bidx=bidx)

            iou_pos = _get_ci(curve, metric="iou", method=method, bidx=bidx)
            iou_all = _get_ci(curve, metric="iou_all", method=method, bidx=bidx)

            row["methods"][method] = {
                "n_frames_pred_total": None if n_pred_total is None else n_pred_total.__dict__,
                "n_frames_pred_pos": None if n_pred_pos is None else n_pred_pos.__dict__,
                "n_frames_pred_absent": None if n_pred_abs is None else n_pred_abs.__dict__,
                "n_citations_total": None if n_cite_total is None else n_cite_total.__dict__,
                "n_citations_pos": None if n_cite_pos is None else n_cite_pos.__dict__,
                "n_citations_nonpos": None if n_cite_nonpos is None else n_cite_nonpos.__dict__,
                "iou_pos_only": None if iou_pos is None else iou_pos.__dict__,
                "iou_all_frames": None if iou_all is None else iou_all.__dict__,
                "pos_frame_rate_mean": _ratio(
                    None if n_pred_pos is None else n_pred_pos.mean,
                    None if n_pred_total is None else n_pred_total.mean,
                ),
                "cite_pos_share_mean": _ratio(
                    None if n_cite_pos is None else n_cite_pos.mean,
                    None if n_cite_total is None else n_cite_total.mean,
                ),
            }
        rows.append(row)

    return {
        "meta": {
            "seeds": seeds,
            "n_samples": n_samples,
            "n_bootstrap_curve": n_boot,
            "ci": ci,
            "budgets": budgets,
            "methods": methods,
        },
        "notes": {
            "iou_pos_only": "IoU computed against union lesion mask using union citations over frames with polarity in {present,positive} (positive_only=True).",
            "iou_all_frames": "Diagnostic only: same IoU but union citations over all frames (positive_only=False). Not used for claims because negative statements lack lesion masks.",
        },
        "rows": rows,
    }


def render_markdown(report: Dict[str, Any], *, curve_json: Path) -> str:
    meta = report.get("meta") or {}
    rows = report.get("rows") or []
    methods = list(meta.get("methods") or [])
    budgets = list(meta.get("budgets") or [])

    out: List[str] = []
    out.append("# Table 6. Llama2PCG Polarity/Citation Diagnostics (Grounding Semantics)")
    out.append("")
    out.append(f"- Curve: `{curve_json}`")
    out.append(f"- Budgets: {', '.join(str(int(b)) for b in budgets)}")
    out.append(f"- Seeds: {meta.get('seeds')}")
    out.append(f"- N: {meta.get('n_samples')} (test samples)")
    out.append(f"- n_bootstrap(curve CI): {meta.get('n_bootstrap_curve')}, CI: {meta.get('ci')}")
    out.append("")
    out.append("**Grounding semantics**")
    out.append("- `IoU_pos_only`: 只统计 polarity∈{present,positive} 的 frames 上的 citations（见 `provetok/eval/metrics_grounding.py::_select_positive_citations`）。")
    out.append("- `IoU_all_frames` 仅用于诊断：把所有 frames 的 citations union 后算 IoU。由于 absent/negative statements 没有 lesion mask，对主结论不采用该口径。")
    out.append("")

    header = ["Budget"]
    for method in methods:
        header += [
            f"{method}: pos/total",
            f"{method}: cite_pos/total",
            f"{method}: IoU_pos",
            f"{method}: IoU_all",
        ]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")

    for r in rows:
        b = int(float(r.get("budget", 0.0) or 0.0))
        cells: List[str] = [str(b)]
        by_m = r.get("methods") or {}
        for method in methods:
            mm = by_m.get(method) or {}
            n_pos = ((mm.get("n_frames_pred_pos") or {}) or {}).get("mean")
            n_tot = ((mm.get("n_frames_pred_total") or {}) or {}).get("mean")
            c_pos = ((mm.get("n_citations_pos") or {}) or {}).get("mean")
            c_tot = ((mm.get("n_citations_total") or {}) or {}).get("mean")
            iou_pos = ((mm.get("iou_pos_only") or {}) or {}).get("mean")
            iou_all = ((mm.get("iou_all_frames") or {}) or {}).get("mean")
            cells.append(f"{_fmt_int(n_pos)}/{_fmt_int(n_tot)}")
            cells.append(f"{_fmt_int(c_pos)}/{_fmt_int(c_tot)}")
            cells.append(_fmt(iou_pos, 4))
            cells.append(_fmt(iou_all, 4))
        out.append("| " + " | ".join(cells) + " |")

    out.append("")
    out.append("Notes: 表中为 mean 值；CI 详见曲线 JSON。若 `pos/total` 很低，则 `IoU_pos_only≈0` 可能是口径导致的自然结果；此时需要同时看 frame-level correctness 与 anti-silencing gates。")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose Llama2PCG grounding via polarity/citation stats.")
    ap.add_argument("--curve-json", type=str, required=True, help="baselines_curve_multiseed.json")
    ap.add_argument("--out-dir", type=str, default="outputs/V0006-llama2-grounding-diagnosis")
    ap.add_argument("--write-docs-table", type=str, default="", help="Optional path to write the markdown table (repo-tracked).")
    args = ap.parse_args()

    curve_json = Path(args.curve_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = _load_json(curve_json)
    report = build_report(curve)

    (out_dir / "llama2_grounding_diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = render_markdown(report, curve_json=curve_json)
    (out_dir / "llama2_grounding_diagnosis.md").write_text(md, encoding="utf-8")

    if args.write_docs_table:
        out_path = Path(args.write_docs_table).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "out_json": str(out_dir / 'llama2_grounding_diagnosis.json')}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

