#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _budget_millions(budgets: Sequence[float]) -> List[float]:
    return [float(b) / 1_000_000.0 for b in budgets]


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, fc: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#2b2b2b",
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=10, color="#111111")


def _draw_arrow(ax, p0: Tuple[float, float], p1: Tuple[float, float]) -> None:
    arrow = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#333333")
    ax.add_patch(arrow)


def build_fig1_system(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_box(ax, 0.03, 0.25, 0.18, 0.5, "3D CT Volume\n+ Report Context", "#e8f1ff")
    _draw_box(ax, 0.27, 0.25, 0.18, 0.5, "BET\n(Budgeted Evidence\nTokenization)", "#e8ffe8")
    _draw_box(ax, 0.51, 0.25, 0.18, 0.5, "PCG\n(Proof-Carrying\nGeneration)", "#fff6df")
    _draw_box(ax, 0.75, 0.25, 0.2, 0.5, "Verifier + Refusal\nCalibration\n+ Audit Trace", "#ffe9e9")

    _draw_arrow(ax, (0.21, 0.5), (0.27, 0.5))
    _draw_arrow(ax, (0.45, 0.5), (0.51, 0.5))
    _draw_arrow(ax, (0.69, 0.5), (0.75, 0.5))
    _draw_arrow(ax, (0.85, 0.23), (0.60, 0.08))
    _draw_arrow(ax, (0.60, 0.08), (0.36, 0.23))

    ax.text(0.60, 0.03, "Refine / Re-route under fixed budget B = B_enc + B_gen", ha="center", fontsize=10, color="#222222")
    ax.set_title("Figure 1. ProveTok Closed-Loop Pipeline", fontsize=13, pad=8)
    _save_figure(fig, out_path)


def _plot_series_with_ci(ax, x, entries, label: str, color: str, value_key: str = "mean") -> None:
    y = [float(e.get(value_key, 0.0)) for e in entries]
    lo = [float(e.get("ci_low", v)) for e, v in zip(entries, y)]
    hi = [float(e.get("ci_high", v)) for e, v in zip(entries, y)]
    ax.plot(x, y, marker="o", linewidth=2.0, label=label, color=color)
    ax.fill_between(x, lo, hi, color=color, alpha=0.13)


def build_fig2_budget_curves(baselines_json: Path, out_path: Path) -> None:
    d = _load_json(baselines_json)
    budgets = [float(b) for b in d.get("budgets", [])]
    if not budgets:
        raise ValueError(f"No budgets found in {baselines_json}")
    x = _budget_millions(budgets)
    metrics = d.get("metrics", {})
    latency = d.get("latency", {})

    chosen = {
        "provetok_lesionness": ("ProveTok-Lesionness", "#1f77b4"),
        "fixed_grid": ("Fixed-Grid", "#ff7f0e"),
        "roi_variance": ("ROI-Variance", "#2ca02c"),
        "ct2rep_strong": ("CT2Rep-Strong", "#9467bd"),
        "ct2rep_like": ("CT2Rep-NoProof", "#8c564b"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.2))
    ax0, ax1, ax2 = axes

    for key, (label, color) in chosen.items():
        if key in metrics.get("combined", {}):
            _plot_series_with_ci(ax0, x, metrics["combined"][key], label, color)
    ax0.set_title("Combined vs Budget")
    ax0.set_xlabel("Budget (M units)")
    ax0.set_ylabel("Combined")
    ax0.grid(alpha=0.25)

    for key, (label, color) in chosen.items():
        if key in metrics.get("iou", {}) and key in {"provetok_lesionness", "fixed_grid", "roi_variance"}:
            _plot_series_with_ci(ax1, x, metrics["iou"][key], label, color)
    ax1.set_title("IoU (grounding) vs Budget")
    ax1.set_xlabel("Budget (M units)")
    ax1.set_ylabel("IoU")
    ax1.grid(alpha=0.25)

    for key, (label, color) in chosen.items():
        if key in latency.get("warm_time_p95_s", {}) and key in {"provetok_lesionness", "fixed_grid", "roi_variance"}:
            _plot_series_with_ci(ax2, x, latency["warm_time_p95_s"][key], label, color, value_key="p95_s")
    ax2.set_title("Warm P95 Latency vs Budget")
    ax2.set_xlabel("Budget (M units)")
    ax2.set_ylabel("Latency (s)")
    ax2.grid(alpha=0.25)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Figure 2. Multi-Budget Performance and Latency (E0164, real profile)", fontsize=13, y=1.11)
    _save_figure(fig, out_path)


def build_fig3_regret(regret_json: Path, out_path: Path) -> None:
    d = _load_json(regret_json)
    rows = sorted(d.get("test", {}).get("rows", []), key=lambda r: float(r["budget"]))
    if not rows:
        raise ValueError(f"No test rows found in {regret_json}")
    x = [float(r["budget"]) / 1_000_000.0 for r in rows]
    oracle = [float(r.get("oracle_metric", 0.0)) for r in rows]
    achieved = [float(r.get("achieved_metric_test", 0.0)) for r in rows]
    norm_regret = [float(r.get("normalized_regret", 0.0)) for r in rows]

    mean_norm = float(d.get("regret", {}).get("mean_normalized_regret", 0.0))
    naive_point = float(d.get("naive_policies", {}).get("point_estimates", {}).get("always_fixed_grid", {}).get("mean_normalized_regret", 0.0))
    naive_ci = d.get("regret", {}).get("bootstrap", {}).get("naive_policies", {}).get("always_fixed_grid", {})
    naive_lo = float(naive_ci.get("mean_normalized_regret_ci_low", naive_point))
    naive_hi = float(naive_ci.get("mean_normalized_regret_ci_high", naive_point))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2))
    ax0, ax1 = axes

    ax0.plot(x, oracle, marker="o", linewidth=2.1, color="#1f77b4", label="Oracle on test")
    ax0.plot(x, achieved, marker="o", linewidth=2.1, color="#2ca02c", label="Predicted policy (test)")
    ax0.set_title("Oracle vs Predicted Achieved Metric")
    ax0.set_xlabel("Budget (M units)")
    ax0.set_ylabel("Combined")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False)

    ax1.bar(x, norm_regret, width=0.42, color="#1f77b4", alpha=0.75, label="Per-budget normalized regret")
    ax1.axhline(mean_norm, color="#2ca02c", linestyle="-", linewidth=2.0, label=f"Overall mean={mean_norm:.3f}")
    ax1.axhline(naive_point, color="#d62728", linestyle="--", linewidth=2.0, label=f"Naive mean={naive_point:.3f}")
    ax1.fill_between([min(x) - 0.25, max(x) + 0.25], [naive_lo, naive_lo], [naive_hi, naive_hi], color="#d62728", alpha=0.14)
    ax1.set_title("Normalized Regret by Budget")
    ax1.set_xlabel("Budget (M units)")
    ax1.set_ylabel("Normalized regret")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, loc="upper right")

    fig.suptitle("Figure 3. Allocation Regret Sweep (E0161)", fontsize=13, y=1.03)
    _save_figure(fig, out_path)


def build_fig4_counterfactual(omega_json: Path, out_path: Path) -> None:
    d = _load_json(omega_json)
    family = d.get("pooled_family", {})
    ordered = ["omega_perm", "no_cite", "evidence_drop", "cite_swap", "token_perm"]
    keys = [k for k in ordered if k in family]
    if not keys:
        keys = sorted(family.keys())
    if not keys:
        raise ValueError(f"No pooled family found in {omega_json}")

    means = [float(family[k].get("mean_diff", 0.0)) for k in keys]
    lo = [float(family[k].get("ci_low", 0.0)) for k in keys]
    hi = [float(family[k].get("ci_high", 0.0)) for k in keys]
    yerr = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
    p_holm = [float(family[k].get("p_value_holm_secondary", 1.0)) for k in keys]

    colors = []
    for k in keys:
        if k == "omega_perm":
            colors.append("#1f77b4")
        elif k == "no_cite":
            colors.append("#2ca02c")
        else:
            colors.append("#7f7f7f")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))
    ax0, ax1 = axes
    x = list(range(len(keys)))

    ax0.bar(x, means, yerr=yerr, color=colors, alpha=0.85, capsize=4)
    ax0.axhline(0.0, color="#111111", linewidth=1.0)
    ax0.set_xticks(x, keys, rotation=15)
    ax0.set_ylabel("Mean diff (orig - cf)")
    ax0.set_title("Effect Size with 95% CI")
    ax0.grid(axis="y", alpha=0.25)

    safe_p = [max(v, 1e-8) for v in p_holm]
    y = [-1.0 * (0 if v <= 0 else __import__("math").log10(v)) for v in safe_p]
    ax1.bar(x, y, color=colors, alpha=0.85)
    thr = -1.0 * __import__("math").log10(0.05)
    ax1.axhline(thr, color="#d62728", linestyle="--", linewidth=1.8, label="p=0.05")
    ax1.set_xticks(x, keys, rotation=15)
    ax1.set_ylabel("-log10(p_holm)")
    ax1.set_title("Secondary Holm Significance")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, loc="upper right")

    primary = d.get("primary", {})
    title = (
        "Figure 4. Counterfactual Pooled Test (E0167R2) | "
        f"omega_perm mean={float(primary.get('mean_diff', 0.0)):.4f}, "
        f"p1={float(primary.get('p_value_one_sided', 1.0)):.4g}, "
        f"p_holm={float(primary.get('p_value_holm_secondary', 1.0)):.4g}"
    )
    fig.suptitle(title, fontsize=12.5, y=1.03)
    _save_figure(fig, out_path)


def _case_issue_summary(case_json: dict) -> str:
    issues = case_json.get("issues", [])
    if not issues:
        return "no-issue"
    c = Counter([str(i.get("issue_type", "NA")) for i in issues])
    top = c.most_common(2)
    return ",".join([f"{k}:{v}" for k, v in top])


def build_fig5_cases(case_root: Path, out_path: Path, max_cases: int = 3) -> None:
    if not case_root.exists():
        raise FileNotFoundError(f"Case root does not exist: {case_root}")
    case_dirs = [p for p in sorted(case_root.glob("case_*")) if p.is_dir() and (p / "case.png").exists() and (p / "case.json").exists()]
    if not case_dirs:
        raise ValueError(f"No case_* directories with case.png/case.json found under: {case_root}")

    scored = []
    for c in case_dirs:
        d = _load_json(c / "case.json")
        scored.append((len(d.get("issues", [])), c, d))
    scored.sort(key=lambda x: (-x[0], x[1].name))
    picked = scored[: max(1, int(max_cases))]

    fig, axes = plt.subplots(1, len(picked), figsize=(5.2 * len(picked), 4.0))
    if len(picked) == 1:
        axes = [axes]

    for ax, (_, cdir, cjson) in zip(axes, picked):
        img = plt.imread(cdir / "case.png")
        ax.imshow(img)
        cid = cdir.name.replace("case_", "")[:8]
        ax.set_title(f"{cid} | {_case_issue_summary(cjson)}", fontsize=9.5)
        ax.axis("off")

    fig.suptitle(f"Figure 5. Qualitative Cases from {case_root}", fontsize=12.5, y=1.02)
    _save_figure(fig, out_path)


def build_fig6_refusal_calibration(refusal_json: Path, out_path: Path) -> None:
    d = _load_json(refusal_json)
    rows = d.get("test", {}).get("rows", [])
    if not rows:
        raise ValueError(f"No test rows found in {refusal_json}")
    rows = sorted(rows, key=lambda r: float(r.get("budget", 0.0)))

    x = [float(r.get("budget", 0.0)) / 1_000_000.0 for r in rows]

    def _series(bucket: str, key: str) -> List[float]:
        return [float(r.get(bucket, {}).get(key, 0.0)) for r in rows]

    unsupported_nr = _series("no_refusal", "unsupported_rate")
    unsupported_cal = _series("calibrated", "unsupported_rate")
    miss_nr = _series("no_refusal", "critical_miss_rate")
    miss_cal = _series("calibrated", "critical_miss_rate")
    refusal_nr = _series("no_refusal", "refusal_rate")
    refusal_cal = _series("calibrated", "refusal_rate")
    ece_nr = _series("no_refusal", "refusal_ece")
    ece_cal = _series("calibrated", "refusal_ece")

    policy = d.get("refusal_policy", {}) or {}
    tau = float(d.get("best_tau", policy.get("tau_refuse", 0.0)))
    max_miss = float(policy.get("max_critical_miss_rate", d.get("config", {}).get("max_critical_miss_rate", 0.05)))
    max_refusal = float(policy.get("max_refusal_rate", 0.0))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.8))
    (ax0, ax1), (ax2, ax3) = axes

    def _plot2(ax, y0, y1, title: str, ylab: str) -> None:
        ax.plot(x, y0, marker="o", linewidth=2.0, color="#d62728", label="No refusal")
        ax.plot(x, y1, marker="o", linewidth=2.0, color="#2ca02c", label="Calibrated")
        ax.set_title(title)
        ax.set_xlabel("Budget (M units)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)

    _plot2(ax0, unsupported_nr, unsupported_cal, "Unsupported Rate", "unsupported_rate")
    _plot2(ax1, miss_nr, miss_cal, "Critical Miss Rate", "critical_miss_rate")
    _plot2(ax2, refusal_nr, refusal_cal, "Refusal Rate", "refusal_rate")
    _plot2(ax3, ece_nr, ece_cal, "Refusal ECE", "refusal_ece")

    if max_miss > 0:
        ax1.axhline(max_miss, color="#111111", linestyle="--", linewidth=1.6, label=f"Gate={max_miss:g}")
    if max_refusal > 0:
        ax2.axhline(max_refusal, color="#111111", linestyle="--", linewidth=1.6, label=f"Gate={max_refusal:g}")

    # Use a single legend to avoid clutter.
    handles, labels = ax0.get_legend_handles_labels()
    extra_handles, extra_labels = [], []
    for ax in (ax1, ax2):
        h, l = ax.get_legend_handles_labels()
        extra_handles.extend(h)
        extra_labels.extend(l)
    for h, l in zip(extra_handles, extra_labels):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))

    fig.suptitle(f"Figure 6. Refusal Calibration on test (E0144) | tau_refuse={tau:g}", fontsize=13, y=1.08)
    _save_figure(fig, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build unified paper figures for README.")
    ap.add_argument("--baselines-json", type=str, default="outputs/E0164-full/baselines_curve_multiseed.json")
    ap.add_argument("--regret-json", type=str, default="outputs/E0161-full/fig3_regret_sweep.json")
    ap.add_argument("--omega-json", type=str, default="outputs/E0167R2-ct_rate-tsseg-effusion-counterfactual-power-seed20/omega_perm_power_report.json")
    ap.add_argument("--case-root", type=str, default="outputs/E0163-full-v3")
    ap.add_argument("--refusal-json", type=str, default="outputs/E0144-full/figX_refusal_calibration.json")
    ap.add_argument("--out-dir", type=str, default="docs/paper_assets/figures")
    ap.add_argument("--max-cases", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1 = out_dir / "fig1_system_overview.png"
    fig2 = out_dir / "fig2_budget_curves.png"
    fig3 = out_dir / "fig3_regret_sweep.png"
    fig4 = out_dir / "fig4_counterfactual_power.png"
    fig5 = out_dir / "fig5_case_studies.png"
    fig6 = out_dir / "fig6_refusal_calibration.png"

    build_fig1_system(fig1)
    build_fig2_budget_curves(Path(args.baselines_json), fig2)
    build_fig3_regret(Path(args.regret_json), fig3)
    build_fig4_counterfactual(Path(args.omega_json), fig4)
    build_fig5_cases(Path(args.case_root), fig5, max_cases=args.max_cases)
    build_fig6_refusal_calibration(Path(args.refusal_json), fig6)

    manifest = {
        "generated_at_utc": _now_iso(),
        "out_dir": str(out_dir),
        "sources": {
            "baselines_json": str(Path(args.baselines_json).resolve()),
            "regret_json": str(Path(args.regret_json).resolve()),
            "omega_json": str(Path(args.omega_json).resolve()),
            "case_root": str(Path(args.case_root).resolve()),
            "refusal_json": str(Path(args.refusal_json).resolve()),
        },
        "figures": [
            str(fig1),
            str(fig2),
            str(fig3),
            str(fig4),
            str(fig5),
            str(fig6),
        ],
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
