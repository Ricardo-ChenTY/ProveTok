from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _read_table_rows(experiment_md: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse the Experiment List markdown table (minimal parser)."""
    lines = experiment_md.read_text(encoding="utf-8").splitlines()
    header: List[str] = []
    rows: List[Dict[str, str]] = []

    header_line = None
    for ln in lines:
        if ln.strip().startswith("| ID ") and "| 1GPU script" in ln:
            header_line = ln
            break
    if header_line is None:
        raise RuntimeError("Failed to find experiment table header in docs/experiment.md")

    def split_row(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split("|")]
        # Remove leading/trailing empty due to table pipes.
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return parts

    header = split_row(header_line)

    in_table = False
    for ln in lines:
        if ln.strip() == header_line.strip():
            in_table = True
            continue
        if not in_table:
            continue
        if not ln.strip().startswith("|"):
            break
        # skip separator row
        if set(ln.strip()) <= {"|", "-", " "}:
            continue
        cells = split_row(ln)
        if not cells or not cells[0].startswith("E"):
            continue
        row = {h: (cells[i] if i < len(cells) else "") for i, h in enumerate(header)}
        rows.append(row)

    return header, rows


def _strip_backticks(text: str) -> str:
    t = text.strip()
    if t.startswith("`") and t.endswith("`") and len(t) >= 2:
        return t[1:-1].strip()
    return t


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    qdir = root / ".rd_queue"
    (qdir / "logs").mkdir(parents=True, exist_ok=True)
    (qdir / "results").mkdir(parents=True, exist_ok=True)
    print(json.dumps({"root": str(root), "rd_queue": str(qdir)}, indent=2))
    return 0


def cmd_make(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    exp_md = root / "docs" / "experiment.md"
    _, rows = _read_table_rows(exp_md)

    stage = args.stage
    want_ids = set([s.strip() for s in (args.ids or []) if s.strip()])
    jobs: List[Dict[str, Any]] = []

    for row in rows:
        eid = row.get("ID", "").strip()
        if not eid:
            continue
        if want_ids and eid not in want_ids:
            continue

        smoke_cell = row.get("Smoke", "")
        full_cell = row.get("Full", "")
        if stage == "smoke" and "[x]" in smoke_cell:
            continue
        if stage == "full" and "[x]" in full_cell:
            continue

        # Stage-aware command selection:
        # - For smoke, prefer the 1GPU script.
        # - For full, prefer the Multi-GPU script (often used as the "full" command),
        #   and fall back to 1GPU if empty/TBD.
        if stage == "full":
            cmd_cell = row.get("Multi-GPU script", "") or row.get("1GPU script", "")
        else:
            cmd_cell = row.get("1GPU script", "") or row.get("Multi-GPU script", "")
        cmd = _strip_backticks(cmd_cell)
        if not cmd or cmd.upper() == "TBD":
            continue

        jobs.append(
            {
                "id": eid,
                "stage": stage,
                "name": row.get("Goal/Claim", "")[:120],
                "command": cmd,
                "workdir": str(root),
            }
        )

        if args.next and jobs:
            break

    if not jobs:
        print("[rd_queue] No jobs selected.")
        return 1

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"jobs": jobs}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"queue": str(out_path), "num_jobs": len(jobs)}, indent=2))
    return 0


def _update_experiment_md_checkboxes(*, exp_md: Path, results_dir: Path) -> int:
    header, rows = _read_table_rows(exp_md)
    col_idx = {name: i for i, name in enumerate(header)}
    if "Smoke" not in col_idx or "Full" not in col_idx:
        raise RuntimeError("Experiment table missing Smoke/Full columns")

    # Load results
    passed: Dict[Tuple[str, str], bool] = {}
    for p in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        job_id = str(d.get("id") or "").strip()
        stage = str(d.get("stage") or "").strip()
        status = str(d.get("status") or "").strip()
        if job_id and stage:
            # Checkbox semantics are monotonic: once a job stage has a passing
            # run, we keep it as passed even if historical failed artifacts
            # remain in the results directory (e.g. *_failed_<ts>.json).
            passed[(job_id, stage)] = bool(passed.get((job_id, stage), False) or (status == "passed"))

    # Rewrite the table lines in-place.
    lines = exp_md.read_text(encoding="utf-8").splitlines()

    def split_row(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return parts

    def join_row(cells: List[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    updated = 0
    for i, ln in enumerate(lines):
        if not ln.strip().startswith("| E"):
            continue
        cells = split_row(ln)
        if not cells:
            continue
        row_id = cells[col_idx["ID"]].strip() if col_idx.get("ID") is not None and col_idx["ID"] < len(cells) else ""
        if not row_id.startswith("E"):
            continue

        # Update smoke/full if we have a passing result.
        smoke_passed = bool(passed.get((row_id, "smoke")))
        full_passed = bool(passed.get((row_id, "full")))
        if smoke_passed or full_passed:
            # Full implies smoke for checkbox semantics (avoid Full=[x] with Smoke=[ ]).
            cells[col_idx["Smoke"]] = "[x]"
        if full_passed:
            cells[col_idx["Full"]] = "[x]"

        new_ln = join_row(cells)
        if new_ln != ln:
            lines[i] = new_ln
            updated += 1

    if updated:
        exp_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return updated


def cmd_sync(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    exp_md = root / "docs" / "experiment.md"
    results_dir = root / ".rd_queue" / "results"
    updated = _update_experiment_md_checkboxes(exp_md=exp_md, results_dir=results_dir)
    results_md = _write_results_md(root=root, results_dir=results_dir)
    print(json.dumps({"updated_rows": int(updated), "experiment_md": str(exp_md), "results_md": str(results_md)}, indent=2))
    return 0


def _infer_output_dir(command: str) -> str:
    """Best-effort parse for `--output-dir <path>` in a bash -lc command string."""
    try:
        import shlex

        toks = shlex.split(command)
    except Exception:
        toks = command.split()

    out_dir = ""
    for i, t in enumerate(toks):
        if t == "--output-dir" and i + 1 < len(toks):
            out_dir = toks[i + 1]
    return out_dir


def _find_latest(path: Path, *, filename: str) -> Path | None:
    if not path.exists():
        return None
    # Direct hit
    direct = path / filename
    if direct.exists():
        return direct
    # Search recursively and pick latest mtime.
    candidates = list(path.glob(f"**/{filename}"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_key_metrics(artifact_path: Path) -> str:
    try:
        d = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"failed_to_parse({type(exc).__name__})"

    name = artifact_path.name
    if name == "fig2_raw_data.json":
        budgets = d.get("budgets") or []
        combined = d.get("combined_scores") or []
        if budgets and combined:
            return f"Fig2 combined@maxB={combined[-1]:.4f} (B={budgets[-1]})"
        return "Fig2 raw"

    if name == "fig2_multiseed.json":
        budgets = d.get("budgets") or []
        combined = (d.get("metrics") or {}).get("combined") or []
        if budgets and combined:
            last = combined[-1]
            return f"Fig2 CI combined@maxB={last.get('mean',0):.4f} [{last.get('ci_low',0):.4f},{last.get('ci_high',0):.4f}]"
        return "Fig2 multiseed"

    if name == "baselines.json":
        summary = d.get("summary") or {}
        # Prefer a stable subset to keep the line short.
        pk = summary.get("provetok_no_refine") or {}
        fg = summary.get("fixed_grid") or {}
        return f"Baselines iou(prove={pk.get('iou',0):.4f},grid={fg.get('iou',0):.4f})"

    if name == "baselines_multiseed.json":
        summary = d.get("summary") or {}
        pk = summary.get("provetok_no_refine") or {}
        fg = summary.get("fixed_grid") or {}
        return f"Baselines CI iou(prove={pk.get('iou',0):.4f},grid={fg.get('iou',0):.4f})"

    if name == "baselines_curve_multiseed.json":
        budgets = d.get("budgets") or []
        metrics = d.get("metrics") or {}
        combined = metrics.get("combined") or {}
        if budgets and isinstance(combined, dict):
            method = "provetok_lesionness" if combined.get("provetok_lesionness") else "provetok_no_refine"
            pk = (combined.get(method) or [])[-1] if combined.get(method) else {}
            fg = (combined.get("fixed_grid") or [])[-1] if combined.get("fixed_grid") else {}
            if isinstance(pk, dict) and isinstance(fg, dict):
                return (
                    f"Baselines curve@maxB={budgets[-1]:g} "
                    f"combined({method}={pk.get('mean',0):.4f},grid={fg.get('mean',0):.4f})"
                )
        return "Baselines curve"

    if name == "train_lesionness_head.json":
        best = d.get("best_val_f1")
        if isinstance(best, (int, float)):
            return f"Lesionness best_val_f1={best:.4f}"
        return "Lesionness head"

    if name == "train_saliency_cnn3d.json":
        best = d.get("best_val_f1")
        if isinstance(best, (int, float)):
            return f"SaliencyCNN3D best_val_f1={best:.4f}"
        return "SaliencyCNN3D"

    if name == "fig3_results.json":
        regret = d.get("regret") or {}
        mean_regret = regret.get("mean_regret")
        if isinstance(mean_regret, (int, float)):
            return f"Regret mean={mean_regret:.4f}"
        return "Fig3 regret"

    if name == "fig3_regret_sweep.json":
        regret = d.get("regret") or {}
        boot = (regret.get("bootstrap") or {}) if isinstance(regret, dict) else {}
        pol = (boot.get("policies") or {}) if isinstance(boot, dict) else {}
        learned = pol.get("learned") if isinstance(pol, dict) else None
        if isinstance(learned, dict):
            m = learned.get("mean")
            lo = learned.get("ci_low")
            hi = learned.get("ci_high")
            if isinstance(m, (int, float)) and isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                return f"Regret sweep norm={float(m):.4f} [{float(lo):.4f},{float(hi):.4f}]"
        mean_norm = regret.get("mean_normalized_regret")
        if isinstance(mean_norm, (int, float)):
            return f"Regret sweep norm={float(mean_norm):.4f}"
        return "Fig3 regret sweep"

    if name == "figX_grounding_proof.json":
        budgets = d.get("budgets") or []
        pb = d.get("paired_bootstrap") or {}
        if budgets:
            b = budgets[-1]
            key0 = f"{b:.0e}"  # e.g. "5e+06"
            key1 = key0.replace("e+0", "e+").replace("e-0", "e-")  # fallback
            by_base = pb.get(key0) or pb.get(key1) or {}
            fg = by_base.get("fixed_grid") or {}
            iou = (fg.get("iou_union") or {}) if isinstance(fg, dict) else {}
            mean_diff = iou.get("mean_diff")
            p_holm = iou.get("p_value_holm", iou.get("p_value"))
            if isinstance(mean_diff, (int, float)) and isinstance(p_holm, (int, float)):
                return f"Grounding ΔIoU@maxB={mean_diff:.4f} p_holm={p_holm:g}"
        return "Grounding proof"

    if name == "figX_counterfactual.json":
        pb = (d.get("paired_bootstrap") or {}).get("grounding_iou_union_orig_minus_cf") or {}
        key = "omega_perm" if "omega_perm" in pb else (next(iter(pb.keys()), ""))
        if key:
            rec = pb.get(key) or {}
            return f"CF ΔIoU(orig-{key})={rec.get('mean_diff',0):.4f} p_holm={rec.get('p_value_holm',rec.get('p_value',1))}"
        return "Counterfactual"

    if name == "latency_bench_baselines.json":
        pm = d.get("per_method") or {}
        fg = pm.get("fixed_grid") if isinstance(pm, dict) else None
        if isinstance(fg, dict):
            p95 = fg.get("warm_p95_s")
            if isinstance(p95, (int, float)):
                return f"Latency fixed_grid warm_p95={float(p95):.4f}s"
        return "Latency bench"

    if name == "omega_perm_power_report.json":
        primary = d.get("primary") or {}
        md = primary.get("mean_diff")
        p1 = primary.get("p_value_one_sided")
        ph = primary.get("p_value_holm_secondary")
        if isinstance(md, (int, float)) and isinstance(p1, (int, float)) and isinstance(ph, (int, float)):
            return f"OmegaPerm pooled ΔIoU={float(md):.4f} p1={float(p1):.4g} p_holm={float(ph):.4g}"
        return "OmegaPerm power report"

    return artifact_path.name


def _extract_margin_to_threshold(artifact_path: Path) -> str:
    try:
        d = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    name = artifact_path.name
    if name in {"fig3_results.json", "fig3_regret_sweep.json"}:
        regret = d.get("regret") or {}
        boot = (regret.get("bootstrap") or {}) if isinstance(regret, dict) else {}
        ci_high = None
        if isinstance(boot, dict):
            if "mean_normalized_regret_ci_high" in boot:
                ci_high = boot.get("mean_normalized_regret_ci_high")
            else:
                learned = ((boot.get("policies") or {}).get("learned") or {}) if isinstance(boot.get("policies"), dict) else {}
                if isinstance(learned, dict):
                    ci_high = learned.get("ci_high")
        if isinstance(ci_high, (int, float)):
            margin = float(0.15 - float(ci_high))
            return f"regret_ci_margin={margin:+.4f}"
        return ""

    if name == "figX_refusal_calibration.json":
        rows = ((d.get("test") or {}).get("rows") or []) if isinstance(d.get("test"), dict) else []
        if not isinstance(rows, list) or not rows:
            return ""
        miss_margin = float("inf")
        ece_margin = float("inf")
        refusal_margin = float("inf")
        for row in rows:
            cal = row.get("calibrated") or {}
            miss = cal.get("critical_miss_rate")
            ece = cal.get("refusal_ece")
            rr = cal.get("refusal_rate")
            if isinstance(miss, (int, float)):
                miss_margin = min(miss_margin, 0.05 - float(miss))
            if isinstance(ece, (int, float)):
                ece_margin = min(ece_margin, 0.15 - float(ece))
            if isinstance(rr, (int, float)):
                refusal_margin = min(refusal_margin, 0.20 - float(rr))
        parts: List[str] = []
        if miss_margin != float("inf"):
            parts.append(f"miss={miss_margin:+.4f}")
        if ece_margin != float("inf"):
            parts.append(f"ece={ece_margin:+.4f}")
        if refusal_margin != float("inf"):
            parts.append(f"refusal={refusal_margin:+.4f}")
        return ",".join(parts)

    if name == "figX_grounding_proof.json":
        pb = d.get("paired_bootstrap") or {}
        if not isinstance(pb, dict) or not pb:
            return ""
        min_mean = float("inf")
        min_p_margin = float("inf")
        for budget_rec in pb.values():
            if not isinstance(budget_rec, dict):
                continue
            for baseline in ["fixed_grid", "roi_variance"]:
                rec = ((budget_rec.get(baseline) or {}).get("iou_union") or {}) if isinstance(budget_rec.get(baseline), dict) else {}
                mean_diff = rec.get("mean_diff")
                p_val = rec.get("p_value_holm", rec.get("p_value"))
                if isinstance(mean_diff, (int, float)):
                    min_mean = min(min_mean, float(mean_diff))
                if isinstance(p_val, (int, float)):
                    min_p_margin = min(min_p_margin, 0.05 - float(p_val))
        parts = []
        if min_mean != float("inf"):
            parts.append(f"min_delta_iou={min_mean:+.4f}")
        if min_p_margin != float("inf"):
            parts.append(f"min_p_margin={min_p_margin:+.4f}")
        return ",".join(parts)

    if name == "figX_counterfactual.json":
        pb = d.get("paired_bootstrap") or {}
        if not isinstance(pb, dict):
            return ""
        no_cite = (pb.get("grounding_iou_union_orig_minus_cf") or {}).get("no_cite") if isinstance(pb.get("grounding_iou_union_orig_minus_cf"), dict) else None
        cite_swap = (pb.get("unsupported_rate_cf_minus_orig") or {}).get("cite_swap") if isinstance(pb.get("unsupported_rate_cf_minus_orig"), dict) else None
        parts: List[str] = []
        if isinstance(no_cite, dict):
            p = no_cite.get("p_value_holm", no_cite.get("p_value"))
            if isinstance(p, (int, float)):
                parts.append(f"no_cite_p_margin={0.05-float(p):+.4f}")
        if isinstance(cite_swap, dict):
            p = cite_swap.get("p_value_holm", cite_swap.get("p_value"))
            if isinstance(p, (int, float)):
                parts.append(f"cite_swap_p_margin={0.05-float(p):+.4f}")
        return ",".join(parts)

    if name == "baselines_curve_multiseed.json":
        frame_f1_rows = (((d.get("metrics") or {}).get("frame_f1") or {}).get("ct2rep_strong") or [])
        if isinstance(frame_f1_rows, list) and frame_f1_rows:
            last = frame_f1_rows[-1]
            if isinstance(last, dict):
                mean = last.get("mean")
                if isinstance(mean, (int, float)):
                    return f"ct2rep_f1_margin={float(mean)-0.05:+.4f}"
        return ""

    if name == "omega_perm_power_report.json":
        primary = d.get("primary") or {}
        p1 = primary.get("p_value_one_sided")
        ph = primary.get("p_value_holm_secondary")
        parts: List[str] = []
        if isinstance(p1, (int, float)):
            parts.append(f"primary_p1_margin={0.05-float(p1):+.4f}")
        if isinstance(ph, (int, float)):
            parts.append(f"secondary_p_holm_margin={0.05-float(ph):+.4f}")
        return ",".join(parts)

    return ""


def _write_results_md(*, root: Path, results_dir: Path) -> Path:
    out = root / "docs" / "results.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(d)

    # Sort by ended_at (string ISO) if present.
    rows.sort(key=lambda r: str(r.get("ended_at") or ""))

    lines = [
        "# Results",
        "",
        "Generated by `python scripts/rd_queue.py sync` from `.rd_queue/results/*.json`.",
        "",
        "| ended_at | id | stage | status | output | key metrics | margin_to_threshold | log |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in rows:
        ended = str(r.get("ended_at") or "")
        jid = str(r.get("id") or "")
        stage = str(r.get("stage") or "")
        status = str(r.get("status") or "")
        log_path = str(r.get("log_path") or "")
        cmd = str(r.get("command") or "")

        out_dir = _infer_output_dir(cmd)
        output_cell = out_dir
        metrics_cell = ""
        margin_cell = ""
        if out_dir:
            out_path = Path(out_dir)
            if not out_path.is_absolute():
                out_path = (root / out_path).resolve()

            # Try common artifacts (ordered by preference).
            for fname in [
                "omega_perm_power_report.json",
                "fig2_multiseed.json",
                "fig2_raw_data.json",
                "fig3_results.json",
                "fig3_regret_sweep.json",
                "latency_bench_baselines.json",
                "baselines_curve_multiseed.json",
                "baselines_multiseed.json",
                "baselines.json",
                "figX_counterfactual.json",
                "figX_grounding_proof.json",
                "figX_refusal_calibration.json",
                "train_lesionness_head.json",
                "train_saliency_cnn3d.json",
            ]:
                art = _find_latest(out_path, filename=fname)
                if art is not None:
                    output_cell = str(art.relative_to(root)) if root in art.parents else str(art)
                    metrics_cell = _extract_key_metrics(art)
                    margin_cell = _extract_margin_to_threshold(art)
                    break

        # Escape pipes
        def esc(s: str) -> str:
            return s.replace("|", "\\|")

        lines.append(
            "| "
            + " | ".join(
                [
                    esc(ended),
                    esc(jid),
                    esc(stage),
                    esc(status),
                    esc(output_cell),
                    esc(metrics_cell),
                    esc(margin_cell),
                    esc(log_path),
                ]
            )
            + " |"
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _tmux_queue_script() -> Path:
    # Prefer the official skill runner if present.
    p = Path.home() / ".codex" / "skills" / "rd-experiment-runner" / "scripts" / "tmux_queue.py"
    if p.exists():
        return p
    raise FileNotFoundError(f"tmux_queue.py not found at {p}")


def cmd_start(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    queue = Path(args.queue).resolve()
    session = args.session
    script = _tmux_queue_script()

    cmd = [sys.executable, str(script), "start", "--queue", str(queue), "--root", str(root), "--session", session]
    if args.continue_on_fail:
        cmd.append("--continue-on-fail")
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def cmd_worker(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    queue = Path(args.queue).resolve()
    script = _tmux_queue_script()
    cmd = [sys.executable, str(script), "worker", "--queue", str(queue), "--root", str(root)]
    if args.continue_on_fail:
        cmd.append("--continue-on-fail")
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ProveTok .rd_queue helper (wraps rd-experiment-runner tmux_queue).")
    ap.add_argument("--root", type=str, default=".", help="Project root (default: .)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create .rd_queue/{logs,results} directories.")
    p_init.set_defaults(func=cmd_init)

    p_make = sub.add_parser("make", help="Build a .rd_queue/queue.json from docs/experiment.md.")
    p_make.add_argument("--stage", type=str, default="smoke", choices=["smoke", "full"])
    p_make.add_argument("--ids", type=str, nargs="*", default=None, help="Optional E#### ids to include")
    p_make.add_argument("--next", action="store_true", help="Only include the next unfinished experiment for this stage")
    p_make.add_argument("--out", type=str, default=str(ROOT / ".rd_queue" / "queue.json"))
    p_make.set_defaults(func=cmd_make)

    p_start = sub.add_parser("start", help="Start the queue in a detached tmux session.")
    p_start.add_argument("--queue", type=str, default=str(ROOT / ".rd_queue" / "queue.json"))
    p_start.add_argument("--session", type=str, default="rdq-smoke")
    p_start.add_argument("--continue-on-fail", action="store_true")
    p_start.set_defaults(func=cmd_start)

    p_worker = sub.add_parser("worker", help="Run the queue in the current process (no tmux).")
    p_worker.add_argument("--queue", type=str, default=str(ROOT / ".rd_queue" / "queue.json"))
    p_worker.add_argument("--continue-on-fail", action="store_true")
    p_worker.set_defaults(func=cmd_worker)

    p_sync = sub.add_parser("sync", help="Update docs/experiment.md checkboxes from .rd_queue/results/*.json.")
    p_sync.set_defaults(func=cmd_sync)

    return ap


def main() -> int:
    if shutil.which("tmux") is None:
        # Not fatal: worker mode does not require tmux, but start does.
        pass
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
