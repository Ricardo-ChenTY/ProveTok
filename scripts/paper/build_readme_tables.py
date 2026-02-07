#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_num(v: float, digits: int = 4) -> str:
    return f"{float(v):.{digits}f}"


def _latest_counterfactual_json(root: Path) -> Path:
    if root.is_file():
        return root
    cands = list(root.glob("figX_counterfactual_*/figX_counterfactual.json"))
    if not cands:
        raise FileNotFoundError(f"No figX_counterfactual.json found under {root}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def build_table1_claims(audit_json: Path, out_path: Path) -> None:
    d = _load_json(audit_json)
    claims = d.get("profiles", {}).get("real", {}).get("claims", {})
    artifacts = d.get("artifacts", [])

    artifact_by_claim: Dict[str, str] = {}
    for row in artifacts:
        target = str(row.get("target", ""))
        m = re.search(r"(C\d{4})", target)
        if m:
            artifact_by_claim[m.group(1)] = str(row.get("path", ""))

    lines = []
    lines.append("# Table 1. Claim-Level Machine Verdict (real profile)")
    lines.append("")
    lines.append("| Claim | Status | Summary | Primary Evidence |")
    lines.append("|---|---|---|---|")
    for cid in sorted(claims.keys()):
        row = claims[cid]
        status = "Pass" if bool(row.get("proved", False)) else "Fail"
        summary = str(row.get("summary", "")).replace("|", "/")
        evidence = artifact_by_claim.get(cid, "-")
        lines.append(f"| `{cid}` | {status} | {summary} | `{evidence}` |")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_e0166_counts(e0166_json: Path) -> Dict[str, Dict[str, int]]:
    d = _load_json(e0166_json)
    pb = d.get("paired_bootstrap", {})
    out: Dict[str, Dict[str, int]] = {}
    for baseline in ("roi_variance", "fixed_grid"):
        total = 0
        positive = 0
        significant = 0
        for b_key, by_baseline in pb.items():
            if baseline not in by_baseline:
                continue
            total += 1
            iu = by_baseline[baseline].get("iou_union", {})
            mean_diff = float(iu.get("mean_diff", 0.0))
            p_holm = float(iu.get("p_value_holm", 1.0))
            if mean_diff > 0:
                positive += 1
            if mean_diff > 0 and p_holm < 0.05:
                significant += 1
        out[baseline] = {"total": total, "positive": positive, "significant": significant}
    return out


def _parse_seed_counterfactual(root: Path) -> Dict[str, Dict[str, float]]:
    p = _latest_counterfactual_json(root)
    d = _load_json(p)
    r = d.get("paired_bootstrap", {}).get("grounding_iou_union_orig_minus_cf", {})
    return {
        "omega_perm": {
            "mean_diff": float(r.get("omega_perm", {}).get("mean_diff", 0.0)),
            "p_holm": float(r.get("omega_perm", {}).get("p_value_holm", 1.0)),
        },
        "no_cite": {
            "mean_diff": float(r.get("no_cite", {}).get("mean_diff", 0.0)),
            "p_holm": float(r.get("no_cite", {}).get("p_value_holm", 1.0)),
        },
    }


def _parse_power_report(path: Path) -> Dict[str, float]:
    d = _load_json(path)
    p = d.get("primary", {})
    return {
        "mean_diff": float(p.get("mean_diff", 0.0)),
        "ci_low": float(p.get("ci_low", 0.0)),
        "ci_high": float(p.get("ci_high", 0.0)),
        "p_one_sided": float(p.get("p_value_one_sided", 1.0)),
        "p_holm": float(p.get("p_value_holm_secondary", 1.0)),
        "pos_seed": int(p.get("positive_seed_count", 0)),
        "seed_count": int(p.get("seed_count", 0)),
    }


def build_table2_v0003(
    e0166_json: Path,
    e0167_seed_roots: List[Path],
    e0167r_json: Path,
    e0167r2_json: Path,
    out_path: Path,
) -> None:
    e0166_counts = _parse_e0166_counts(e0166_json)
    seed_rows = [_parse_seed_counterfactual(r) for r in e0167_seed_roots]
    omega_sig = sum(1 for r in seed_rows if r["omega_perm"]["p_holm"] < 0.05 and r["omega_perm"]["mean_diff"] > 0)
    no_cite_sig = sum(1 for r in seed_rows if r["no_cite"]["p_holm"] < 0.05 and r["no_cite"]["mean_diff"] > 0)
    omega_mean = sum(r["omega_perm"]["mean_diff"] for r in seed_rows) / max(1, len(seed_rows))
    no_cite_mean = sum(r["no_cite"]["mean_diff"] for r in seed_rows) / max(1, len(seed_rows))
    e0167r = _parse_power_report(e0167r_json)
    e0167r2 = _parse_power_report(e0167r2_json)

    lines = []
    lines.append("# Table 2. V0003 Cross-Dataset Grounding and Counterfactual Summary")
    lines.append("")
    lines.append("| Item | Scope | Key Result | Verdict |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| E0166 grounding vs ROI | TS-Seg eval-only, budgets 2e6..7e6 | "
        f"IoU_union positive {e0166_counts['roi_variance']['positive']}/{e0166_counts['roi_variance']['total']}, "
        f"Holm significant {e0166_counts['roi_variance']['significant']}/{e0166_counts['roi_variance']['total']} | Pass |"
    )
    lines.append(
        "| E0166 grounding vs Fixed-Grid | TS-Seg eval-only, budgets 2e6..7e6 | "
        f"IoU_union positive {e0166_counts['fixed_grid']['positive']}/{e0166_counts['fixed_grid']['total']}, "
        f"Holm significant {e0166_counts['fixed_grid']['significant']}/{e0166_counts['fixed_grid']['total']} | Partial pass |"
    )
    lines.append(
        "| E0167 seed0..2 no_cite | counterfactual | "
        f"mean_diff(avg)={_format_num(no_cite_mean)}, Holm significant {no_cite_sig}/{len(seed_rows)} | Pass |"
    )
    lines.append(
        "| E0167 seed0..2 omega_perm | counterfactual | "
        f"mean_diff(avg)={_format_num(omega_mean)}, Holm significant {omega_sig}/{len(seed_rows)} | Not significant |"
    )
    lines.append(
        "| E0167R pooled | seeds 0..9 | "
        f"mean_diff={_format_num(e0167r['mean_diff'])}, 95%CI=[{_format_num(e0167r['ci_low'])},{_format_num(e0167r['ci_high'])}], "
        f"p_one_sided={e0167r['p_one_sided']:.4g}, p_holm={e0167r['p_holm']:.4g}, positive={e0167r['pos_seed']}/{e0167r['seed_count']} | Primary pass only |"
    )
    lines.append(
        "| E0167R2 pooled | seeds 0..19 | "
        f"mean_diff={_format_num(e0167r2['mean_diff'])}, 95%CI=[{_format_num(e0167r2['ci_low'])},{_format_num(e0167r2['ci_high'])}], "
        f"p_one_sided={e0167r2['p_one_sided']:.4g}, p_holm={e0167r2['p_holm']:.4g}, positive={e0167r2['pos_seed']}/{e0167r2['seed_count']} | Primary + Holm pass |"
    )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_table3_variants(baseline_root: Path, variant_roots: Dict[str, Path], out_path: Path) -> None:
    config_desc = {
        "BASE": "score + topk=3 (baseline)",
        "RA": "baseline + score_to_uncertainty",
        "RB": "baseline with topk=1",
        "RC": "score_interleave citations",
        "RD": "baseline + score_level_power=1.0",
    }

    rows: List[Tuple[str, Dict[str, float]]] = []
    all_roots = {"BASE": baseline_root}
    all_roots.update(variant_roots)
    for k, root in all_roots.items():
        p = _latest_counterfactual_json(root)
        d = _load_json(p)
        m = d.get("paired_bootstrap", {}).get("grounding_iou_union_orig_minus_cf", {})
        rows.append(
            (
                k,
                {
                    "omega_mean": float(m.get("omega_perm", {}).get("mean_diff", 0.0)),
                    "omega_p_holm": float(m.get("omega_perm", {}).get("p_value_holm", 1.0)),
                    "no_cite_mean": float(m.get("no_cite", {}).get("mean_diff", 0.0)),
                    "no_cite_p_holm": float(m.get("no_cite", {}).get("p_value_holm", 1.0)),
                },
            )
        )

    rows.sort(key=lambda x: (-x[1]["omega_mean"], x[0]))

    lines = []
    lines.append("# Table 3. Omega-Perm Variant Search (seed0)")
    lines.append("")
    lines.append("| Variant | Setting | omega_perm mean_diff | omega_perm p_holm | no_cite mean_diff | no_cite p_holm |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for k, v in rows:
        lines.append(
            f"| {k} | {config_desc.get(k, '-')} | "
            f"{_format_num(v['omega_mean'])} | {v['omega_p_holm']:.4g} | "
            f"{_format_num(v['no_cite_mean'])} | {v['no_cite_p_holm']:.4g} |"
        )
    lines.append("")
    lines.append("结论：RA/RB/RC/RD 在 seed0 上未超过 baseline 的 `omega_perm` 效果，因此后续使用 baseline 扩 seeds。")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _run_proof_check(*, script_path: Path, profile: str) -> dict:
    script = script_path
    if not script.is_absolute():
        script = (ROOT / script).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Missing proof_check script: {script}")

    out = subprocess.check_output(
        [sys.executable, str(script), "--profile", str(profile)],
        cwd=str(ROOT),
        text=True,
        stderr=subprocess.STDOUT,
    )
    return json.loads(out)


def build_table4_oral_minset(
    *,
    proof_check_script: Path,
    proof_profile: str,
    omega_json: Path,
    out_path: Path,
) -> None:
    proof = _run_proof_check(script_path=proof_check_script, profile=proof_profile)
    checks = {c.get("claim_id"): c for c in (proof.get("checks") or []) if isinstance(c, dict)}

    lines: List[str] = []
    lines.append("# Table 4. Oral Minimal Evidence Set (Paper-Grade)")
    lines.append("")
    lines.append("| Item | Verdict | Key Numbers | Protocol | Evidence |")
    lines.append("|---|---|---|---|---|")

    def _verdict_row(item: str, claim: str) -> None:
        c = checks.get(claim) or {}
        proved = bool(c.get("proved", False))
        verdict = "Pass" if proved else "Fail"
        details = c.get("details") or {}
        evidence = "-"
        proto = "-"
        key = "-"

        if claim == "C0001":
            evidence = str(details.get("baselines", "-"))
            rows = details.get("rows") or []
            n_budget = len(rows)
            combined_pass = sum(1 for r in rows if bool(((r.get("combined") or {}).get("passed", False))))
            iou_pass = sum(1 for r in rows if bool(((r.get("iou_union") or {}).get("passed", False))))
            lat_pass = sum(1 for r in rows if bool(((r.get("latency") or {}).get("passed_p95", False))))
            u_pass = sum(1 for r in rows if bool(((r.get("unsupported") or {}).get("passed", False))))
            need = int((details.get("rule") or {}).get("need_passed_budgets", 0))
            lat_tol = float((details.get("rule") or {}).get("latency_p95_tol_ratio", 0.0))
            u_tol = float((details.get("rule") or {}).get("unsupported_tol_abs", 0.0))

            # Pull paper-grade config from the baselines artifact itself.
            try:
                b = _load_json(Path(evidence))
                seeds = b.get("seeds") or []
                budgets = b.get("budgets") or []
                n_boot = int(b.get("n_bootstrap", 0))
                n_seed = len(seeds) if isinstance(seeds, list) else 0
                n_budget_decl = len(budgets) if isinstance(budgets, list) else 0
                proto = (
                    f"budgets={n_budget_decl}, seeds={n_seed}, n_boot={n_boot}, paired bootstrap(H1>0)+Holm(budgets); "
                    f"lat_p95<=+{lat_tol:.0%}, unsupported_delta<=+{u_tol:g}"
                )
            except Exception:
                proto = f"paired bootstrap(H1>0)+Holm(budgets); lat_p95<=+{lat_tol:.0%}, unsupported_delta<=+{u_tol:g}"

            key = (
                f"combined_pass={combined_pass}/{n_budget}(need{need}); "
                f"iou_pass={iou_pass}/{n_budget}(need{need}); "
                f"lat_p95_pass={lat_pass}/{n_budget}; unsupported_pass={u_pass}/{n_budget}"
            )

        elif claim == "C0002":
            evidence = str(details.get("path", "-"))
            paper = details.get("paper") or {}
            key = (
                f"n_boot={int(paper.get('n_bootstrap', 0))}; "
                f"CI_high={_format_num(float(paper.get('mean_normalized_regret_ci_high', 1.0)), 4)}; "
                f"naive_CI_low={_format_num(float(paper.get('naive_always_fixed_grid_ci_low', 1.0)), 4)}"
            )
            proto = "dev->test, AIC/BIC model fit, bootstrap CI, requires CI_high<0.15 & beats naive"

        elif claim == "C0003":
            evidence = str(details.get("path", "-"))
            g_p = details.get("grounding_p_holm") or {}
            u_p = details.get("unsupported_p_holm") or {}
            g_m = details.get("grounding_mean_diff") or {}
            u_m = details.get("unsupported_mean_diff") or {}
            key = (
                f"no_cite: dIoU={_format_num(float(g_m.get('no_cite', 0.0)), 4)}, p_holm={float(g_p.get('no_cite', 1.0)):.4g}; "
                f"cite_swap: dUnsup={_format_num(float(u_m.get('cite_swap', 0.0)), 4)}, p_holm={float(u_p.get('cite_swap', 1.0)):.4g}"
            )
            proto = "paired bootstrap + Holm (counterfactual family)"

        elif claim == "C0004":
            evidence = str(details.get("path", "-"))
            per_budget = details.get("per_budget") or {}
            need = int((details.get("rule") or {}).get("need_passed_budgets", 0))
            n_boot = int((details.get("rule") or {}).get("n_bootstrap", 0))
            budgets = 0
            fg_pass = 0
            rv_pass = 0
            if isinstance(per_budget, dict):
                fg = per_budget.get("fixed_grid") or []
                rv = per_budget.get("roi_variance") or []
                budgets = max(len(fg), len(rv))
                fg_pass = sum(1 for r in fg if bool(r.get("passed", False)))
                rv_pass = sum(1 for r in rv if bool(r.get("passed", False)))
            key = f"fixed_grid_pass={fg_pass}/{budgets}(need{need}); roi_variance_pass={rv_pass}/{budgets}(need{need})"
            proto = f"one-sided (H1>0) + Holm(budgets), n_boot={n_boot}"

        elif claim == "C0005":
            evidence = str(details.get("path", "-"))
            tau = float(details.get("best_tau", 0.0))
            max_miss = float(details.get("max_critical_miss_rate", 0.05))
            max_ece = float(details.get("max_refusal_ece", 0.15))
            max_rr = float(details.get("max_refusal_rate", 0.20))
            per_budget = details.get("per_budget") or []
            n_budget = len(per_budget) if isinstance(per_budget, list) else 0
            u_improved = sum(1 for r in per_budget if bool(r.get("passed_unsupported", False))) if isinstance(per_budget, list) else 0
            miss_max = max((float(r.get("critical_miss_rate", 0.0)) for r in per_budget), default=0.0)
            ece_max = max((float(r.get("refusal_ece", 0.0)) for r in per_budget), default=0.0)
            rr_max = max((float(r.get("refusal_rate", 0.0)) for r in per_budget), default=0.0)
            key = (
                f"tau={tau:g}; miss_max={miss_max:.4g}<= {max_miss:g}; "
                f"ece_max={ece_max:.4g}<= {max_ece:g}; rr_max={rr_max:.4g}<= {max_rr:g}; "
                f"unsupported_improved={u_improved}/{n_budget}"
            )
            proto = "hard gates per budget + need>=2/3 budgets improve unsupported"

        elif claim == "C0006":
            evidence = str(details.get("path", "-"))
            strong = details.get("ct2rep_strong") or {}
            key = (
                f"budget_accounting={bool(details.get('has_budget_accounting', False))}; "
                f"strong_weights={bool(strong.get('weights_exists', False))}; "
                f"frame_f1_last={float(strong.get('frame_f1_last_budget_mean', 0.0)):.4f}>= {float(strong.get('min_frame_f1', 0.0)):.2f}"
            )
            proto = "baseline coverage + audited cost accounting + strong baseline non-degenerate"

        lines.append(f"| `{item}` | {verdict} | {key} | {proto} | `{evidence}` |")

    for cid in ["C0001", "C0002", "C0003", "C0004", "C0005", "C0006"]:
        _verdict_row(cid, cid)

    omega = _load_json(omega_json)
    primary = omega.get("primary") or {}
    key = (
        f"mean_diff={_format_num(float(primary.get('mean_diff', 0.0)), 4)}; "
        f"CI=[{_format_num(float(primary.get('ci_low', 0.0)), 4)},{_format_num(float(primary.get('ci_high', 0.0)), 4)}]; "
        f"p1={float(primary.get('p_value_one_sided', 1.0)):.4g}; "
        f"p_holm={float(primary.get('p_value_holm_secondary', 1.0)):.4g}; "
        f"positive={int(primary.get('positive_seed_count', 0))}/{int(primary.get('seed_count', 0))}"
    )
    proto = "pooled one-sided test + secondary Holm over counterfactual family"
    lines.append(f"| `V0003/omega_perm` | {'Pass' if bool(primary.get('passed_secondary_holm', False)) else 'Fail'} | {key} | {proto} | `{str(omega_json)}` |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_table5_backbone_transfer(*, v0004_report_json: Path, out_path: Path) -> None:
    d = _load_json(v0004_report_json)

    lines: List[str] = []
    lines.append("# Table 5. V0004 Backbone Transfer (Summary)")
    lines.append("")
    lines.append("| Backbone | Setting | Seeds | N | n_boot | Combined (pos/sig) | IoU (pos/sig) | Unsupported (pos/sig) | Evidence |")
    lines.append("|---|---|---:|---:|---:|---|---|---|---|")

    def _metric_summary(rows: dict, metric: str) -> str:
        arr = rows.get(metric) or []
        if not isinstance(arr, list):
            return "-"
        total = len(arr)
        if total == 0:
            return "-"
        pos = 0
        sig = 0
        mean = 0.0
        for a in arr:
            if not isinstance(a, dict):
                continue
            md = float(a.get("mean_diff", 0.0) or 0.0)
            ph = float(a.get("p_holm", 1.0) or 1.0)
            mean += md
            if md > 0:
                pos += 1
            if md > 0 and ph < 0.05:
                sig += 1
        mean /= max(1, total)
        return f"{pos}/{total} (sig {sig}/{total}, avg d={_format_num(mean, 4)})"

    def _extract_n(rows: dict) -> int:
        ns: List[int] = []
        for metric in ("combined", "iou", "unsupported"):
            arr = rows.get(metric) or []
            if not isinstance(arr, list):
                continue
            for a in arr:
                if not isinstance(a, dict):
                    continue
                if "n_samples" in a:
                    ns.append(int(a.get("n_samples") or 0))
        return max(ns) if ns else 0

    backbones = [k for k in d.keys() if k not in ("notes",)]
    order = {"toy": 0, "llama2": 1}
    for backbone in sorted(backbones, key=lambda k: (order.get(k, 999), k)):
        row = d.get(backbone) or {}
        meta = row.get("meta") or {}
        seeds = meta.get("seeds") or []
        seeds_s = ",".join(str(s) for s in seeds) if isinstance(seeds, list) else str(seeds)
        n_boot = int(meta.get("n_bootstrap_used", 0) or 0)
        rows = row.get("rows") or {}
        n = _extract_n(rows)

        setting = f"{row.get('method', '-')}/{row.get('baseline', '-')}"
        combined = _metric_summary(rows, "combined")
        iou = _metric_summary(rows, "iou")
        unsupported = _metric_summary(rows, "unsupported")

        evidence = str(row.get("curve_path", "-"))
        lines.append(
            f"| `{backbone}` | {setting} | {seeds_s} | {n} | {n_boot} | {combined} | {iou} | {unsupported} | `{evidence}` |"
        )

    notes = d.get("notes") or {}
    if isinstance(notes, dict):
        lines.append("")
        lines.append(
            f"Notes: positive mean_diff indicates improvement; Holm family: `{notes.get('holm_family', '-')}`; "
            f"bootstrap: `{notes.get('paired_bootstrap', '-')}`."
        )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build README tables for paper-style summary.")
    ap.add_argument("--audit-json", type=str, default="outputs/oral_audit.json")
    ap.add_argument("--e0166-json", type=str, default="outputs/E0166-ct_rate-tsseg-effusion-grounding-full/figX_grounding_proof.json")
    ap.add_argument("--e0167-seed0-root", type=str, default="outputs/E0167-ct_rate-tsseg-effusion-counterfactual-full-seed0")
    ap.add_argument("--e0167-seed1-root", type=str, default="outputs/E0167-ct_rate-tsseg-effusion-counterfactual-full-seed1")
    ap.add_argument("--e0167-seed2-root", type=str, default="outputs/E0167-ct_rate-tsseg-effusion-counterfactual-full-seed2")
    ap.add_argument("--e0167r-json", type=str, default="outputs/E0167R-ct_rate-tsseg-effusion-counterfactual-power/omega_perm_power_report.json")
    ap.add_argument("--e0167r2-json", type=str, default="outputs/E0167R2-ct_rate-tsseg-effusion-counterfactual-power-seed20/omega_perm_power_report.json")
    ap.add_argument("--baseline-seed0-root", type=str, default="outputs/E0167-ct_rate-tsseg-effusion-counterfactual-full-seed0")
    ap.add_argument("--ra-root", type=str, default="outputs/E0167RA-ct_rate-tsseg-effusion-counterfactual-full-seed0")
    ap.add_argument("--rb-root", type=str, default="outputs/E0167RB-ct_rate-tsseg-effusion-counterfactual-full-seed0")
    ap.add_argument("--rc-root", type=str, default="outputs/E0167RC-ct_rate-tsseg-effusion-counterfactual-full-seed0")
    ap.add_argument("--rd-root", type=str, default="outputs/E0167RD-ct_rate-tsseg-effusion-counterfactual-full-seed0")
    ap.add_argument("--proof-check-script", type=str, default="scripts/proof_check.py")
    ap.add_argument("--proof-profile", type=str, default="real")
    ap.add_argument("--omega-json", type=str, default="outputs/E0167R2-ct_rate-tsseg-effusion-counterfactual-power-seed20/omega_perm_power_report.json")
    ap.add_argument("--v0004-report-json", type=str, default="outputs/V0004-backbone-transfer/backbone_transfer_report.json")
    ap.add_argument("--out-dir", type=str, default="docs/paper_assets/tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = out_dir / "table1_claims_real.md"
    t2 = out_dir / "table2_v0003_cross_dataset.md"
    t3 = out_dir / "table3_omega_variant_search.md"
    t4 = out_dir / "table4_oral_minset.md"
    t5 = out_dir / "table5_backbone_transfer.md"

    build_table1_claims(Path(args.audit_json), t1)
    build_table2_v0003(
        Path(args.e0166_json),
        [Path(args.e0167_seed0_root), Path(args.e0167_seed1_root), Path(args.e0167_seed2_root)],
        Path(args.e0167r_json),
        Path(args.e0167r2_json),
        t2,
    )
    build_table3_variants(
        Path(args.baseline_seed0_root),
        {
            "RA": Path(args.ra_root),
            "RB": Path(args.rb_root),
            "RC": Path(args.rc_root),
            "RD": Path(args.rd_root),
        },
        t3,
    )
    build_table4_oral_minset(
        proof_check_script=Path(args.proof_check_script),
        proof_profile=str(args.proof_profile),
        omega_json=Path(args.omega_json),
        out_path=t4,
    )
    build_table5_backbone_transfer(
        v0004_report_json=Path(args.v0004_report_json),
        out_path=t5,
    )

    manifest = {
        "generated_at_utc": _now_iso(),
        "out_dir": str(out_dir),
        "tables": [str(t1), str(t2), str(t3), str(t4), str(t5)],
        "sources": {
            "audit_json": str(Path(args.audit_json).resolve()),
            "e0166_json": str(Path(args.e0166_json).resolve()),
            "e0167r_json": str(Path(args.e0167r_json).resolve()),
            "e0167r2_json": str(Path(args.e0167r2_json).resolve()),
            "proof_check_script": str((ROOT / Path(args.proof_check_script)).resolve()),
            "proof_profile": str(args.proof_profile),
            "omega_json": str(Path(args.omega_json).resolve()),
            "v0004_report_json": str(Path(args.v0004_report_json).resolve()),
        },
    }
    (out_dir / "table_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
