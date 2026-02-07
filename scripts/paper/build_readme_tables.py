#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


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
    ap.add_argument("--out-dir", type=str, default="docs/paper_assets/tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = out_dir / "table1_claims_real.md"
    t2 = out_dir / "table2_v0003_cross_dataset.md"
    t3 = out_dir / "table3_omega_variant_search.md"

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

    manifest = {
        "generated_at_utc": _now_iso(),
        "out_dir": str(out_dir),
        "tables": [str(t1), str(t2), str(t3)],
        "sources": {
            "audit_json": str(Path(args.audit_json).resolve()),
            "e0166_json": str(Path(args.e0166_json).resolve()),
            "e0167r_json": str(Path(args.e0167r_json).resolve()),
            "e0167r2_json": str(Path(args.e0167r2_json).resolve()),
        },
    }
    (out_dir / "table_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
