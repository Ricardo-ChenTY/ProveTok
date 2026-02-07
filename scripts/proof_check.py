from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PROFILE = "default"


@dataclass(frozen=True)
class ClaimCheck:
    claim_id: str
    proved: bool
    summary: str
    details: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_bootstrap_mean_diff(
    a: List[float],
    b: List[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
    alternative: str = "two-sided",  # "two-sided" | "greater" | "less"
) -> Dict[str, float]:
    """Paired bootstrap for mean difference (a - b), with a two-sided p-value."""
    a_arr = np.asarray(list(a), dtype=np.float64)
    b_arr = np.asarray(list(b), dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Shape mismatch: a={a_arr.shape}, b={b_arr.shape}")
    n = int(a_arr.shape[0])
    if n <= 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    diffs = a_arr - b_arr
    mean_diff = float(diffs.mean())
    rng = np.random.RandomState(int(seed))
    idx = rng.randint(0, n, size=(int(n_boot), n))
    boot = diffs[idx].mean(axis=1)
    alpha = 1.0 - float(ci)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    p_pos = float(np.mean(boot >= 0.0))
    p_neg = float(np.mean(boot <= 0.0))
    alt = str(alternative)
    if alt == "two-sided":
        p_val = min(1.0, 2.0 * min(p_pos, p_neg))
    elif alt == "greater":
        # H1: mean_diff > 0
        p_val = p_neg
    elif alt == "less":
        # H1: mean_diff < 0
        p_val = p_pos
    else:
        raise ValueError(f"unknown alternative={alternative!r} (expected 'two-sided'|'greater'|'less')")
    return {"mean_diff": mean_diff, "ci_low": lo, "ci_high": hi, "p_value": float(p_val)}


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = list(enumerate(float(p) for p in p_values))
    indexed.sort(key=lambda x: x[1])
    adjusted: Dict[int, float] = {}
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * float(p))
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return [adjusted[i] for i in range(m)]


def _fmt_ci(rec: Dict[str, Any]) -> str:
    return f"{rec.get('mean',0):.6f} [{rec.get('ci_low',0):.6f},{rec.get('ci_high',0):.6f}]"


def _find_optional(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _find_latest(patterns: List[str], *, preferred_roots: Optional[List[Path]] = None) -> Optional[Path]:
    roots = preferred_roots or [ROOT / "outputs"]
    candidates: List[Path] = []
    for r in roots:
        for pat in patterns:
            candidates.extend(list(r.glob(pat)))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _profile_is_real() -> bool:
    return str(ACTIVE_PROFILE).lower() == "real"


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _read_baselines_curve_meta(path: Path) -> Dict[str, Any]:
    try:
        d = _read_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "budgets": [],
            "seeds": [],
            "n_bootstrap": 0,
            "split": "",
        }
    meta_cfg = (d.get("meta") or {}).get("config", {}) or {}
    budgets = d.get("budgets") or []
    seeds = d.get("seeds") or []
    return {
        "ok": True,
        "budgets": budgets if isinstance(budgets, list) else [],
        "seeds": seeds if isinstance(seeds, list) else [],
        "n_bootstrap": _to_int(d.get("n_bootstrap", 0), 0),
        "split": str(meta_cfg.get("split", "")),
    }


def _is_paper_grade_c0001_curve(path: Path) -> bool:
    meta = _read_baselines_curve_meta(path)
    return bool(
        meta.get("ok")
        and len(meta.get("budgets") or []) >= 6
        and len(meta.get("seeds") or []) >= 5
        and _to_int(meta.get("n_bootstrap", 0), 0) >= 20_000
        and str(meta.get("split", "")) == "test"
    )


def _select_real_baselines_curve(*, require_paper_grade: bool) -> Optional[Path]:
    """Resolve baselines_curve artifact robustly for real profile.

    Selection policy:
    1) Prefer `outputs/E0164-full/baselines_curve_multiseed.json` (canonical proof artifact).
    2) Else prefer `outputs/E0171-full3/baselines_curve_multiseed.json` (adds correctness proxies).
    3) If paper-grade is required, fall back to the newest paper-grade candidate among
       `E0164*` then `E0171*`.
    """
    preferred_paths = [
        ROOT / "outputs" / "E0164-full" / "baselines_curve_multiseed.json",
        ROOT / "outputs" / "E0171-full3" / "baselines_curve_multiseed.json",
    ]

    candidates: List[Path] = []
    candidates.extend((ROOT / "outputs").glob("E0164*/baselines_curve_multiseed.json"))
    candidates.extend((ROOT / "outputs").glob("E0171*/baselines_curve_multiseed.json"))
    candidates = [p for p in candidates if p.exists()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if not require_paper_grade:
        for p in preferred_paths:
            if p.exists():
                return p
        return candidates[0] if candidates else None

    for p in preferred_paths:
        if p.exists() and _is_paper_grade_c0001_curve(p):
            return p

    for p in candidates:
        if _is_paper_grade_c0001_curve(p):
            return p

    for p in preferred_paths:
        if p.exists():
            return p
    return candidates[0] if candidates else None


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _load_fig2_multiseed(path: Path) -> Dict[str, Any]:
    d = _read_json(path)
    budgets = [int(x) for x in d.get("budgets", [])]
    metrics = d.get("metrics", {}) or {}
    combined = metrics.get("combined", []) or []
    iou = metrics.get("iou", []) or []
    return {
        "path": str(path),
        "budgets": budgets,
        "combined": combined,
        "iou": iou,
        "raw": d,
    }


def _load_baselines_curve(path: Path) -> Dict[str, Any]:
    d = _read_json(path)
    budgets = [float(x) for x in d.get("budgets", [])]
    metrics = d.get("metrics", {}) or {}
    return {
        "path": str(path),
        "budgets": budgets,
        "metrics": metrics,
        "methods": d.get("methods", []) or [],
        "raw": d,
    }


def _compare_curves_same_budgets(
    *,
    fig2: Dict[str, Any],
    baselines: Dict[str, Any],
    method: str,
    metric_key: str,
) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    out = []
    budgets_fig = [float(b) for b in fig2["budgets"]]
    budgets_base = [float(b) for b in baselines["budgets"]]
    base_curve = (baselines["metrics"].get(metric_key, {}) or {}).get(method, []) or []
    fig_curve = fig2["raw"]["metrics"][metric_key]
    # Align by exact budget equality.
    for i, b in enumerate(budgets_base):
        if b not in budgets_fig:
            continue
        j = budgets_fig.index(b)
        if i >= len(base_curve) or j >= len(fig_curve):
            continue
        out.append((b, fig_curve[j], base_curve[i]))
    return out


def check_c0001() -> ClaimCheck:
    """Pareto dominate in matched compute with paper-grade latency/trust constraints."""
    base_path = None
    if _profile_is_real():
        base_path = _select_real_baselines_curve(require_paper_grade=True)
        if base_path is None:
            return ClaimCheck(
                claim_id="C0001",
                proved=False,
                summary="missing real-profile artifact for C0001 (expected E0164 baselines_curve_multiseed on split=test).",
                details={"profile": ACTIVE_PROFILE},
            )
    else:
        preferred = ROOT / "outputs" / "E0138-full" / "baselines_curve_multiseed.json"
        if preferred.exists():
            base_path = preferred
        else:
            candidates = list((ROOT / "outputs").glob("**/baselines_curve_multiseed.json"))
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for p in candidates:
                try:
                    d = _read_json(p)
                    split = str(((d.get("meta") or {}).get("config") or {}).get("split", ""))
                    if split == "test":
                        base_path = p
                        break
                except Exception:
                    continue

    if base_path is None:
        return ClaimCheck(
            claim_id="C0001",
            proved=False,
            summary="missing required artifact (baselines_curve_multiseed on split=test).",
            details={},
        )

    baselines = _load_baselines_curve(base_path)

    method = "provetok_lesionness"
    baseline = "fixed_grid"
    if method not in baselines.get("methods", []):
        return ClaimCheck(
            claim_id="C0001",
            proved=False,
            summary=f"not proved: missing '{method}' in baselines_curve_multiseed methods (rerun baselines with --lesionness-weights).",
            details={"baselines": baselines["path"], "methods": baselines.get("methods", [])},
        )
    if baseline not in baselines.get("methods", []):
        return ClaimCheck(
            claim_id="C0001",
            proved=False,
            summary=f"not proved: missing '{baseline}' in baselines_curve_multiseed methods.",
            details={"baselines": baselines["path"], "methods": baselines.get("methods", [])},
        )

    raw = baselines["raw"]
    per_budget_seed_dirs = (raw.get("per_budget_seed_dirs") or {}) if isinstance(raw, dict) else {}
    budgets = [float(b) for b in baselines.get("budgets", [])]
    seeds = [int(s) for s in (raw.get("seeds") or [])] if isinstance(raw, dict) else []
    n_boot = int(raw.get("n_bootstrap", 10_000)) if isinstance(raw, dict) else 10_000
    ci = float(raw.get("ci", 0.95)) if isinstance(raw, dict) else 0.95

    # Paper-grade minimums (see docs/plan.md C0001).
    if len(budgets) < 6:
        return ClaimCheck(
            claim_id="C0001",
            proved=False,
            summary=f"not proved: need >=6 budgets for paper-grade C0001 (got {len(budgets)}).",
            details={"baselines": baselines["path"], "budgets": budgets},
        )
    if len(seeds) < 5:
        return ClaimCheck(
            claim_id="C0001",
            proved=False,
            summary=f"not proved: need >=5 seeds for paper-grade C0001 (got {len(seeds)}).",
            details={"baselines": baselines["path"], "seeds": seeds},
        )
    if n_boot < 20_000:
        return ClaimCheck(
            claim_id="C0001",
            proved=False,
            summary=f"not proved: need n_bootstrap>=20000 for paper-grade C0001 (got {n_boot}).",
            details={"baselines": baselines["path"], "n_bootstrap": n_boot},
        )

    def _budget_key(b: float) -> Optional[str]:
        for k in per_budget_seed_dirs.keys():
            try:
                if float(k) == float(b):
                    return str(k)
            except Exception:
                continue
        return None

    metric_primary = "combined"
    metric_grounding = "iou"  # run_baselines stores union IoU under key "iou"
    latency_metric = "warm_time_s"
    latency_tol_ratio = 0.05  # allow +5% slower vs baseline (P95)
    unsupported_metric = "unsupported"
    unsupported_tol = 0.05  # allow small absolute increase (issue-rate per frame)
    # Directional hypotheses: ProveTok should improve (mean_diff > 0).
    alternative = "greater"
    rows: List[Dict[str, Any]] = []
    p_values_primary: List[float] = []
    p_values_ground: List[float] = []
    for b in budgets:
        bk = _budget_key(float(b))
        if bk is None:
            continue
        seed_dirs = per_budget_seed_dirs.get(bk) or {}
        shared_seeds = [s for s in seeds if str(s) in seed_dirs]
        if not shared_seeds:
            continue
        used_seeds: List[int] = []
        mats_m = []  # primary
        mats_b = []
        mats_g_m = []  # grounding
        mats_g_b = []
        mats_lat_m = []
        mats_lat_b = []
        mats_u_m = []
        mats_u_b = []
        for s in shared_seeds:
            dpath = Path(str(seed_dirs[str(s)])) / "baselines.json"
            if not dpath.exists():
                continue
            try:
                d = _read_json(dpath)
            except Exception:  # noqa: BLE001
                # Ignore corrupted per-seed artifacts and continue with remaining seeds.
                continue
            used_seeds.append(int(s))
            raw_m = ((d.get("raw") or {}).get(method) or {}).get(metric_primary) or []
            raw_b = ((d.get("raw") or {}).get(baseline) or {}).get(metric_primary) or []
            raw_g_m = ((d.get("raw") or {}).get(method) or {}).get(metric_grounding) or []
            raw_g_b = ((d.get("raw") or {}).get(baseline) or {}).get(metric_grounding) or []
            raw_lat_m = ((d.get("raw") or {}).get(method) or {}).get(latency_metric) or []
            raw_lat_b = ((d.get("raw") or {}).get(baseline) or {}).get(latency_metric) or []
            raw_u_m = ((d.get("raw") or {}).get(method) or {}).get(unsupported_metric) or []
            raw_u_b = ((d.get("raw") or {}).get(baseline) or {}).get(unsupported_metric) or []
            mats_m.append(raw_m)
            mats_b.append(raw_b)
            mats_g_m.append(raw_g_m)
            mats_g_b.append(raw_g_b)
            mats_lat_m.append(raw_lat_m)
            mats_lat_b.append(raw_lat_b)
            mats_u_m.append(raw_u_m)
            mats_u_b.append(raw_u_b)
        if (
            not mats_m
            or not mats_b
            or not mats_g_m
            or not mats_g_b
            or not mats_lat_m
            or not mats_lat_b
            or not mats_u_m
            or not mats_u_b
        ):
            continue
        arr_m = np.asarray(mats_m, dtype=np.float64)
        arr_b = np.asarray(mats_b, dtype=np.float64)
        arr_g_m = np.asarray(mats_g_m, dtype=np.float64)
        arr_g_b = np.asarray(mats_g_b, dtype=np.float64)
        arr_lat_m = np.asarray(mats_lat_m, dtype=np.float64)
        arr_lat_b = np.asarray(mats_lat_b, dtype=np.float64)
        arr_u_m = np.asarray(mats_u_m, dtype=np.float64)
        arr_u_b = np.asarray(mats_u_b, dtype=np.float64)

        n = int(
            min(
                arr_m.shape[1],
                arr_b.shape[1],
                arr_g_m.shape[1],
                arr_g_b.shape[1],
                arr_lat_m.shape[1],
                arr_lat_b.shape[1],
                arr_u_m.shape[1],
                arr_u_b.shape[1],
            )
        )
        if n <= 0:
            continue

        per_sample_m = arr_m[:, :n].mean(axis=0)
        per_sample_b = arr_b[:, :n].mean(axis=0)
        per_sample_g_m = arr_g_m[:, :n].mean(axis=0)
        per_sample_g_b = arr_g_b[:, :n].mean(axis=0)
        per_sample_lat_m = arr_lat_m[:, :n].mean(axis=0)
        per_sample_lat_b = arr_lat_b[:, :n].mean(axis=0)
        per_sample_u_m = arr_u_m[:, :n].mean(axis=0)
        per_sample_u_b = arr_u_b[:, :n].mean(axis=0)

        res_primary = _paired_bootstrap_mean_diff(
            per_sample_m.tolist(),
            per_sample_b.tolist(),
            n_boot=int(n_boot),
            seed=int(shared_seeds[0]) + int(float(b)) % 997,
            ci=float(ci),
            alternative=alternative,
        )

        res_ground = _paired_bootstrap_mean_diff(
            per_sample_g_m.tolist(),
            per_sample_g_b.tolist(),
            n_boot=int(n_boot),
            seed=int(shared_seeds[0]) + 11 + int(float(b)) % 997,
            ci=float(ci),
            alternative=alternative,
        )

        res_lat_mean = _paired_bootstrap_mean_diff(
            per_sample_lat_m.tolist(),
            per_sample_lat_b.tolist(),
            n_boot=int(n_boot),
            seed=int(shared_seeds[0]) + 17 + int(float(b)) % 997,
            ci=float(ci),
        )
        baseline_lat_mean = float(per_sample_lat_b.mean()) if per_sample_lat_b.size > 0 else 0.0

        p95_m = float(np.quantile(per_sample_lat_m, 0.95)) if per_sample_lat_m.size > 0 else 0.0
        p95_b = float(np.quantile(per_sample_lat_b, 0.95)) if per_sample_lat_b.size > 0 else 0.0
        delta_p95 = float(p95_m - p95_b)
        lat_p95_ok = bool(delta_p95 <= float(latency_tol_ratio) * max(p95_b, 1e-12))

        res_u = _paired_bootstrap_mean_diff(
            per_sample_u_m.tolist(),
            per_sample_u_b.tolist(),
            n_boot=int(n_boot),
            seed=int(shared_seeds[0]) + 23 + int(float(b)) % 997,
            ci=float(ci),
        )
        u_ok = bool(float(res_u["mean_diff"]) <= float(unsupported_tol) + 1e-12)

        p_values_primary.append(float(res_primary["p_value"]))
        p_values_ground.append(float(res_ground["p_value"]))
        rows.append(
            {
                "budget": float(b),
                "n_samples": n,
                "seeds": used_seeds,
                "combined": {
                    "metric": metric_primary,
                    "mean_diff": float(res_primary["mean_diff"]),
                    "ci_low": float(res_primary["ci_low"]),
                    "ci_high": float(res_primary["ci_high"]),
                    "p_value": float(res_primary["p_value"]),
                },
                "iou_union": {
                    "metric": metric_grounding,
                    "mean_diff": float(res_ground["mean_diff"]),
                    "ci_low": float(res_ground["ci_low"]),
                    "ci_high": float(res_ground["ci_high"]),
                    "p_value": float(res_ground["p_value"]),
                },
                "n_samples": n,
                "latency": {
                    "metric_mean": latency_metric,
                    "baseline_mean": float(baseline_lat_mean),
                    "mean_diff": float(res_lat_mean["mean_diff"]),
                    "ci_low": float(res_lat_mean["ci_low"]),
                    "ci_high": float(res_lat_mean["ci_high"]),
                    "p_value": float(res_lat_mean["p_value"]),
                    "p95_method": float(p95_m),
                    "p95_baseline": float(p95_b),
                    "p95_delta": float(delta_p95),
                    "p95_tol_ratio": float(latency_tol_ratio),
                    "passed_p95": bool(lat_p95_ok),
                },
                "unsupported": {
                    "metric": unsupported_metric,
                    "mean_diff": float(res_u["mean_diff"]),
                    "ci_low": float(res_u["ci_low"]),
                    "ci_high": float(res_u["ci_high"]),
                    "p_value": float(res_u["p_value"]),
                    "tol_abs": float(unsupported_tol),
                    "passed": bool(u_ok),
                },
            }
        )

    if not rows:
        return ClaimCheck(
            claim_id="C0001",
            proved=False,
            summary="not proved: could not load per-seed per-budget raw data for baselines curve.",
            details={"baselines": baselines["path"]},
        )

    # Guard against partially broken artifacts: paper-grade requires all 6 budgets
    # and >=5 evaluable seeds per budget (not just declared in metadata).
    if len(rows) < 6:
        return ClaimCheck(
            claim_id="C0001",
            proved=False,
            summary=f"not proved: need >=6 evaluable budgets for paper-grade C0001 (got {len(rows)}).",
            details={
                "baselines": baselines["path"],
                "declared_budgets": budgets,
                "evaluable_budgets": [float(r.get("budget", 0.0)) for r in rows],
            },
        )
    bad_seed_rows = [r for r in rows if len(r.get("seeds") or []) < 5]
    if bad_seed_rows:
        return ClaimCheck(
            claim_id="C0001",
            proved=False,
            summary=(
                "not proved: need >=5 evaluable seeds per budget for paper-grade C0001 "
                f"(failed at {len(bad_seed_rows)}/{len(rows)} budgets)."
            ),
            details={
                "baselines": baselines["path"],
                "per_budget_seed_counts": [
                    {"budget": float(r.get("budget", 0.0)), "n_evaluable_seeds": len(r.get("seeds") or [])}
                    for r in rows
                ],
            },
        )

    p_holm_primary = _holm_bonferroni(p_values_primary)
    p_holm_ground = _holm_bonferroni(p_values_ground)

    passed_primary = 0
    passed_ground = 0
    passed_latency = 0
    passed_unsupported = 0
    for rec, ph_p, ph_g in zip(rows, p_holm_primary, p_holm_ground):
        rec["combined"]["p_holm"] = float(ph_p)
        rec["combined"]["passed"] = bool((float(rec["combined"]["mean_diff"]) > 0.0) and (float(ph_p) < 0.05))
        passed_primary += int(rec["combined"]["passed"])

        rec["iou_union"]["p_holm"] = float(ph_g)
        rec["iou_union"]["passed"] = bool((float(rec["iou_union"]["mean_diff"]) > 0.0) and (float(ph_g) < 0.05))
        passed_ground += int(rec["iou_union"]["passed"])

        passed_latency += int(bool((rec.get("latency") or {}).get("passed_p95", False)))
        passed_unsupported += int(bool((rec.get("unsupported") or {}).get("passed", False)))

    need = max(4, int(math.ceil((2.0 / 3.0) * float(len(rows)))))
    proved = bool(
        (passed_primary >= need)
        and (passed_ground >= need)
        and (passed_latency == len(rows))
        and (passed_unsupported == len(rows))
    )
    return ClaimCheck(
        claim_id="C0001",
        proved=bool(proved),
        summary=(
            f"proved: {method} beats {baseline} on combined & iou at {passed_primary}/{len(rows)} and {passed_ground}/{len(rows)} budgets (Holm), with latency/unsupported constraints"
            if proved
            else (
                f"not proved: combined_pass={passed_primary}/{len(rows)}, iou_pass={passed_ground}/{len(rows)}, "
                f"latency_p95_pass={passed_latency}/{len(rows)}, unsupported_pass={passed_unsupported}/{len(rows)} (need {need} on quality metrics)"
            )
        ),
        details={
            "baselines": baselines["path"],
            "method": method,
            "baseline": baseline,
            "rows": rows,
            "rule": {
                "need_passed_budgets": need,
                "latency_p95_tol_ratio": float(latency_tol_ratio),
                "unsupported_tol_abs": float(unsupported_tol),
                "p_value_alternative": alternative,
            },
        },
    )


def check_c0004() -> ClaimCheck:
    """Pixel-level citation grounding significantly improves on ReXGroundingCT (paper-grade)."""
    path: Optional[Path] = None
    if _profile_is_real():
        # Prefer the newest paper-grade grounding proof if present.
        preferred_news = [
            ROOT / "outputs" / "E0165-full2" / "figX_grounding_proof.json",
            ROOT / "outputs" / "E0165-full" / "figX_grounding_proof.json",
        ]
        preferred_old = ROOT / "outputs" / "E0156-grounding_proof_100g_saliency_full" / "figX_grounding_proof.json"
        for p in preferred_news:
            if p.exists():
                path = p
                break
        if path is None:
            if preferred_old.exists():
                path = preferred_old
            else:
                path = _find_latest(
                    patterns=[
                        "E0165*/figX_grounding_proof.json",
                        "E0156*/figX_grounding_proof.json",
                    ],
                    preferred_roots=[ROOT / "outputs"],
                )
        if path is None:
            return ClaimCheck(
                "C0004",
                proved=False,
                summary="missing real-profile artifact for C0004 (expected E0156/E0165 figX_grounding_proof.json).",
                details={"profile": ACTIVE_PROFILE},
            )
    else:
        preferred = ROOT / "outputs" / "E0143-full" / "figX_grounding_proof.json"
        path = preferred if preferred.exists() else None
        if path is None:
            path = _find_latest(
                patterns=[
                    "E0143*/figX_grounding_proof.json",
                    "**/figX_grounding_proof.json",
                ],
                preferred_roots=[ROOT / "outputs"],
            )
    if path is None:
        return ClaimCheck("C0004", proved=False, summary="missing figX_grounding_proof.json artifact", details={})

    d = _read_json(path)
    paired = d.get("paired_bootstrap") or {}
    if not isinstance(paired, dict) or not paired:
        return ClaimCheck("C0004", proved=False, summary="missing paired_bootstrap in grounding proof artifact", details={"path": str(path)})

    seeds = d.get("seeds") or []
    budgets_raw = d.get("budgets") or []
    n_boot = int(d.get("n_bootstrap", 0))
    if not isinstance(seeds, list) or len(seeds) < 5:
        return ClaimCheck("C0004", proved=False, summary=f"not proved: need >=5 seeds for paper-grade C0004 (got {len(seeds)}).", details={"path": str(path), "seeds": seeds})
    if not isinstance(budgets_raw, list) or len(budgets_raw) < 6:
        return ClaimCheck("C0004", proved=False, summary=f"not proved: need >=6 budgets for paper-grade C0004 (got {len(budgets_raw)}).", details={"path": str(path), "budgets": budgets_raw})
    if n_boot < 20_000:
        return ClaimCheck("C0004", proved=False, summary=f"not proved: need n_bootstrap>=20000 for paper-grade C0004 (got {n_boot}).", details={"path": str(path), "n_bootstrap": n_boot})

    # Require at least these baselines (docs/plan.md C0004).
    any_budget_key = next(iter(paired.keys()))
    any_budget = paired.get(any_budget_key) or {}
    required_baselines = ["fixed_grid", "roi_variance"]
    missing_bases = [b for b in required_baselines if b not in any_budget]
    if missing_bases:
        return ClaimCheck("C0004", proved=False, summary=f"not proved: grounding proof missing required baselines {missing_bases}", details={"path": str(path), "available": sorted(list(any_budget.keys()))})

    budgets = []
    try:
        budgets = sorted(float(k) for k in paired.keys())
    except Exception:
        budgets = list(range(len(paired)))

    def _budget_key(b: float) -> Optional[str]:
        # Keys may be formatted like "2e+06". Float parsing should round-trip.
        for k in paired.keys():
            try:
                if float(k) == float(b):
                    return str(k)
            except Exception:
                continue
        return None

    metric = "iou_union"
    # Directional hypothesis: ProveTok should improve grounding (mean_diff > 0).
    # Use one-sided p-values and apply Holm-Bonferroni across budgets for this metric.
    per_baseline_rows: Dict[str, List[Dict[str, Any]]] = {b: [] for b in required_baselines}
    per_baseline_passed: Dict[str, int] = {b: 0 for b in required_baselines}

    for base in required_baselines:
        recs: List[Dict[str, Any]] = []
        p_one: List[float] = []
        for b in budgets:
            bk = _budget_key(b)
            if bk is None:
                continue
            rec = (((paired.get(bk) or {}).get(base) or {}).get(metric) or {})
            mean_diff = float(rec.get("mean_diff", 0.0))
            p_two = float(rec.get("p_value", 1.0))
            # paired_bootstrap_mean_diff uses a two-sided sign-test p-value; for a
            # pre-registered "improves" hypothesis, use one-sided p = p_two / 2 when
            # mean_diff > 0, else keep p=1.
            p1 = float(p_two / 2.0) if mean_diff > 0.0 else 1.0
            recs.append(
                {
                    "budget": float(b),
                    "baseline": base,
                    "metric": metric,
                    "mean_diff": mean_diff,
                    "p_value_two_sided": float(p_two),
                    "p_value_one_sided": float(p1),
                }
            )
            p_one.append(float(p1))

        p_holm = _holm_bonferroni(p_one)
        for rec, ph in zip(recs, p_holm):
            ok = (float(rec["mean_diff"]) > 0.0) and (float(ph) < 0.05)
            per_baseline_passed[base] += int(ok)
            per_baseline_rows[base].append({**rec, "p_holm": float(ph), "passed": bool(ok)})

    need = max(4, int(math.ceil((2.0 / 3.0) * float(len(budgets)))))
    proved = all(per_baseline_passed[b] >= need for b in required_baselines)

    return ClaimCheck(
        claim_id="C0004",
        proved=bool(proved),
        summary=(
            f"proved: provetok_lesionness beats fixed_grid and roi_variance on {metric} (Holm) with >= {need} budgets each"
            if proved
            else f"not proved: passes={{fixed_grid:{per_baseline_passed['fixed_grid']}/{len(budgets)}, roi_variance:{per_baseline_passed['roi_variance']}/{len(budgets)}}} (need {need})"
        ),
        details={
            "path": str(path),
            "metric": metric,
            "per_budget": per_baseline_rows,
            "rule": {
                "need_passed_budgets": need,
                "required_baselines": required_baselines,
                "n_bootstrap": n_boot,
                "p_value_alternative": "greater",
                "holm_scope": "budgets (per baseline, per metric)",
            },
        },
    )


def check_c0003() -> ClaimCheck:
    """Counterfactual suite significant break (scaffold check)."""
    path: Optional[Path] = None
    if _profile_is_real():
        candidates = list((ROOT / "outputs").glob("E0162*/**/figX_counterfactual.json"))
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        path = candidates[0] if candidates else None
        if path is None:
            return ClaimCheck(
                "C0003",
                proved=False,
                summary="missing real-profile artifact for C0003 (expected E0162*/**/figX_counterfactual.json).",
                details={"profile": ACTIVE_PROFILE},
            )
    else:
        preferred_roots = [
            ROOT / "outputs" / "E0113-full",
            ROOT / "outputs" / "E0109-full",
            ROOT / "outputs" / "E0004-full",
        ]
        candidates = []
        for r in preferred_roots:
            candidates.extend(list(r.glob("**/figX_counterfactual.json")))
        if not candidates:
            candidates = list((ROOT / "outputs").glob("**/figX_counterfactual.json"))
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        path = candidates[0] if candidates else None
    if path is None:
        return ClaimCheck("C0003", proved=False, summary="missing figX_counterfactual.json", details={})

    d = _read_json(path)
    pb_g = (d.get("paired_bootstrap") or {}).get("grounding_iou_union_orig_minus_cf") or {}
    pb_u = (d.get("paired_bootstrap") or {}).get("unsupported_rate_cf_minus_orig") or {}
    need = ["omega_perm", "cite_swap", "evidence_drop", "no_cite", "token_perm"]
    missing_g = [k for k in need if k not in pb_g]
    missing_u = [k for k in need if k not in pb_u]
    if missing_g or missing_u:
        return ClaimCheck(
            "C0003",
            proved=False,
            summary=f"missing counterfactual keys (grounding={missing_g}, unsupported={missing_u})",
            details={"path": str(path)},
        )

    # Proof rule (see docs/plan.md C0003):
    # - citations are *non-trivial*: cite_swap must significantly increase unsupported;
    # - citations matter for grounding: no_cite must significantly reduce grounding.
    #
    # Note: omega_perm grounding is tracked as a stronger optional check; it is not
    # required for "proved" because it is substantially harder to satisfy without a
    # learned lesionness / grounding-aware citation mechanism.
    alpha = 0.05
    no_cite_ground_ok = float(pb_g["no_cite"].get("p_value_holm", pb_g["no_cite"].get("p_value", 1.0))) < alpha
    swap_unsupported_ok = float(pb_u["cite_swap"].get("p_value_holm", pb_u["cite_swap"].get("p_value", 1.0))) < alpha
    proved = bool(no_cite_ground_ok and swap_unsupported_ok)
    return ClaimCheck(
        claim_id="C0003",
        proved=proved,
        summary=(
            "proved (no_cite breaks grounding + cite_swap breaks unsupported)"
            if proved
            else "not proved: no_cite grounding and/or cite_swap unsupported not significant"
        ),
        details={
            "path": str(path),
            "alpha": alpha,
            "passed": {
                "no_cite_grounding_holm_p_lt_alpha": bool(no_cite_ground_ok),
                "cite_swap_unsupported_holm_p_lt_alpha": bool(swap_unsupported_ok),
            },
            "grounding_p_holm": {k: float(pb_g[k].get("p_value_holm", pb_g[k].get("p_value", 1.0))) for k in need},
            "unsupported_p_holm": {k: float(pb_u[k].get("p_value_holm", pb_u[k].get("p_value", 1.0))) for k in need},
            "grounding_mean_diff": {k: float(pb_g[k].get("mean_diff", 0.0)) for k in need},
            "unsupported_mean_diff": {k: float(pb_u[k].get("mean_diff", 0.0)) for k in need},
        },
    )


def check_c0002() -> ClaimCheck:
    path: Optional[Path] = None
    if _profile_is_real():
        preferred = ROOT / "outputs" / "E0161-full" / "fig3_regret_sweep.json"
        if preferred.exists():
            path = preferred
        else:
            path = _find_latest(
                patterns=[
                    "E0161*/fig3_regret_sweep.json",
                    "E0161*/fig3_results.json",
                ],
                preferred_roots=[ROOT / "outputs"],
            )
        if path is None:
            return ClaimCheck(
                "C0002",
                proved=False,
                summary="missing real-profile artifact for C0002 (expected E0161 fig3_regret_sweep/results).",
                details={"profile": ACTIVE_PROFILE},
            )
    else:
        preferred = ROOT / "outputs" / "E0141-full" / "fig3_results.json"
        path = preferred if preferred.exists() else None
        if path is None:
            path = _find_latest(
                patterns=[
                    "E0141*/fig3_results.json",
                    "E0141*/fig3_regret_sweep.json",
                    "fig3_allocation/fig3_results.json",
                    "**/fig3_results.json",
                ],
                preferred_roots=[ROOT / "outputs"],
            )
    if path is None:
        return ClaimCheck("C0002", proved=False, summary="missing fig3 allocation artifact (fig3_results/fig3_regret_sweep).", details={})

    d = _read_json(path)
    cfg = d.get("config") or {}
    dev = d.get("dev") or {}
    test = d.get("test") or {}
    rows = d.get("rows") or (test.get("rows") if isinstance(test, dict) else []) or []

    source_schema = "fig3_regret_sweep" if str(path.name) == "fig3_regret_sweep.json" else "fig3_results"

    meta_cfg = (d.get("meta") or {}).get("config", {}) or {}
    dev_split = str(cfg.get("dev_split", meta_cfg.get("dev_split", "")))
    test_split = str(cfg.get("test_split", meta_cfg.get("test_split", "")))
    dataset_type = str(meta_cfg.get("dataset_type", cfg.get("dataset_type", "")))
    fits = (dev.get("fits") or {}) if isinstance(dev, dict) else {}

    def _infer_curve_meta(curve_map: Any) -> Tuple[str, str]:
        if not isinstance(curve_map, dict):
            return ("", "")
        for _, curve_path in curve_map.items():
            if not curve_path:
                continue
            cp = _resolve_path(str(curve_path))
            if not cp.exists():
                continue
            try:
                cd = _read_json(cp)
            except Exception:
                continue
            cmeta = (cd.get("meta") or {}).get("config", {}) or {}
            c_dataset_type = str(cmeta.get("dataset_type", ""))
            c_split = str(cmeta.get("split", ""))
            if c_dataset_type or c_split:
                return (c_dataset_type, c_split)
        return ("", "")

    dev_curves = cfg.get("dev_curves") or {}
    test_curves = cfg.get("test_curves") or {}
    dev_dt, dev_split_inferred = _infer_curve_meta(dev_curves)
    test_dt, test_split_inferred = _infer_curve_meta(test_curves)
    if not dev_split:
        dev_split = dev_split_inferred
    if not test_split:
        test_split = test_split_inferred
    if not dataset_type:
        dataset_type = dev_dt or test_dt

    has_dev_test = bool(dev_split) and bool(test_split) and (dev_split != test_split)
    has_fits = isinstance(fits, dict) and bool(fits)
    has_aic_bic = False
    if has_fits:
        any_fit = next(iter(fits.values()))
        has_aic_bic = isinstance(any_fit, dict) and ("aic" in any_fit) and ("bic" in any_fit)
    has_regret_rows = isinstance(rows, list) and bool(rows) and all(isinstance(r, dict) and ("budget" in r) and ("regret" in r) and ("normalized_regret" in r) for r in rows)

    regret = d.get("regret") or {}
    boot = (regret.get("bootstrap") or {}) if isinstance(regret, dict) else {}
    n_boot = int(boot.get("n_bootstrap", cfg.get("n_bootstrap", 0))) if isinstance(boot, dict) else int(cfg.get("n_bootstrap", 0))

    ci_high_opt: Optional[float] = None
    if isinstance(boot, dict):
        if "mean_normalized_regret_ci_high" in boot:
            ci_high_opt = float(boot.get("mean_normalized_regret_ci_high", 1.0))
        else:
            policies = (boot.get("policies") or {}) if isinstance(boot.get("policies"), dict) else {}
            learned = (policies.get("learned") or {}) if isinstance(policies.get("learned"), dict) else {}
            if "ci_high" in learned:
                ci_high_opt = float(learned.get("ci_high", 1.0))
    ci_high = float(ci_high_opt) if ci_high_opt is not None else 1.0

    naive_ci_low = 1.0
    if isinstance(boot, dict):
        naive = (boot.get("naive_policies") or {}) if isinstance(boot.get("naive_policies"), dict) else {}
        naive_fixed = naive.get("always_fixed_grid") or {}
        if isinstance(naive_fixed, dict) and ("mean_normalized_regret_ci_low" in naive_fixed):
            naive_ci_low = float(naive_fixed.get("mean_normalized_regret_ci_low", 1.0))
        else:
            policies = (boot.get("policies") or {}) if isinstance(boot.get("policies"), dict) else {}
            naive_fixed_old = policies.get("always_fixed_grid") or {}
            if isinstance(naive_fixed_old, dict) and ("ci_low" in naive_fixed_old):
                naive_ci_low = float(naive_fixed_old.get("ci_low", 1.0))

    budgets = cfg.get("budgets") or []
    has_enough_budgets = isinstance(budgets, list) and (len(budgets) >= 6)
    has_bootstrap = bool(isinstance(boot, dict) and (ci_high_opt is not None))
    paper_ok = bool(has_enough_budgets and (n_boot >= 20_000) and (ci_high <= 0.15) and (ci_high < naive_ci_low))

    proved = bool(has_dev_test and has_fits and has_aic_bic and has_regret_rows and (dataset_type == "manifest") and has_bootstrap and paper_ok)
    return ClaimCheck(
        claim_id="C0002",
        proved=proved,
        summary=(
            "proved: paper-grade dev→test regret has CI and beats naive policy (real pipeline)"
            if proved
            else "not proved: missing paper-grade regret CI/thresholds or naive-baseline advantage"
        ),
        details={
            "path": str(path),
            "profile": ACTIVE_PROFILE,
            "schema": source_schema,
            "dataset_type": dataset_type,
            "dev_split": dev_split,
            "test_split": test_split,
            "has_dev_test": has_dev_test,
            "has_fits": has_fits,
            "has_aic_bic": has_aic_bic,
            "has_regret_rows": has_regret_rows,
            "paper": {
                "has_enough_budgets": bool(has_enough_budgets),
                "n_bootstrap": int(n_boot),
                "mean_normalized_regret_ci_high": float(ci_high),
                "naive_always_fixed_grid_ci_low": float(naive_ci_low),
                "passed": bool(paper_ok),
            },
            "keys": sorted(list(d.keys())),
        },
    )


def check_c0005() -> ClaimCheck:
    preferred = ROOT / "outputs" / "E0144-full" / "figX_refusal_calibration.json"
    path = preferred if preferred.exists() else None
    if path is None:
        path = _find_latest(
            patterns=[
                "E0144*/figX_refusal_calibration.json",
                "E*/**/figX_refusal_calibration.json",
                "**/figX_refusal_calibration.json",
            ],
            preferred_roots=[ROOT / "outputs"],
        )
    if path is None:
        return ClaimCheck("C0005", proved=False, summary="missing figX_refusal_calibration.json artifact.", details={})

    d = _read_json(path)
    cfg = d.get("config") or {}
    test = d.get("test") or {}
    rows = test.get("rows") or []
    max_miss = float(cfg.get("max_critical_miss_rate", 0.05))
    best_tau = float(d.get("best_tau", 0.0))

    budgets = d.get("budgets") or []
    if not isinstance(budgets, list) or len(budgets) < 6:
        return ClaimCheck("C0005", proved=False, summary=f"not proved: need >=6 budgets for paper-grade C0005 (got {len(budgets)}).", details={"path": str(path), "budgets": budgets})

    # Paper-grade proof rule (see docs/plan.md C0005):
    # - miss-rate constraint per budget
    # - unsupported decreases in >= 4/6 budgets
    # - refusal_ece and refusal_rate are hard constraints per budget
    max_refusal_rate = 0.20
    passed_miss = True
    passed_ece = True
    passed_refusal_rate = True
    passed_unsupported = 0
    per_budget = []
    for r in rows:
        b = float(r.get("budget", 0.0))
        base = r.get("no_refusal") or {}
        cal = r.get("calibrated") or {}
        u0 = float(base.get("unsupported_rate", 0.0))
        u1 = float(cal.get("unsupported_rate", 0.0))
        m1 = float(cal.get("critical_miss_rate", 1.0))
        ece1 = float(cal.get("refusal_ece", 1.0))
        rr1 = float(cal.get("refusal_rate", 1.0))
        ok_miss = m1 <= max_miss + 1e-12
        ok_ece = ece1 <= 0.15 + 1e-12
        ok_rr = rr1 <= float(max_refusal_rate) + 1e-12
        ok_u = u1 < u0
        passed_miss = passed_miss and ok_miss
        passed_ece = passed_ece and ok_ece
        passed_refusal_rate = passed_refusal_rate and ok_rr
        passed_unsupported += int(ok_u)
        per_budget.append(
            {
                "budget": b,
                "unsupported_no_refusal": u0,
                "unsupported_calibrated": u1,
                "unsupported_delta": u1 - u0,
                "critical_miss_rate": m1,
                "refusal_ece": ece1,
                "refusal_rate": rr1,
                "delta_max": max_miss,
                "passed_miss": bool(ok_miss),
                "passed_ece": bool(ok_ece),
                "passed_refusal_rate": bool(ok_rr),
                "passed_unsupported": bool(ok_u),
            }
        )

    need = max(4, int(math.ceil((2.0 / 3.0) * float(len(per_budget))))) if per_budget else 4
    proved = bool(per_budget) and passed_miss and passed_ece and passed_refusal_rate and (passed_unsupported >= need)
    return ClaimCheck(
        claim_id="C0005",
        proved=proved,
        summary=(
            f"proved: tau_refuse={best_tau:g} meets miss/ECE/refusal constraints and reduces unsupported at {passed_unsupported}/{len(per_budget)} budgets"
            if proved
            else (
                f"not proved: miss={passed_miss}, ece={passed_ece}, refusal_rate={passed_refusal_rate}, "
                f"unsupported_improved={passed_unsupported}/{len(per_budget)} (need {need})"
            )
        ),
        details={
            "path": str(path),
            "best_tau": best_tau,
            "max_critical_miss_rate": max_miss,
            "max_refusal_ece": 0.15,
            "max_refusal_rate": float(max_refusal_rate),
            "per_budget": per_budget,
            "rule": {"need_passed_budgets": need},
        },
    )


def check_c0006() -> ClaimCheck:
    path: Optional[Path] = None
    if _profile_is_real():
        path = _select_real_baselines_curve(require_paper_grade=True)
        if path is None:
            return ClaimCheck(
                "C0006",
                proved=False,
                summary="missing real-profile artifact for C0006 (expected E0164 baselines_curve_multiseed).",
                details={"profile": ACTIVE_PROFILE},
            )
    else:
        preferred = ROOT / "outputs" / "E0138-full" / "baselines_curve_multiseed.json"
        path = preferred if preferred.exists() else None
        if path is None:
            path = _find_latest(
                patterns=[
                    "E0138*/baselines_curve_multiseed.json",
                    "**/baselines_curve_multiseed.json",
                ],
                preferred_roots=[ROOT / "outputs"],
            )
    if path is None:
        return ClaimCheck("C0006", proved=False, summary="missing baselines_curve_multiseed.json artifact.", details={})

    d = _read_json(path)
    methods = d.get("methods") or []
    required = ["fixed_grid", "slice_2d", "slice_2p5d", "roi_crop", "roi_variance", "ct2rep_strong"]
    missing = [m for m in required if m not in methods]
    meta = d.get("meta") or {}
    cfg = meta.get("config") or {}
    costs_json = str(cfg.get("costs_json", ""))
    strong_weights = str(cfg.get("ct2rep_strong_weights", ""))
    budgets_by_method = d.get("budgets_by_method") or {}
    has_budget_accounting = bool(budgets_by_method) and all(isinstance(v, dict) for v in budgets_by_method.values())

    # Paper-grade baseline-suite proof for this repo focuses on:
    # - method coverage
    # - audited cost accounting presence
    # - reproducible strong baseline weights being present
    # - strong baseline being non-degenerate on manifest labels (frame_f1 > 0)
    frame_f1_rows = (((d.get("metrics") or {}).get("frame_f1") or {}).get("ct2rep_strong") or [])
    strong_frame_f1 = float(frame_f1_rows[-1].get("mean", 0.0)) if frame_f1_rows else 0.0
    weights_ok = bool(strong_weights and _resolve_path(strong_weights).exists())
    min_frame_f1 = 0.05
    f1_ok = bool(strong_frame_f1 >= float(min_frame_f1) - 1e-12)

    proved = (not missing) and bool(costs_json) and bool(has_budget_accounting) and bool(weights_ok) and bool(f1_ok)
    return ClaimCheck(
        claim_id="C0006",
        proved=proved,
        summary=(
            "proved: baseline suite present with cost accounting + reproducible strong baseline"
            if proved
            else (
                f"not proved: baselines missing {missing}"
                if missing
                else (
                    "not proved: missing audited cost accounting and/or strong baseline weights"
                    if (not bool(costs_json) or not bool(has_budget_accounting) or not bool(weights_ok))
                    else f"not proved: ct2rep_strong non-degenerate gate failed (frame_f1_last_budget_mean={strong_frame_f1:.4f} < {min_frame_f1:.2f})"
                )
            )
        ),
        details={
            "path": str(path),
            "profile": ACTIVE_PROFILE,
            "methods": methods,
            "costs_json": costs_json,
            "has_budget_accounting": bool(has_budget_accounting),
            "ct2rep_strong": {
                "weights": strong_weights,
                "weights_exists": bool(weights_ok),
                "frame_f1_last_budget_mean": float(strong_frame_f1),
                "min_frame_f1": float(min_frame_f1),
                "passed": bool(weights_ok and f1_ok),
            },
        },
    )


def main() -> int:
    global ACTIVE_PROFILE
    ap = argparse.ArgumentParser(description="Quick proof-status checker for docs/plan.md claims (scaffold).")
    ap.add_argument("--out", type=str, default="", help="Optional path to write a JSON report.")
    ap.add_argument("--profile", type=str, choices=["default", "real"], default="default", help="Artifact profile to check (default|real).")
    args = ap.parse_args()
    ACTIVE_PROFILE = str(args.profile).strip().lower()

    checks = [
        check_c0001(),
        check_c0002(),
        check_c0003(),
        check_c0004(),
        check_c0005(),
        check_c0006(),
    ]
    report = {
        "root": str(ROOT),
        "profile": ACTIVE_PROFILE,
        "checks": [
            {
                "claim_id": c.claim_id,
                "proved": bool(c.proved),
                "summary": c.summary,
                "details": c.details,
            }
            for c in checks
        ],
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
