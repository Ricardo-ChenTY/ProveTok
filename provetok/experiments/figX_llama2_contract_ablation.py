"""Fig X: LLaMA2 contract ablation (C0008).

We keep the same LLaMA2 backend and tokenization method, and ablate which parts
of the proof-carrying contract are enforced:
- free_form: plain-text generation, frames extracted post-hoc (no schema/citations)
- schema_only: JSON frames schema enforced, but no citations
- schema_citations: JSON schema + citations (default citations via score override)
- full: schema + citations + refusal (refusal can be frozen via --refusal-policy)

Primary goal: show the gains come from the contract (schema/citations/verifier/refusal),
not from "using LLaMA2" alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..eval.stats import bootstrap_mean_ci
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .run_baselines import BaselineRunConfig, run_baselines
from .utils import save_results_json


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_seed(seed0: int, tag: str) -> int:
    h = hashlib.sha1(tag.encode("utf-8")).digest()
    return int(seed0) + int.from_bytes(h[:4], "little", signed=False) % 1_000_000


def _paired_bootstrap_mean_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int,
    seed: int,
    ci: float,
    alternative: str,
) -> Dict[str, float]:
    """Paired bootstrap for mean diff (a - b) with one/two-sided p-value."""
    a_arr = np.asarray(list(a), dtype=np.float64)
    b_arr = np.asarray(list(b), dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"shape mismatch: a={a_arr.shape}, b={b_arr.shape}")
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
        raise ValueError(f"unknown alternative={alternative!r}")

    return {"mean_diff": mean_diff, "ci_low": lo, "ci_high": hi, "p_value": float(p_val)}


def _holm_bonferroni(p_values: Sequence[float]) -> List[float]:
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


def _mean_over_seeds_per_sample(per_seed: Dict[int, Dict[str, Any]], *, method: str, key: str, seeds: List[int]) -> List[float]:
    mats = []
    for s in seeds:
        mats.append(per_seed[int(s)]["raw"][str(method)][str(key)])
    arr = np.asarray(mats, dtype=np.float64)  # (S,N)
    if np.isnan(arr).any():
        keep = ~np.isnan(arr).any(axis=0)
        arr = arr[:, keep]
    if arr.shape[1] == 0:
        return []
    return arr.mean(axis=0).astype(np.float64).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LLaMA2 contract ablation across budgets (C0008).")
    ap.add_argument("--dataset-type", type=str, default="manifest", choices=["synthetic", "manifest"])
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--methods", type=str, nargs="+", default=["provetok_lesionness"], help="Tokenization method(s) to run.")
    ap.add_argument("--budgets", type=float, nargs="+", required=True, help="FLOPs total budgets (budget caps).")
    ap.add_argument("--b-gen", type=int, default=128)
    ap.add_argument("--n-verify", type=int, default=1)
    ap.add_argument("--costs-json", type=str, default="outputs/compute_costs.json")
    ap.add_argument("--selector-ratio", type=float, default=0.1)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=20_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--refusal-policy", type=str, default="", help="Optional frozen refusal_policy.json to apply to all modes.")
    ap.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])
    ap.add_argument(
        "--contract-modes",
        type=str,
        nargs="+",
        default=["free_form", "schema_only", "schema_citations", "full", "inline_citation"],
        choices=["free_form", "schema_only", "schema_citations", "full", "inline_citation"],
    )
    ap.add_argument("--citation-source", type=str, default="score_override", choices=["score_override", "llm"])
    ap.add_argument("--max-frames", type=int, default=5, help="Max frames in LLaMA2 contract modes.")
    ap.add_argument("--lesionness-weights", type=str, default="", help="Optional lesionness_head.pt for provetok_lesionness.")
    ap.add_argument("--lesionness-device", type=str, default="cpu")
    ap.add_argument("--lesionness-score-level-power", type=float, default=0.0)
    ap.add_argument("--saliency-weights", type=str, default="", help="Optional saliency weights for saliency-driven tokenizers.")
    ap.add_argument("--saliency-device", type=str, default="cpu")
    ap.add_argument(
        "--saliency-score-reduce",
        type=str,
        default="max",
        help="How to reduce voxel saliency to cell scores (max|mean|pXX).",
    )
    ap.add_argument(
        "--saliency-score-level-power",
        type=float,
        default=0.0,
        help="Multiply saliency-derived scores by (level+1)^p to favor finer cells when selecting citations.",
    )
    ap.add_argument("--no-text-metrics", action="store_true", help="Disable BLEU/ROUGE metrics (faster).")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output-dir", type=str, default="outputs/E0183-llama2-contract-ablation")
    args = ap.parse_args()

    out_root = Path(str(args.output_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    budgets = [float(b) for b in args.budgets]
    seeds = [int(s) for s in args.seeds]
    methods = [str(m) for m in args.methods]
    modes = [str(m) for m in args.contract_modes]

    per_mode_budget_seed: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    per_mode_budget_dirs: Dict[str, Dict[str, Dict[int, str]]] = {}
    pending: List[Dict[str, Any]] = []

    for mode in modes:
        per_mode_budget_seed[mode] = {}
        per_mode_budget_dirs[mode] = {}
        for b in budgets:
            b_key = f"{b:g}"
            per_mode_budget_seed[mode][b_key] = {}
            per_mode_budget_dirs[mode][b_key] = {}
            for s in seeds:
                seed_dir = out_root / f"mode_{mode}" / f"budget_{b_key.replace('.', '_')}" / f"seed_{int(s)}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                out_path = seed_dir / "baselines.json"

                if args.resume and out_path.exists():
                    try:
                        rep = _load_json(out_path)
                    except Exception:
                        rep = None
                    if isinstance(rep, dict) and rep:
                        per_mode_budget_seed[mode][b_key][int(s)] = rep
                        per_mode_budget_dirs[mode][b_key][int(s)] = str(seed_dir)
                        continue

                cfg = BaselineRunConfig(
                    dataset_type=str(args.dataset_type),
                    manifest_path=str(args.manifest),
                    split=str(args.split),
                    n_samples=int(args.n_samples),
                    seed=int(s),
                    output_dir=str(seed_dir),
                    refusal_policy_path=str(args.refusal_policy),
                    flops_total=float(b),
                    b_gen=int(args.b_gen),
                    n_verify=int(args.n_verify),
                    costs_json=str(args.costs_json),
                    selector_ratio=float(args.selector_ratio),
                    resize_shape=tuple(int(x) for x in args.resize_shape),
                    pcg_backend="llama2",
                    llama2_path=str(args.llama2_path),
                    llama2_quant=str(args.llama2_quant),
                    llama2_contract_mode=str(mode),
                    llama2_citation_source=str(args.citation_source),
                    llama2_max_frames=int(args.max_frames),
                    methods=list(methods),
                    topk_citations=3,
                    lesionness_weights=str(args.lesionness_weights),
                    lesionness_device=str(args.lesionness_device),
                    lesionness_score_level_power=float(args.lesionness_score_level_power),
                    saliency_weights=str(args.saliency_weights),
                    saliency_device=str(args.saliency_device),
                    saliency_score_reduce=str(args.saliency_score_reduce),
                    saliency_score_level_power=float(args.saliency_score_level_power),
                    compute_text_metrics=(not bool(args.no_text_metrics)),
                )
                pending.append(
                    {
                        "mode": mode,
                        "b_key": b_key,
                        "seed": int(s),
                        "seed_dir": str(seed_dir),
                        "cfg_dict": cfg.__dict__,
                        "out_path": str(out_path),
                    }
                )

    if pending:
        workers = max(1, int(args.workers))
        if workers == 1:
            for job in pending:
                rep = run_baselines(BaselineRunConfig(**job["cfg_dict"]))
                save_results_json(rep, str(job["out_path"]))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            def _run_one(cfg_dict: Dict[str, Any], out_path: str) -> None:
                rep = run_baselines(BaselineRunConfig(**cfg_dict))
                save_results_json(rep, out_path)

            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_one, job["cfg_dict"], job["out_path"]) for job in pending]
                for fut in as_completed(futs):
                    fut.result()

        for job in pending:
            rep = _load_json(Path(str(job["out_path"])))
            per_mode_budget_seed[str(job["mode"])][str(job["b_key"])][int(job["seed"])] = rep
            per_mode_budget_dirs[str(job["mode"])][str(job["b_key"])][int(job["seed"])] = str(job["seed_dir"])

    # Aggregate per mode/budget.
    requested_text_metrics = (not bool(args.no_text_metrics))
    text_metrics_enabled = False
    try:
        sample_rep = per_mode_budget_seed[modes[0]][f"{budgets[0]:g}"][seeds[0]]
        sample_raw = sample_rep.get("raw", {}).get(str(methods[0]), {}) if isinstance(sample_rep, dict) else {}
        text_metrics_enabled = requested_text_metrics and ("rougeL" in sample_raw or "bleu" in sample_raw)
    except Exception:
        text_metrics_enabled = False

    metrics = ["combined", "iou", "unsupported", "n_frames_pred_pos", "n_frames_with_citations"]
    if text_metrics_enabled:
        metrics.extend(["bleu", "rouge1", "rouge2", "rougeL"])
    agg: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {m: {} for m in modes}
    for mode in modes:
        for bidx, b in enumerate(budgets):
            b_key = f"{b:g}"
            per_seed = per_mode_budget_seed[mode][b_key]
            agg[mode][b_key] = {}
            for method in methods:
                agg[mode][b_key][method] = {}
                for k in metrics:
                    vals = _mean_over_seeds_per_sample(per_seed, method=method, key=k, seeds=seeds)
                    res = bootstrap_mean_ci(vals, n_boot=int(args.n_bootstrap), seed=_stable_seed(seeds[0], f"{mode}:{bidx}:{method}:{k}"), ci=float(args.ci))
                    agg[mode][b_key][method][k] = {"mean": float(res.mean), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}

    # Paired bootstrap: full - free_form, Holm over budgets per metric.
    compare = {"full": "free_form"}
    paired: Dict[str, Dict[str, Dict[str, Any]]] = {}
    pass_rule = {
        # Primary: combined↑ (better) across budgets (Holm).
        # Secondary gate: IoU should be non-inferior (CI low >= threshold) for all budgets.
        "metric": ["combined", "iou"],
        "alternatives": {"combined": "greater", "iou": "greater", "unsupported": "less"},
        "p_holm_threshold": 0.05,
        "min_pass_budgets": 4,
        "min_pass_budgets_by_metric": {"iou": 6},
        "required_budgets": 6,
        "iou_noninferior_ci_low": 0.0,
    }

    for m_full, m_base in compare.items():
        if m_full not in modes or m_base not in modes:
            continue
        paired_key = f"{m_full}_minus_{m_base}"
        paired_metrics = list(pass_rule["metric"])
        if "unsupported" not in paired_metrics:
            paired_metrics.append("unsupported")
        if text_metrics_enabled:
            paired_metrics.extend(["bleu", "rougeL"])
        paired[paired_key] = {k: {} for k in paired_metrics}

        for metric in paired_metrics:
            alt_map = pass_rule.get("alternatives") if isinstance(pass_rule.get("alternatives"), dict) else {}
            alt = str((alt_map or {}).get(metric, "greater"))
            p_vals = []
            tmp = []
            for bidx, b in enumerate(budgets):
                b_key = f"{b:g}"
                per_seed_full = per_mode_budget_seed[m_full][b_key]
                per_seed_base = per_mode_budget_seed[m_base][b_key]
                method = methods[0]  # primary method; keep C0008 definition simple
                a = _mean_over_seeds_per_sample(per_seed_full, method=method, key=metric, seeds=seeds)
                bb = _mean_over_seeds_per_sample(per_seed_base, method=method, key=metric, seeds=seeds)
                rec = _paired_bootstrap_mean_diff(
                    a,
                    bb,
                    n_boot=int(args.n_bootstrap),
                    seed=_stable_seed(seeds[0], f"{paired_key}:{metric}:{bidx}"),
                    ci=float(args.ci),
                    alternative=alt,
                )
                tmp.append((b_key, rec))
                p_vals.append(float(rec["p_value"]))

            p_holm = _holm_bonferroni(p_vals)
            for (b_key, rec), ph in zip(tmp, p_holm):
                paired[paired_key][metric][b_key] = {**rec, "p_value_holm": float(ph)}

    # Verdict for C0008 (on full - free_form).
    verdict: Dict[str, Any] = {"claim_id": "C0008", "proved": False, "details": {}}
    target_key = "full_minus_free_form"
    if target_key in paired:
        min_by_metric = pass_rule.get("min_pass_budgets_by_metric") if isinstance(pass_rule.get("min_pass_budgets_by_metric"), dict) else {}
        iou_ci_low_threshold = float(pass_rule.get("iou_noninferior_ci_low", 0.0))
        metric_pass_counts: Dict[str, int] = {}
        for metric in pass_rule["metric"]:
            alt_map = pass_rule.get("alternatives") if isinstance(pass_rule.get("alternatives"), dict) else {}
            alt = str((alt_map or {}).get(metric, "greater"))
            by_budget = paired[target_key].get(metric, {})
            ok = 0
            for b_key, rec in by_budget.items():
                if metric == "iou":
                    if float(rec.get("ci_low", 0.0)) >= iou_ci_low_threshold:
                        ok += 1
                else:
                    mean_diff = float(rec.get("mean_diff", 0.0))
                    passed_dir = (mean_diff > 0.0) if alt == "greater" else (mean_diff < 0.0)
                    if passed_dir and float(rec.get("p_value_holm", 1.0)) < float(pass_rule["p_holm_threshold"]):
                        ok += 1
            metric_pass_counts[metric] = int(ok)
        proved = (
            len(budgets) >= int(pass_rule["required_budgets"])
            and all(int(metric_pass_counts[m]) >= int((min_by_metric or {}).get(m, pass_rule["min_pass_budgets"])) for m in pass_rule["metric"])
        )
        verdict = {
            "claim_id": "C0008",
            "proved": bool(proved),
            "metric_pass_counts": metric_pass_counts,
            "pass_rule": pass_rule,
        }

    # Meta
    repo_root = Path(__file__).resolve().parents[2]
    data_revision = "synthetic"
    split_manifest_path = ""
    if args.dataset_type == "manifest":
        data_revision, split_manifest_path = try_manifest_revision(str(args.manifest))
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(seeds[0]),
        config={
            "dataset_type": args.dataset_type,
            "manifest_path": args.manifest,
            "split": args.split,
            "resize_shape": list(args.resize_shape),
            "methods": methods,
            "budgets": budgets,
            "b_gen": int(args.b_gen),
            "n_verify": int(args.n_verify),
            "costs_json": str(args.costs_json),
            "selector_ratio": float(args.selector_ratio),
            "n_samples": int(args.n_samples),
            "seeds": seeds,
            "n_bootstrap": int(args.n_bootstrap),
            "ci": float(args.ci),
            "llama2_path": str(args.llama2_path),
            "llama2_quant": str(args.llama2_quant),
            "contract_modes": modes,
            "citation_source": str(args.citation_source),
            "max_frames": int(args.max_frames),
            "refusal_policy": str(args.refusal_policy),
            "text_metrics_enabled": bool(text_metrics_enabled),
        },
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    report: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "modes": modes,
        "budgets": budgets,
        "methods": methods,
        "metrics": metrics,
        "agg": agg,
        "paired_bootstrap": paired,
        "verdict": verdict,
        "text_metrics_enabled": bool(text_metrics_enabled),
        "per_mode_budget_seed_dirs": per_mode_budget_dirs,
    }

    out_json = out_root / "figX_llama2_contract_ablation.json"
    save_results_json(report, str(out_json))
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
