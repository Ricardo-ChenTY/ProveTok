"""Fig 2: Scaling Law multi-seed runner (paper-grade scaffold).

Runs `fig2_scaling_law` for multiple seeds and aggregates per-budget metrics with
bootstrap CIs (hierarchical: average across seeds per-sample, then bootstrap over
samples).

Outputs:
- <output_dir>/seed_<seed>/fig2_raw_data.json (per seed)
- <output_dir>/seed_<seed>/fig2_scaling_fit.json (per seed)
- <output_dir>/fig2_multiseed.json (aggregated summary + CI)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..eval.scaling import fit_scaling_law, compute_diminishing_returns_point
from ..eval.stats import bootstrap_mean_ci
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .fig2_scaling_law import ScalingExperimentConfig, run_scaling_experiment
from .utils import save_results_json


def _mean_and_ci_from_seed_sample_matrix(
    x: np.ndarray, *, n_boot: int, seed: int, ci: float
) -> Dict[str, float]:
    """Hierarchical mean CI: mean over seeds per sample, bootstrap across samples."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (S,N), got shape={x.shape}")
    if x.shape[1] == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    per_sample = x.mean(axis=0)
    res = bootstrap_mean_ci(per_sample.tolist(), n_boot=n_boot, seed=seed, ci=ci)
    return {"mean": float(res.mean), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}


def _p95_from_seed_sample_matrix(x: np.ndarray) -> float:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (S,N), got shape={x.shape}")
    if x.shape[1] == 0:
        return 0.0
    per_sample = x.mean(axis=0)
    return float(np.quantile(per_sample.astype(np.float64), 0.95))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Fig2 scaling experiment for multiple seeds + bootstrap CI.")
    ap.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest"])
    ap.add_argument("--manifest", type=str, default="", help="Manifest path when dataset-type=manifest")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--pcg", type=str, default="toy", choices=["toy", "llama2"])
    ap.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])
    ap.add_argument("--pcg-refresh-period", type=int, default=1)
    ap.add_argument("--encoder", type=str, default="toy", choices=["toy", "cnn3d"])
    ap.add_argument("--encoder-device", type=str, default="cuda")
    ap.add_argument("--no-evidence-head", action="store_true", help="Disable EvidenceHead and use the simple allocator.")
    ap.add_argument("--require-full-budget", action="store_true", help="Avoid early-stop on no-issues/epsilon; spend budget when possible.")
    ap.add_argument("--lambda-uncertainty", type=float, default=0.3, help="Uncertainty weight in EvidenceHead Δ(c).")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--budget-mode", type=str, default="tokens", choices=["tokens", "flops"])
    ap.add_argument("--costs-json", type=str, default="")
    ap.add_argument("--b-gen", type=int, default=128)
    ap.add_argument("--n-verify", type=int, default=1)
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--output-dir", type=str, default="./outputs/fig2_multiseed")
    ap.add_argument("--no-plot", action="store_true", help="(kept for parity; plotting not implemented here)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    per_seed_raw: Dict[int, Dict[str, Any]] = {}
    per_seed_fit: Dict[int, Dict[str, Any]] = {}
    per_seed_dirs: Dict[int, str] = {}

    budgets = [int(b) for b in args.budgets]

    shared_generator_fn = None
    if str(args.pcg) == "llama2":
        # Avoid re-loading LLaMA-2 weights for every seed; generation is deterministic
        # (temperature=0), and the model is large enough that repeated loads dominate
        # the runtime for smoke/oral runs.
        from ..pcg.llama2_pcg import Llama2PCG, Llama2PCGConfig

        pcg = Llama2PCG(
            Llama2PCGConfig(
                model_path=str(args.llama2_path),
                device="cuda",
                quantization=str(args.llama2_quant),
                # Keep output cap aligned with compute accounting (b_gen).
                max_new_tokens=max(128, int(args.b_gen)),
                temperature=0.0,
                topk_citations=3,
            )
        )

        def shared_generator_fn(tokens):  # noqa: ANN001
            return pcg(tokens)

    for seed in args.seeds:
        out_dir = os.path.join(args.output_dir, f"seed_{seed}")
        per_seed_dirs[int(seed)] = out_dir

        cfg = ScalingExperimentConfig(
            budgets=budgets,
            n_samples=int(args.n_samples),
            max_steps=int(args.max_steps),
            seed=int(seed),
            output_dir=out_dir,
            dataset_type=str(args.dataset_type),
            manifest_path=str(args.manifest),
            split=str(args.split),
            resize_shape=tuple(int(x) for x in args.resize_shape),
            pcg_backend=str(args.pcg),
            llama2_path=str(args.llama2_path),
            llama2_quant=str(args.llama2_quant),
            pcg_refresh_period=int(args.pcg_refresh_period),
            encoder_backend=str(args.encoder),
            encoder_device=str(args.encoder_device),
            use_evidence_head=not bool(args.no_evidence_head),
            require_full_budget=bool(args.require_full_budget),
            lambda_uncertainty=float(args.lambda_uncertainty),
            budget_mode=str(args.budget_mode),
            costs_json=str(args.costs_json),
            b_gen=int(args.b_gen),
            n_verify=int(args.n_verify),
        )

        _ = run_scaling_experiment(config=cfg, generator_fn=shared_generator_fn, verbose=True)

        raw_path = os.path.join(out_dir, "fig2_raw_data.json")
        fit_path = os.path.join(out_dir, "fig2_scaling_fit.json")
        per_seed_raw[int(seed)] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        per_seed_fit[int(seed)] = json.loads(Path(fit_path).read_text(encoding="utf-8"))

    # Aggregate with hierarchical bootstrap.
    seed_list = [int(s) for s in args.seeds]
    num_budgets = len(budgets)

    def stack(metric_key: str) -> np.ndarray:
        mats = []
        for seed in seed_list:
            samples = per_seed_raw[seed].get("samples", {}).get(metric_key, [])
            if len(samples) != num_budgets:
                raise RuntimeError(f"Seed {seed} missing samples.{metric_key} for all budgets")
            mats.append(np.asarray(samples, dtype=np.float64))
        # (S, B, N) -> split by budget
        return np.stack(mats, axis=0)

    frame_f1 = stack("frame_f1")
    iou = stack("iou")
    dice = stack("dice")
    hit = stack("hit_rate")
    flops_total = stack("flops_total")
    warm_time = stack("warm_time_s")

    nlg_ci = []
    iou_ci = []
    dice_ci = []
    hit_ci = []
    combined_ci = []
    flops_ci = []
    warm_mean_ci = []
    warm_p95 = []

    for bidx in range(num_budgets):
        nlg_ci.append(_mean_and_ci_from_seed_sample_matrix(frame_f1[:, bidx, :], n_boot=args.n_bootstrap, seed=seed_list[0] + bidx, ci=args.ci))
        iou_ci.append(_mean_and_ci_from_seed_sample_matrix(iou[:, bidx, :], n_boot=args.n_bootstrap, seed=seed_list[0] + 100 + bidx, ci=args.ci))
        dice_ci.append(_mean_and_ci_from_seed_sample_matrix(dice[:, bidx, :], n_boot=args.n_bootstrap, seed=seed_list[0] + 200 + bidx, ci=args.ci))
        hit_ci.append(_mean_and_ci_from_seed_sample_matrix(hit[:, bidx, :], n_boot=args.n_bootstrap, seed=seed_list[0] + 300 + bidx, ci=args.ci))

        # Combined per-sample from raw metrics
        comb = 0.5 * frame_f1[:, bidx, :] + 0.5 * iou[:, bidx, :]
        combined_ci.append(_mean_and_ci_from_seed_sample_matrix(comb, n_boot=args.n_bootstrap, seed=seed_list[0] + 400 + bidx, ci=args.ci))

        flops_ci.append(_mean_and_ci_from_seed_sample_matrix(flops_total[:, bidx, :], n_boot=args.n_bootstrap, seed=seed_list[0] + 500 + bidx, ci=args.ci))
        warm_mean_ci.append(_mean_and_ci_from_seed_sample_matrix(warm_time[:, bidx, :], n_boot=args.n_bootstrap, seed=seed_list[0] + 600 + bidx, ci=args.ci))
        warm_p95.append(_p95_from_seed_sample_matrix(warm_time[:, bidx, :]))

    # Scaling fits on aggregated means
    nlg_means = [x["mean"] for x in nlg_ci]
    iou_means = [x["mean"] for x in iou_ci]
    comb_means = [x["mean"] for x in combined_ci]
    nlg_fit, _ = fit_scaling_law(budgets, nlg_means)
    iou_fit, _ = fit_scaling_law(budgets, iou_means)
    comb_fit, _ = fit_scaling_law(budgets, comb_means)

    nlg_dr = compute_diminishing_returns_point(nlg_fit)
    iou_dr = compute_diminishing_returns_point(iou_fit)
    comb_dr = compute_diminishing_returns_point(comb_fit)

    # Artifact meta for aggregated report
    repo_root = Path(__file__).resolve().parents[2]
    data_revision = "synthetic"
    split_manifest_path = ""
    if args.dataset_type == "manifest":
        data_revision, split_manifest_path = try_manifest_revision(str(args.manifest))
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(seed_list[0]),
        config={
            "dataset_type": args.dataset_type,
            "manifest_path": args.manifest,
            "split": args.split,
            "resize_shape": list(args.resize_shape),
            "pcg": args.pcg,
            "llama2_path": args.llama2_path,
            "llama2_quant": args.llama2_quant,
            "pcg_refresh_period": args.pcg_refresh_period,
            "encoder": args.encoder,
            "encoder_device": args.encoder_device,
            "use_evidence_head": (not bool(args.no_evidence_head)),
            "require_full_budget": bool(args.require_full_budget),
            "lambda_uncertainty": float(args.lambda_uncertainty),
            "budget_mode": args.budget_mode,
            "costs_json": args.costs_json,
            "b_gen": args.b_gen,
            "n_verify": args.n_verify,
            "budgets": budgets,
            "n_samples": args.n_samples,
            "seeds": seed_list,
            "n_bootstrap": args.n_bootstrap,
            "ci": args.ci,
        },
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    report: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "budget_mode": str(args.budget_mode),
        "budgets": budgets,
        "seeds": seed_list,
        "n_samples": int(args.n_samples),
        "n_bootstrap": int(args.n_bootstrap),
        "ci": float(args.ci),
        "metrics": {
            "frame_f1": nlg_ci,
            "iou": iou_ci,
            "dice": dice_ci,
            "hit_rate": hit_ci,
            "combined": combined_ci,
        },
        "compute": {
            "flops_total": flops_ci,
        },
        "latency": {
            "warm_mean_s": warm_mean_ci,
            "warm_p95_s": warm_p95,
        },
        "scaling_fit": {
            "nlg": {
                "model_type": nlg_fit.model_type,
                "params": nlg_fit.params,
                "r_squared": nlg_fit.r_squared,
                "aic": nlg_fit.aic,
                "bic": nlg_fit.bic,
                "diminishing_point": nlg_dr,
            },
            "grounding": {
                "model_type": iou_fit.model_type,
                "params": iou_fit.params,
                "r_squared": iou_fit.r_squared,
                "aic": iou_fit.aic,
                "bic": iou_fit.bic,
                "diminishing_point": iou_dr,
            },
            "combined": {
                "model_type": comb_fit.model_type,
                "params": comb_fit.params,
                "r_squared": comb_fit.r_squared,
                "aic": comb_fit.aic,
                "bic": comb_fit.bic,
                "diminishing_point": comb_dr,
            },
        },
        "per_seed": {
            "dirs": {str(k): v for k, v in per_seed_dirs.items()},
        },
    }

    out_path = os.path.join(args.output_dir, "fig2_multiseed.json")
    save_results_json(report, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
