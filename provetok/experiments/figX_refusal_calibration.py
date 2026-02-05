"""Fig X: Refusal calibration runner (proposal §4.3.5 / §7.1)

This is a minimal scaffold that:
1) Generates a set of (Generation, Issues) pairs on synthetic data
2) Tunes tau_refuse on a "val" set under a critical miss-rate constraint
3) Reports unsupported_rate / critical_miss_rate / refusal_rate / ECE
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from ..bet.refine_loop import run_refine_loop
from ..data import make_dataloader
from ..eval.compute_budget import ComputeUnitCosts, match_b_enc_for_total_flops
from ..pcg.generator import ToyPCG
from ..verifier import verify
from ..pcg.refusal import (
    RefusalCalibrator,
    RefusalPolicy,
    apply_q_calibration_to_generation,
    CalibrationMetrics,
)
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .utils import create_synthetic_volume, make_output_dir, save_results_json, set_seed


@dataclass
class RefusalCalibConfig:
    dataset_type: str = "synthetic"  # "synthetic" | "manifest"
    manifest_path: str = ""
    dev_split: str = "val"
    test_split: str = "test"
    n_samples: int = 50
    n_samples_test: int = 50
    volume_shape: tuple[int, int, int] = (64, 64, 64)
    resize_shape: tuple[int, int, int] = (64, 64, 64)
    n_lesions: int = 3
    budget_tokens: int = 64
    budget_mode: str = "tokens"  # "tokens" | "flops"
    budgets: List[float] = None  # type: ignore[assignment]
    costs_json: str = ""
    b_gen: int = 128
    n_verify: int = 1
    refine_steps: int = 3
    emb_dim: int = 32
    topk_citations: int = 3
    seed: int = 42
    candidate_taus: List[float] = None  # type: ignore[assignment]
    max_critical_miss_rate: float = 0.05
    output_dir: str = "./outputs/figX_refusal_calibration"


def _metrics_to_dict(m: CalibrationMetrics) -> Dict[str, Any]:
    return {
        "unsupported_rate": float(m.unsupported_rate),
        "critical_miss_rate": float(m.critical_miss_rate),
        "refusal_rate": float(m.refusal_rate),
        "refusal_ece": float(m.refusal_ece),
        "reliability_bins": m.reliability_bins,
        "critical_refusal_count": int(m.critical_refusal_count),
        "total_critical_count": int(m.total_critical_count),
    }


def _load_manifest_samples(manifest_path: str, *, split: str, max_samples: int, resize_shape: tuple[int, int, int]) -> List[Dict[str, Any]]:
    dl = make_dataloader(
        {
            "dataset_type": "manifest",
            "manifest_path": str(manifest_path),
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": int(max_samples),
            "resize_shape": tuple(int(x) for x in resize_shape),
        },
        split=str(split),
    )
    out: List[Dict[str, Any]] = []
    for batch in dl:
        out.append(
            {
                "volume": batch["volume"][0],
                "gt_frames": batch.get("frames", [[]])[0] or [],
            }
        )
    if not out:
        raise RuntimeError(f"No samples loaded for split={split!r}. Check manifest or split names.")
    return out


def _budget_to_b_enc(cfg: RefusalCalibConfig, costs: ComputeUnitCosts, budget: float) -> int:
    if cfg.budget_mode == "flops":
        return match_b_enc_for_total_flops(
            flops_total=float(budget),
            b_gen=int(cfg.b_gen),
            n_verify=int(cfg.n_verify),
            costs=costs,
            flops_extra=0.0,
            min_b_enc=1,
            max_b_enc=4096,
        )
    return int(budget)


def _run_provetok(
    *,
    volume: torch.Tensor,
    gt_frames: List[Any],
    budget_tokens: int,
    cfg: RefusalCalibConfig,
    sample_seed: int,
) -> Dict[str, Any]:
    pcg = ToyPCG(
        emb_dim=cfg.emb_dim,
        topk=cfg.topk_citations,
        seed=sample_seed,
        refusal_threshold=0.0,
        q_strategy="support",
    )
    result = run_refine_loop(
        volume=volume,
        budget_tokens=int(budget_tokens),
        steps=cfg.refine_steps,
        generator_fn=lambda toks: pcg(toks),
        verifier_fn=lambda gen, toks: verify(gen, toks),
        emb_dim=cfg.emb_dim,
        seed=sample_seed,
        require_full_budget=True,
        use_evidence_head=False,
    )
    return {"tokens": result.tokens, "gen": result.gen, "issues": result.issues, "gt_frames": gt_frames}


def run_refusal_calibration(cfg: RefusalCalibConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)

    repo_root = Path(__file__).resolve().parents[2]
    data_revision = "synthetic"
    split_manifest_path = ""
    if cfg.dataset_type == "manifest":
        data_revision, split_manifest_path = try_manifest_revision(cfg.manifest_path)
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(cfg.seed),
        config=asdict(cfg),
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    if cfg.candidate_taus is None:
        # Include fine-grained small taus so we can avoid the degenerate regime
        # where tau is either too small to refuse anything or so large that
        # max_refusal_rate capping always saturates.
        candidate_taus = sorted(
            {
                0.0,
                *[i / 1000.0 for i in range(1, 21)],  # 0.001 .. 0.020
                *[i / 100.0 for i in range(1, 21)],  # 0.01 .. 0.20
                0.25,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            }
        )
    else:
        candidate_taus = list(cfg.candidate_taus)

    costs = ComputeUnitCosts.from_json(cfg.costs_json) if cfg.costs_json else ComputeUnitCosts()
    budgets = cfg.budgets or ([float(cfg.budget_tokens)] if cfg.budget_mode == "tokens" else [2_000_000.0, 3_000_000.0, 4_000_000.0, 5_000_000.0])
    budgets = [float(b) for b in budgets]

    if cfg.dataset_type == "manifest":
        dev_samples = _load_manifest_samples(cfg.manifest_path, split=cfg.dev_split, max_samples=cfg.n_samples, resize_shape=cfg.resize_shape)
        test_samples = _load_manifest_samples(cfg.manifest_path, split=cfg.test_split, max_samples=cfg.n_samples_test, resize_shape=cfg.resize_shape)
        calib_budget = float(budgets[-1])
        calib_b_enc = int(_budget_to_b_enc(cfg, costs, calib_budget))

        generations = []
        ground_truths = []
        issues_list = []
        tokens_list = []
        for i, s in enumerate(dev_samples):
            sample_seed = int(cfg.seed) + int(i)
            out = _run_provetok(volume=s["volume"], gt_frames=s["gt_frames"], budget_tokens=calib_b_enc, cfg=cfg, sample_seed=sample_seed)
            generations.append(out["gen"])
            ground_truths.append(out["gt_frames"])
            issues_list.append(out["issues"])
            tokens_list.append(out["tokens"])
    else:
        generations = []
        ground_truths = []
        issues_list = []
        tokens_list = []
        for i in range(cfg.n_samples):
            sample_seed = cfg.seed + i
            volume, _ = create_synthetic_volume(shape=cfg.volume_shape, n_lesions=cfg.n_lesions, seed=sample_seed)
            out = _run_provetok(volume=volume, gt_frames=[], budget_tokens=int(cfg.budget_tokens), cfg=cfg, sample_seed=sample_seed)
            generations.append(out["gen"])
            issues_list.append(out["issues"])
            tokens_list.append(out["tokens"])
            # Scaffold GT: treat generated frames as "correct" so calibration code path is exercised.
            ground_truths.append(out["gen"].frames)

    calibrator = RefusalCalibrator(
        tau_refuse=0.5,
        max_critical_miss_rate=cfg.max_critical_miss_rate,
        num_bins=10,
    )
    # Stricter-than-proof anti-gaming cap: we keep the paper-grade proof gate at 0.20
    # (see scripts/proof_check.py), but use a tighter cap in the calibration run so the
    # selected policy has margin and is less likely to sit exactly on the boundary.
    max_refusal_rate = 0.15
    # Calibrate q_k for support/reliability using the dev set (ECE should be meaningful).
    q_calibration = calibrator.fit_q_calibration_map(generations, ground_truths, issues_list=issues_list)
    generations_calib_q = [apply_q_calibration_to_generation(g, q_calibration) for g in generations]
    best_tau, best_metrics = calibrator.find_optimal_threshold(
        val_generations=generations_calib_q,
        val_ground_truths=ground_truths,
        val_issues=issues_list,
        candidate_taus=candidate_taus,
        max_refusal_rate=max_refusal_rate,
        max_refusal_ece=0.15,
        val_tokens=tokens_list,
        verifier_fn=lambda gen, toks: verify(gen, toks),
    )

    policy = RefusalPolicy(
        tau_refuse=float(best_tau),
        max_critical_miss_rate=float(cfg.max_critical_miss_rate),
        max_refusal_rate=max_refusal_rate,
        num_bins=10,
        q_calibration=q_calibration,
    )

    report: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "config": asdict(cfg),
        "best_tau": float(best_tau),
        "refusal_policy": policy.to_dict(),
        "q_calibration": q_calibration,
        "best_metrics_dev": _metrics_to_dict(best_metrics),
        "budget_mode": str(cfg.budget_mode),
        "budgets": budgets,
    }

    if cfg.dataset_type == "manifest":
        test_rows: List[Dict[str, Any]] = []
        for b in budgets:
            b_enc = int(_budget_to_b_enc(cfg, costs, float(b)))
            gens = []
            toks = []
            gts = []
            issues = []
            for i, s in enumerate(test_samples):
                sample_seed = int(cfg.seed) + 10_000 + int(i)
                out = _run_provetok(volume=s["volume"], gt_frames=s["gt_frames"], budget_tokens=b_enc, cfg=cfg, sample_seed=sample_seed)
                gens.append(out["gen"])
                toks.append(out["tokens"])
                gts.append(out["gt_frames"])
                issues.append(out["issues"])

            # Baseline: no refusal.
            base_policy = RefusalPolicy(
                tau_refuse=0.0,
                max_critical_miss_rate=float(cfg.max_critical_miss_rate),
                num_bins=10,
                q_calibration=q_calibration,
            )
            gens_base = [base_policy.apply(g) for g in gens]
            issues_base = [verify(gb, tb) for gb, tb in zip(gens_base, toks)]
            m_base = calibrator.compute_calibration_metrics(gens_base, gts, issues_base)

            # Calibrated: fixed tau across budgets.
            gens_cal = [policy.apply(g) for g in gens]
            issues_cal = [verify(gc, tc) for gc, tc in zip(gens_cal, toks)]
            m_cal = calibrator.compute_calibration_metrics(gens_cal, gts, issues_cal)

            test_rows.append(
                {
                    "budget": float(b),
                    "b_enc": int(b_enc),
                    "no_refusal": _metrics_to_dict(m_base),
                    "calibrated": _metrics_to_dict(m_cal),
                }
            )

        report["test"] = {
            "split": str(cfg.test_split),
            "rows": test_rows,
        }

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run refusal calibration (synthetic scaffold)")
    ap.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest"])
    ap.add_argument("--manifest", type=str, default="", help="Manifest path when dataset-type=manifest")
    ap.add_argument("--dev-split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--test-split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--smoke", action="store_true", help="Small config for quick sanity")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--n-samples-test", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--budget-mode", type=str, default="tokens", choices=["tokens", "flops"])
    ap.add_argument("--budgets", type=float, nargs="+", default=None, help="Budgets (tokens or total FLOPs caps).")
    ap.add_argument("--costs-json", type=str, default="")
    ap.add_argument("--b-gen", type=int, default=128)
    ap.add_argument("--n-verify", type=int, default=1)
    ap.add_argument("--output-dir", type=str, default="./outputs/figX_refusal_calibration")
    ap.add_argument("--max-critical-miss-rate", type=float, default=0.05)
    args = ap.parse_args()

    cfg = RefusalCalibConfig(
        dataset_type=str(args.dataset_type),
        manifest_path=str(args.manifest),
        dev_split=str(args.dev_split),
        test_split=str(args.test_split),
        n_samples=args.n_samples,
        n_samples_test=int(args.n_samples_test),
        seed=args.seed,
        resize_shape=tuple(int(x) for x in args.resize_shape),
        budget_mode=str(args.budget_mode),
        budgets=list(args.budgets) if args.budgets else None,
        costs_json=str(args.costs_json),
        b_gen=int(args.b_gen),
        n_verify=int(args.n_verify),
        max_critical_miss_rate=args.max_critical_miss_rate,
        output_dir=args.output_dir,
    )
    if args.smoke:
        cfg = RefusalCalibConfig(
            dataset_type=str(args.dataset_type),
            manifest_path=str(args.manifest),
            dev_split=str(args.dev_split),
            test_split=str(args.test_split),
            n_samples=10,
            n_samples_test=10,
            volume_shape=(32, 32, 32),
            n_lesions=3,
            budget_tokens=32,
            budget_mode=str(args.budget_mode),
            budgets=list(args.budgets) if args.budgets else None,
            costs_json=str(args.costs_json),
            b_gen=int(args.b_gen),
            n_verify=int(args.n_verify),
            refine_steps=2,
            emb_dim=32,
            topk_citations=2,
            seed=args.seed,
            max_critical_miss_rate=args.max_critical_miss_rate,
            output_dir=args.output_dir,
            resize_shape=(32, 32, 32),
        )

    out_dir = make_output_dir(cfg.output_dir, "figX_refusal_calibration")
    base_dir = Path(cfg.output_dir)
    cfg = RefusalCalibConfig(**{**asdict(cfg), "output_dir": out_dir})

    report = run_refusal_calibration(cfg)
    save_results_json(report, os.path.join(out_dir, "figX_refusal_calibration.json"))
    save_results_json(report["refusal_policy"], os.path.join(out_dir, "refusal_policy.json"))
    # Also write stable "latest" artifacts at the base output_dir for proof_check / docs links.
    base_dir.mkdir(parents=True, exist_ok=True)
    save_results_json(report, str(base_dir / "figX_refusal_calibration.json"))
    save_results_json(report["refusal_policy"], str(base_dir / "refusal_policy.json"))
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
