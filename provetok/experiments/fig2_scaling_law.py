"""Fig 2: Scaling Law 实验

复现 proposal Fig 2 - Performance vs Budget 曲线

实验设计:
1. 在不同 budget (B_enc) 下运行 BET refine loop
2. 测量 NLG 指标 (Frame F1, BLEU, ROUGE) 和 Grounding 指标 (IoU, hit-rate)
3. 拟合 scaling law (power law / log saturation)
4. 绘制 scaling 曲线
5. 计算边际收益递减点

输出:
- fig2_scaling_curve.png: Scaling 曲线图
- fig2_scaling_fit.json: 拟合参数和统计量
- fig2_raw_data.json: 原始实验数据
"""
from __future__ import annotations
from typing import Any, List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import torch
import numpy as np
import json
import os
import time
from tqdm import tqdm

from ..bet.refine_loop import run_refine_loop, RefineResult
from ..bet.evidence_head import EvidenceHead
from ..data import make_dataloader
from ..eval.compute_budget import ComputeUnitCosts, format_budget_report, match_b_enc_for_total_flops
from ..eval.scaling import (
    fit_scaling_law,
    compute_diminishing_returns_point,
    format_scaling_report,
    ScalingFit,
)
from ..eval.metrics_grounding import (
    compute_generation_grounding,
)
from ..eval.metrics_frames import compute_frame_f1
from ..eval.metrics_text import MissingTextMetricDependency, compute_text_metrics
from ..pcg.generator import ToyPCG
from ..pcg.llama2_pcg import Llama2PCG, Llama2PCGConfig
from ..verifier import verify
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from ..bet.encoders.simple_cnn3d import SimpleCNN3D
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..data.frame_extractor import frames_to_report
from .utils import (
    set_seed,
    make_output_dir,
    mock_generator_fn,
    mock_verifier_fn,
    create_synthetic_volume,
    compute_nlg_metrics,
    aggregate_metrics,
    save_results_json,
)


@dataclass
class ScalingExperimentConfig:
    """Scaling 实验配置"""
    # Budget 范围
    budgets: List[float] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    budget_mode: str = "tokens"  # "tokens" or "flops"

    # Refine loop 参数
    max_steps: int = 20
    emb_dim: int = 32
    epsilon: float = 0.01
    max_depth: int = 4

    # 数据参数
    n_samples: int = 50  # 每个 budget 测试的样本数
    volume_shape: Tuple[int, int, int] = (64, 64, 64)
    n_lesions_per_sample: int = 3
    dataset_type: str = "synthetic"  # "synthetic" or "manifest"
    manifest_path: str = ""          # required when dataset_type="manifest"
    split: str = "test"
    resize_shape: Tuple[int, int, int] = (64, 64, 64)
    pcg_backend: str = "toy"  # "toy" or "llama2"
    llama2_path: str = "/data/models/Llama-2-7b-chat-hf"
    llama2_quant: str = "fp16"  # "fp16" or "8bit"
    pcg_refresh_period: int = 1
    encoder_backend: str = "toy"  # "toy" or "cnn3d"
    encoder_device: str = "cuda"

    # Evidence Head
    use_evidence_head: bool = True
    lambda_uncertainty: float = 0.3
    require_full_budget: bool = False

    # 实验控制
    seed: int = 42
    output_dir: str = "./outputs/fig2_scaling"

    # 指标权重（用于综合 scaling）
    nlg_weight: float = 0.5
    grounding_weight: float = 0.5

    # Compute accounting (toy unit-cost model, see scripts/profile_flops.py)
    b_gen: int = 128
    n_verify: int = 1
    costs_json: str = ""
    fail_fast_matched: bool = True
    compute_text_metrics: bool = True


@dataclass
class SingleBudgetResult:
    """单个 budget 的实验结果"""
    budget: float
    n_samples: int

    # NLG metrics (mean, std)
    frame_f1_mean: float
    frame_f1_std: float

    # Grounding metrics (mean, std)
    iou_mean: float
    iou_std: float
    dice_mean: float
    dice_std: float
    hit_rate: float

    # Per-sample raw metrics (for CI / paired bootstrap)
    frame_f1_samples: List[float]
    iou_samples: List[float]
    dice_samples: List[float]
    hit_samples: List[float]

    # Text metrics (optional; vs raw report_text when available)
    bleu_mean: float
    bleu_std: float
    rouge1_mean: float
    rouge1_std: float
    rouge2_mean: float
    rouge2_std: float
    rougeL_mean: float
    rougeL_std: float
    bleu_samples: List[float]
    rouge1_samples: List[float]
    rouge2_samples: List[float]
    rougeL_samples: List[float]

    # Efficiency
    avg_tokens_used: float
    avg_steps: float
    avg_stop_reason: Dict[str, int]
    tokens_used_samples: List[float]
    steps_used_samples: List[float]

    # Compute + latency
    flops_total_target: float
    flops_total_mean: float
    flops_total_std: float
    flops_total_samples: List[float]
    warm_mean_s: float
    warm_p95_s: float
    cold_mean_s: float
    cold_p95_s: float
    warm_times_s: List[float]

    # Combined score
    combined_score: float


@dataclass
class ScalingExperimentResult:
    """完整实验结果"""
    config: ScalingExperimentConfig
    budget_results: List[SingleBudgetResult]

    # Scaling law fits
    nlg_fit: Optional[ScalingFit]
    grounding_fit: Optional[ScalingFit]
    combined_fit: Optional[ScalingFit]

    # Analysis
    nlg_diminishing_point: float
    grounding_diminishing_point: float
    combined_diminishing_point: float


def run_single_budget(
    budget: float,
    config: ScalingExperimentConfig,
    generator_fn: Callable,
    verifier_fn: Callable,
    evidence_head: Optional[EvidenceHead],
    costs: ComputeUnitCosts,
    encoder: Optional[torch.nn.Module] = None,
    samples: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
) -> SingleBudgetResult:
    """在单个 budget 下运行实验

    Args:
        budget: Token 预算
        config: 实验配置
        generator_fn: PCG 生成函数
        verifier_fn: Verifier 函数
        evidence_head: Evidence Head 模型
        verbose: 是否打印进度

    Returns:
        SingleBudgetResult
    """
    frame_f1_scores = []
    iou_scores = []
    dice_scores = []
    hit_scores = []
    tokens_used = []
    steps_used = []
    stop_reasons = {}
    flops_total_samples = []
    warm_times_s: List[float] = []
    bleu_scores: List[float] = []
    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rougeL_scores: List[float] = []

    iterator = range(config.n_samples if samples is None else len(samples))
    if verbose:
        iterator = tqdm(iterator, desc=f"Budget={budget}")

    flops_total_target = float(budget) if config.budget_mode == "flops" else 0.0
    b_enc_target = (
        match_b_enc_for_total_flops(
            flops_total=float(budget),
            b_gen=int(config.b_gen),
            n_verify=int(config.n_verify),
            costs=costs,
            min_b_enc=1,
            max_b_enc=4096,
        )
        if config.budget_mode == "flops"
        else int(budget)
    )

    for sample_idx in iterator:
        sample_seed = config.seed + sample_idx

        if samples is None:
            # 创建合成数据
            volume, lesion_masks = create_synthetic_volume(
                shape=config.volume_shape,
                n_lesions=config.n_lesions_per_sample,
                seed=sample_seed,
            )
            gt_frames = None
            ref_report = None
            volume_shape = config.volume_shape
        else:
            s = samples[sample_idx]
            volume = s["volume"]
            lesion_masks = s.get("lesion_masks", {}) or {}
            gt_frames = s.get("gt_frames")
            ref_report = s.get("report_text")
            volume_shape = tuple(volume.shape)

        # 运行 refine loop
        t0 = time.perf_counter()
        result = run_refine_loop(
            volume=volume,
            budget_tokens=int(b_enc_target),
            steps=config.max_steps,
            generator_fn=generator_fn,
            verifier_fn=verifier_fn,
            emb_dim=config.emb_dim,
            seed=sample_seed,
            encoder=encoder,
            require_full_budget=bool(config.require_full_budget),
            evidence_head=evidence_head,
            use_evidence_head=config.use_evidence_head,
            epsilon=config.epsilon,
            max_depth=config.max_depth,
            lambda_uncertainty=config.lambda_uncertainty,
            pcg_refresh_period=config.pcg_refresh_period,
        )
        t1 = time.perf_counter()
        warm_times_s.append(float(t1 - t0))

        # 计算 Frame F1（用 mock ground truth）
        # 实际使用时应该用真实 GT frames
        pred_frames = result.gen.frames
        if gt_frames is None:
            gt_frames = pred_frames[: max(1, len(pred_frames) - 1)]  # Mock GT
        # NOTE: threshold=0.3 can yield false matches when finding types differ but
        # other slots are unspecified/present. Use a stricter cutoff so a finding
        # mismatch does not accidentally count as a match.
        frame_metrics = compute_frame_f1(pred_frames, gt_frames, threshold=0.4)
        frame_f1_scores.append(frame_metrics.f1)

        # Text metrics (optional) using a simple canonical report rendering.
        if bool(config.compute_text_metrics) and ref_report is not None:
            pred_text = frames_to_report(result.gen.frames)
            try:
                tm = compute_text_metrics(pred_text, str(ref_report))
            except MissingTextMetricDependency:
                tm = {}
            bleu_scores.append(float(tm.get("bleu", 0.0)))
            rouge1_scores.append(float(tm.get("rouge1", 0.0)))
            rouge2_scores.append(float(tm.get("rouge2", 0.0)))
            rougeL_scores.append(float(tm.get("rougeL", 0.0)))

        # 计算 Grounding 指标（union-level，避免 frame_idx↔mask 索引不一致导致退化为 0）
        g = compute_generation_grounding(
            generation=result.gen,
            tokens=result.tokens,
            lesion_masks=lesion_masks,
            volume_shape=volume_shape,
        )
        iou_scores.append(float(g["iou_union"]))
        dice_scores.append(float(g["dice_union"]))
        hit_scores.append(float(g["hit"]))

        # 记录效率
        tokens_used.append(len(result.tokens))
        steps_used.append(result.total_steps)

        bud_realized = format_budget_report(
            b_enc=int(len(result.tokens)),
            b_gen=int(config.b_gen),
            n_verify=int(config.n_verify),
            costs=costs,
            flops_extra=0.0,
        )
        flops_total_samples.append(float(bud_realized["flops_total"]))
        if config.budget_mode == "flops" and config.fail_fast_matched:
            # In flops-budget mode, matching is defined on the *budget cap* (b_enc_target),
            # while the realized token count may be lower due to early stopping (no-issues/epsilon).
            bud_target = format_budget_report(
                b_enc=int(b_enc_target),
                b_gen=int(config.b_gen),
                n_verify=int(config.n_verify),
                costs=costs,
                flops_extra=0.0,
            )
            tol = max(1e-6, 8.0 * float(costs.flops_per_enc_token))
            if abs(float(bud_target["flops_total"]) - float(flops_total_target)) > tol:
                raise RuntimeError(
                    "FLOPs-matching failed for budget_mode=flops (budget cap): "
                    f"total={float(bud_target['flops_total']):.2f} vs target={float(flops_total_target):.2f} "
                    f"(tol={tol:.2f}). Consider adjusting --b-gen/--n-verify or budgets."
                )

        reason = result.stopped_reason.split()[0]
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1

    cold_times = warm_times_s[: min(3, len(warm_times_s))]
    warm_mean_s = float(np.mean(warm_times_s)) if warm_times_s else 0.0
    warm_p95_s = float(np.quantile(np.asarray(warm_times_s, dtype=np.float64), 0.95)) if warm_times_s else 0.0
    cold_mean_s = float(np.mean(cold_times)) if cold_times else 0.0
    cold_p95_s = float(np.quantile(np.asarray(cold_times, dtype=np.float64), 0.95)) if cold_times else 0.0

    # 计算 combined score
    nlg_mean = np.mean(frame_f1_scores)
    grounding_mean = np.mean(iou_scores)
    combined = (
        config.nlg_weight * nlg_mean +
        config.grounding_weight * grounding_mean
    )

    return SingleBudgetResult(
        budget=budget,
        n_samples=config.n_samples,
        frame_f1_mean=nlg_mean,
        frame_f1_std=np.std(frame_f1_scores),
        iou_mean=grounding_mean,
        iou_std=np.std(iou_scores),
        dice_mean=np.mean(dice_scores),
        dice_std=np.std(dice_scores),
        hit_rate=np.mean(hit_scores),
        frame_f1_samples=[float(x) for x in frame_f1_scores],
        iou_samples=[float(x) for x in iou_scores],
        dice_samples=[float(x) for x in dice_scores],
        hit_samples=[float(x) for x in hit_scores],
        bleu_mean=float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        bleu_std=float(np.std(bleu_scores)) if bleu_scores else 0.0,
        rouge1_mean=float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
        rouge1_std=float(np.std(rouge1_scores)) if rouge1_scores else 0.0,
        rouge2_mean=float(np.mean(rouge2_scores)) if rouge2_scores else 0.0,
        rouge2_std=float(np.std(rouge2_scores)) if rouge2_scores else 0.0,
        rougeL_mean=float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        rougeL_std=float(np.std(rougeL_scores)) if rougeL_scores else 0.0,
        bleu_samples=[float(x) for x in bleu_scores],
        rouge1_samples=[float(x) for x in rouge1_scores],
        rouge2_samples=[float(x) for x in rouge2_scores],
        rougeL_samples=[float(x) for x in rougeL_scores],
        avg_tokens_used=np.mean(tokens_used),
        avg_steps=np.mean(steps_used),
        avg_stop_reason=stop_reasons,
        tokens_used_samples=[float(x) for x in tokens_used],
        steps_used_samples=[float(x) for x in steps_used],
        flops_total_target=float(flops_total_target),
        flops_total_mean=float(np.mean(flops_total_samples)) if flops_total_samples else 0.0,
        flops_total_std=float(np.std(flops_total_samples)) if flops_total_samples else 0.0,
        flops_total_samples=[float(x) for x in flops_total_samples],
        warm_mean_s=warm_mean_s,
        warm_p95_s=warm_p95_s,
        cold_mean_s=cold_mean_s,
        cold_p95_s=cold_p95_s,
        warm_times_s=[float(x) for x in warm_times_s],
        combined_score=combined,
    )


def run_scaling_experiment(
    config: ScalingExperimentConfig = None,
    generator_fn: Callable = None,
    verifier_fn: Callable = None,
    verbose: bool = True,
) -> ScalingExperimentResult:
    """运行完整的 Scaling Law 实验

    Args:
        config: 实验配置（默认使用 ScalingExperimentConfig()）
        generator_fn: PCG 生成函数（默认使用 mock）
        verifier_fn: Verifier 函数（默认使用 mock）
        verbose: 是否打印进度

    Returns:
        ScalingExperimentResult
    """
    config = config or ScalingExperimentConfig()
    if generator_fn is None:
        if config.pcg_backend == "llama2":
            pcg = Llama2PCG(
                Llama2PCGConfig(
                    model_path=config.llama2_path,
                    device="cuda",
                    quantization=config.llama2_quant,
                    # Keep output cap aligned with compute accounting (b_gen).
                    max_new_tokens=max(128, int(config.b_gen)),
                    temperature=0.0,
                    topk_citations=3,
                )
            )
            generator_fn = lambda toks: pcg(toks)
        else:
            pcg = ToyPCG(emb_dim=config.emb_dim, topk=3, seed=config.seed)
            generator_fn = lambda toks: pcg(toks)
    if verifier_fn is None:
        verifier_fn = lambda gen, toks: verify(gen, toks)

    set_seed(config.seed)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    costs = ComputeUnitCosts.from_json(config.costs_json) if config.costs_json else ComputeUnitCosts()

    samples: Optional[List[Dict[str, Any]]] = None
    if config.dataset_type == "manifest":
        if not config.manifest_path:
            raise ValueError("dataset_type=manifest requires `manifest_path`")
        dl = make_dataloader(
            {
                "dataset_type": "manifest",
                "manifest_path": config.manifest_path,
                "batch_size": 1,
                "num_workers": 0,
                "max_samples": config.n_samples,
                "resize_shape": config.resize_shape,
            },
            split=config.split,
        )
        samples = []
        for batch in dl:
            vol = batch["volume"][0]
            masks = batch.get("lesion_masks", [{}])[0] or {}
            gt_frames = batch.get("frames", [[]])[0] or []
            report_text = str((batch.get("report_text") or [""])[0])
            samples.append({"volume": vol, "lesion_masks": masks, "gt_frames": gt_frames, "report_text": report_text})

    # 创建 Evidence Head
    evidence_head = None
    if config.use_evidence_head:
        evidence_head = EvidenceHead(
            emb_dim=config.emb_dim,
            lambda_uncertainty=config.lambda_uncertainty,
        )

    encoder: Optional[torch.nn.Module] = None
    if config.encoder_backend == "cnn3d":
        encoder = SimpleCNN3D(in_channels=1, emb_dim=config.emb_dim).to(config.encoder_device)

    # 对每个 budget 运行实验
    budget_results = []
    for budget in config.budgets:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running budget = {budget}")
            print(f"{'='*50}")

        result = run_single_budget(
            budget=budget,
            config=config,
            generator_fn=generator_fn,
            verifier_fn=verifier_fn,
            evidence_head=evidence_head,
            costs=costs,
            encoder=encoder,
            samples=samples,
            verbose=verbose,
        )
        budget_results.append(result)

        if verbose:
            print(f"  Frame F1: {result.frame_f1_mean:.4f} ± {result.frame_f1_std:.4f}")
            print(f"  IoU:      {result.iou_mean:.4f} ± {result.iou_std:.4f}")
            print(f"  Combined: {result.combined_score:.4f}")

    # 拟合 Scaling Law
    budgets = [r.budget for r in budget_results]
    nlg_scores = [r.frame_f1_mean for r in budget_results]
    grounding_scores = [r.iou_mean for r in budget_results]
    combined_scores = [r.combined_score for r in budget_results]

    nlg_fit, nlg_reason = fit_scaling_law(budgets, nlg_scores)
    grounding_fit, grounding_reason = fit_scaling_law(budgets, grounding_scores)
    combined_fit, combined_reason = fit_scaling_law(budgets, combined_scores)

    if verbose:
        print(f"\n{'='*50}")
        print("Scaling Law Fits")
        print(f"{'='*50}")
        print(f"NLG: {nlg_fit.model_type} (R²={nlg_fit.r_squared:.4f})")
        print(f"Grounding: {grounding_fit.model_type} (R²={grounding_fit.r_squared:.4f})")
        print(f"Combined: {combined_fit.model_type} (R²={combined_fit.r_squared:.4f})")

    # 计算边际收益递减点
    nlg_dr = compute_diminishing_returns_point(nlg_fit)
    grounding_dr = compute_diminishing_returns_point(grounding_fit)
    combined_dr = compute_diminishing_returns_point(combined_fit)

    if verbose:
        print(f"\nDiminishing Returns Points:")
        print(f"  NLG: B = {nlg_dr:.0f}")
        print(f"  Grounding: B = {grounding_dr:.0f}")
        print(f"  Combined: B = {combined_dr:.0f}")

    # 构建结果
    result = ScalingExperimentResult(
        config=config,
        budget_results=budget_results,
        nlg_fit=nlg_fit,
        grounding_fit=grounding_fit,
        combined_fit=combined_fit,
        nlg_diminishing_point=nlg_dr,
        grounding_diminishing_point=grounding_dr,
        combined_diminishing_point=combined_dr,
    )

    # 保存结果
    _save_scaling_results(result, config.output_dir)

    return result


def _save_scaling_results(result: ScalingExperimentResult, output_dir: str):
    """保存实验结果"""
    repo_root = Path(__file__).resolve().parents[2]
    data_revision = "synthetic"
    split_manifest_path = ""
    if result.config.dataset_type == "manifest":
        data_revision, split_manifest_path = try_manifest_revision(result.config.manifest_path)
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(result.config.seed),
        config=asdict(result.config),
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    # 原始数据
    raw_data = {
        "meta": meta.to_dict(),
        "budget_mode": result.config.budget_mode,
        "budgets": [r.budget for r in result.budget_results],
        "nlg_scores": [(r.frame_f1_mean, r.frame_f1_std) for r in result.budget_results],
        "text_scores": {
            "bleu": [(r.bleu_mean, r.bleu_std) for r in result.budget_results],
            "rouge1": [(r.rouge1_mean, r.rouge1_std) for r in result.budget_results],
            "rouge2": [(r.rouge2_mean, r.rouge2_std) for r in result.budget_results],
            "rougeL": [(r.rougeL_mean, r.rougeL_std) for r in result.budget_results],
        },
        "iou_scores": [(r.iou_mean, r.iou_std) for r in result.budget_results],
        "dice_scores": [(r.dice_mean, r.dice_std) for r in result.budget_results],
        "hit_rates": [r.hit_rate for r in result.budget_results],
        "combined_scores": [r.combined_score for r in result.budget_results],
        "efficiency": {
            "tokens_used": [r.avg_tokens_used for r in result.budget_results],
            "steps_used": [r.avg_steps for r in result.budget_results],
        },
        "compute": {
            "b_gen": int(result.config.b_gen),
            "n_verify": int(result.config.n_verify),
            "costs_json": str(result.config.costs_json),
            "flops_total_target": [r.flops_total_target for r in result.budget_results],
            "flops_total_mean": [r.flops_total_mean for r in result.budget_results],
            "flops_total_std": [r.flops_total_std for r in result.budget_results],
        },
        "latency": {
            "warm_mean_s": [r.warm_mean_s for r in result.budget_results],
            "warm_p95_s": [r.warm_p95_s for r in result.budget_results],
            "cold_mean_s": [r.cold_mean_s for r in result.budget_results],
            "cold_p95_s": [r.cold_p95_s for r in result.budget_results],
        },
        "samples": {
            "frame_f1": [r.frame_f1_samples for r in result.budget_results],
            "bleu": [r.bleu_samples for r in result.budget_results],
            "rouge1": [r.rouge1_samples for r in result.budget_results],
            "rouge2": [r.rouge2_samples for r in result.budget_results],
            "rougeL": [r.rougeL_samples for r in result.budget_results],
            "iou": [r.iou_samples for r in result.budget_results],
            "dice": [r.dice_samples for r in result.budget_results],
            "hit_rate": [r.hit_samples for r in result.budget_results],
            "tokens_used": [r.tokens_used_samples for r in result.budget_results],
            "steps_used": [r.steps_used_samples for r in result.budget_results],
            "flops_total": [r.flops_total_samples for r in result.budget_results],
            "warm_time_s": [r.warm_times_s for r in result.budget_results],
        },
    }
    save_results_json(raw_data, os.path.join(output_dir, "fig2_raw_data.json"))

    # Scaling fit
    fit_data = {
        "meta": meta.to_dict(),
        "nlg": {
            "model_type": result.nlg_fit.model_type,
            "params": result.nlg_fit.params,
            "r_squared": result.nlg_fit.r_squared,
            "aic": result.nlg_fit.aic,
            "bic": result.nlg_fit.bic,
            "diminishing_point": result.nlg_diminishing_point,
        },
        "grounding": {
            "model_type": result.grounding_fit.model_type,
            "params": result.grounding_fit.params,
            "r_squared": result.grounding_fit.r_squared,
            "aic": result.grounding_fit.aic,
            "bic": result.grounding_fit.bic,
            "diminishing_point": result.grounding_diminishing_point,
        },
        "combined": {
            "model_type": result.combined_fit.model_type,
            "params": result.combined_fit.params,
            "r_squared": result.combined_fit.r_squared,
            "aic": result.combined_fit.aic,
            "bic": result.combined_fit.bic,
            "diminishing_point": result.combined_diminishing_point,
        },
    }
    save_results_json(fit_data, os.path.join(output_dir, "fig2_scaling_fit.json"))


def plot_scaling_curves(
    result: ScalingExperimentResult,
    output_path: str = None,
    show: bool = True,
):
    """绘制 Scaling Law 曲线

    Args:
        result: 实验结果
        output_path: 输出文件路径
        show: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    budgets = np.array([r.budget for r in result.budget_results])
    nlg_scores = np.array([r.frame_f1_mean for r in result.budget_results])
    nlg_stds = np.array([r.frame_f1_std for r in result.budget_results])
    iou_scores = np.array([r.iou_mean for r in result.budget_results])
    iou_stds = np.array([r.iou_std for r in result.budget_results])
    combined_scores = np.array([r.combined_score for r in result.budget_results])

    # 拟合曲线的点
    fit_budgets = np.linspace(budgets.min(), budgets.max() * 1.2, 100)

    # Plot 1: NLG (Frame F1)
    ax = axes[0]
    ax.errorbar(budgets, nlg_scores, yerr=nlg_stds, fmt='o', capsize=5, label='Data')
    ax.plot(fit_budgets, [result.nlg_fit.predict(b) for b in fit_budgets],
            'r-', label=f'{result.nlg_fit.model_type} fit')
    ax.axvline(result.nlg_diminishing_point, color='g', linestyle='--',
               label=f'DR point: {result.nlg_diminishing_point:.0f}')
    ax.set_xlabel('Budget (tokens)')
    ax.set_ylabel('Frame F1')
    ax.set_title('NLG Performance Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Grounding (IoU)
    ax = axes[1]
    ax.errorbar(budgets, iou_scores, yerr=iou_stds, fmt='o', capsize=5, label='Data')
    ax.plot(fit_budgets, [result.grounding_fit.predict(b) for b in fit_budgets],
            'r-', label=f'{result.grounding_fit.model_type} fit')
    ax.axvline(result.grounding_diminishing_point, color='g', linestyle='--',
               label=f'DR point: {result.grounding_diminishing_point:.0f}')
    ax.set_xlabel('Budget (tokens)')
    ax.set_ylabel('IoU')
    ax.set_title('Grounding Performance Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Combined
    ax = axes[2]
    ax.plot(budgets, combined_scores, 'o-', label='Data')
    ax.plot(fit_budgets, [result.combined_fit.predict(b) for b in fit_budgets],
            'r-', label=f'{result.combined_fit.model_type} fit')
    ax.axvline(result.combined_diminishing_point, color='g', linestyle='--',
               label=f'DR point: {result.combined_diminishing_point:.0f}')
    ax.set_xlabel('Budget (tokens)')
    ax.set_ylabel('Combined Score')
    ax.set_title('Combined Performance Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved scaling curves to {output_path}")

    if show:
        plt.show()

    plt.close()


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Run Fig 2 Scaling Law Experiment")
    parser.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest"])
    parser.add_argument("--manifest", type=str, default="", help="Manifest path when dataset-type=manifest")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64], help="Resize (D,H,W) for manifest volumes")
    parser.add_argument("--pcg", type=str, default="toy", choices=["toy", "llama2"], help="PCG backend")
    parser.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    parser.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])
    parser.add_argument("--pcg-refresh-period", type=int, default=1, help="Refresh PCG every N refine steps (LLM mode)")
    parser.add_argument("--encoder", type=str, default="toy", choices=["toy", "cnn3d"], help="3D encoder backend")
    parser.add_argument("--encoder-device", type=str, default="cuda", help="Device for encoder (e.g. cuda)")
    parser.add_argument("--no-evidence-head", action="store_true", help="Disable EvidenceHead and use the simple allocator.")
    parser.add_argument("--require-full-budget", action="store_true", help="Avoid early-stop on no-issues/epsilon; spend budget when possible.")
    parser.add_argument("--lambda-uncertainty", type=float, default=0.3, help="Uncertainty weight in EvidenceHead Δ(c).")
    parser.add_argument("--max-steps", type=int, default=20, help="Max refine iterations per sample")
    parser.add_argument("--budget-mode", type=str, default="tokens", choices=["tokens", "flops"], help="Interpret --budgets as B_enc tokens or total FLOPs budgets.")
    parser.add_argument("--costs-json", type=str, default="", help="Optional JSON with ComputeUnitCosts (see scripts/profile_flops.py --out-costs).")
    parser.add_argument("--b-gen", type=int, default=128, help="Decoder token budget for matched accounting (toy unit-cost model).")
    parser.add_argument("--n-verify", type=int, default=1, help="Verifier call count for matched accounting (toy unit-cost model).")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[8, 16, 32, 64, 128],
                        help="Budget values to test")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of samples per budget")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./outputs/fig2_scaling",
                        help="Output directory")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")

    args = parser.parse_args()

    config = ScalingExperimentConfig(
        budgets=args.budgets,
        n_samples=args.n_samples,
        max_steps=int(args.max_steps),
        seed=args.seed,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        manifest_path=args.manifest,
        split=args.split,
        resize_shape=tuple(args.resize_shape),
        pcg_backend=args.pcg,
        llama2_path=args.llama2_path,
        llama2_quant=args.llama2_quant,
        pcg_refresh_period=int(args.pcg_refresh_period),
        encoder_backend=args.encoder,
        encoder_device=args.encoder_device,
        use_evidence_head=not bool(args.no_evidence_head),
        require_full_budget=bool(args.require_full_budget),
        lambda_uncertainty=float(args.lambda_uncertainty),
        budget_mode=str(args.budget_mode),
        costs_json=str(args.costs_json),
        b_gen=int(args.b_gen),
        n_verify=int(args.n_verify),
    )

    result = run_scaling_experiment(config=config, verbose=True)

    if not args.no_plot:
        plot_scaling_curves(
            result,
            output_path=os.path.join(args.output_dir, "fig2_scaling_curve.png"),
            show=False,
        )

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
