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
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import torch
import numpy as np
import json
import os
from tqdm import tqdm

from ..bet.refine_loop import run_refine_loop, RefineResult
from ..bet.evidence_head import EvidenceHead
from ..eval.scaling import (
    fit_scaling_law,
    compute_diminishing_returns_point,
    format_scaling_report,
    ScalingFit,
)
from ..eval.metrics_grounding import (
    compute_citation_grounding,
    compute_grounding_metrics,
)
from ..eval.metrics_frames import compute_frame_f1
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
    budgets: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])

    # Refine loop 参数
    max_steps: int = 20
    emb_dim: int = 32
    epsilon: float = 0.01
    max_depth: int = 4

    # 数据参数
    n_samples: int = 50  # 每个 budget 测试的样本数
    volume_shape: Tuple[int, int, int] = (64, 64, 64)
    n_lesions_per_sample: int = 3

    # Evidence Head
    use_evidence_head: bool = True
    lambda_uncertainty: float = 0.3

    # 实验控制
    seed: int = 42
    output_dir: str = "./outputs/fig2_scaling"

    # 指标权重（用于综合 scaling）
    nlg_weight: float = 0.5
    grounding_weight: float = 0.5


@dataclass
class SingleBudgetResult:
    """单个 budget 的实验结果"""
    budget: int
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

    # Efficiency
    avg_tokens_used: float
    avg_steps: float
    avg_stop_reason: Dict[str, int]

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
    budget: int,
    config: ScalingExperimentConfig,
    generator_fn: Callable,
    verifier_fn: Callable,
    evidence_head: Optional[EvidenceHead],
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

    iterator = range(config.n_samples)
    if verbose:
        iterator = tqdm(iterator, desc=f"Budget={budget}")

    for sample_idx in iterator:
        sample_seed = config.seed + sample_idx

        # 创建合成数据
        volume, lesion_masks = create_synthetic_volume(
            shape=config.volume_shape,
            n_lesions=config.n_lesions_per_sample,
            seed=sample_seed,
        )

        # 运行 refine loop
        result = run_refine_loop(
            volume=volume,
            budget_tokens=budget,
            steps=config.max_steps,
            generator_fn=generator_fn,
            verifier_fn=verifier_fn,
            emb_dim=config.emb_dim,
            seed=sample_seed,
            evidence_head=evidence_head,
            use_evidence_head=config.use_evidence_head,
            epsilon=config.epsilon,
            max_depth=config.max_depth,
            lambda_uncertainty=config.lambda_uncertainty,
        )

        # 计算 Frame F1（用 mock ground truth）
        # 实际使用时应该用真实 GT frames
        pred_frames = result.gen.frames
        gt_frames = pred_frames[:max(1, len(pred_frames) - 1)]  # Mock GT
        frame_metrics = compute_frame_f1(pred_frames, gt_frames, threshold=0.3)
        frame_f1_scores.append(frame_metrics.f1)

        # 计算 Grounding 指标
        grounding_samples = []
        for frame_idx, cites in result.gen.citations.items():
            if frame_idx in lesion_masks:
                g_result = compute_citation_grounding(
                    citations=cites,
                    tokens=result.tokens,
                    lesion_mask=lesion_masks[frame_idx],
                    volume_shape=config.volume_shape,
                )
                grounding_samples.append(g_result)

        if grounding_samples:
            iou_scores.append(np.mean([s["iou_union"] for s in grounding_samples]))
            dice_scores.append(np.mean([s["dice_union"] for s in grounding_samples]))
            hit_scores.append(np.mean([s["hit"] for s in grounding_samples]))
        else:
            iou_scores.append(0.0)
            dice_scores.append(0.0)
            hit_scores.append(0.0)

        # 记录效率
        tokens_used.append(len(result.tokens))
        steps_used.append(result.total_steps)

        reason = result.stopped_reason.split()[0]
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1

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
        avg_tokens_used=np.mean(tokens_used),
        avg_steps=np.mean(steps_used),
        avg_stop_reason=stop_reasons,
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
    generator_fn = generator_fn or mock_generator_fn
    verifier_fn = verifier_fn or mock_verifier_fn

    set_seed(config.seed)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 创建 Evidence Head
    evidence_head = None
    if config.use_evidence_head:
        evidence_head = EvidenceHead(
            emb_dim=config.emb_dim,
            lambda_uncertainty=config.lambda_uncertainty,
        )

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
    # 原始数据
    raw_data = {
        "budgets": [r.budget for r in result.budget_results],
        "nlg_scores": [(r.frame_f1_mean, r.frame_f1_std) for r in result.budget_results],
        "iou_scores": [(r.iou_mean, r.iou_std) for r in result.budget_results],
        "dice_scores": [(r.dice_mean, r.dice_std) for r in result.budget_results],
        "hit_rates": [r.hit_rate for r in result.budget_results],
        "combined_scores": [r.combined_score for r in result.budget_results],
        "efficiency": {
            "tokens_used": [r.avg_tokens_used for r in result.budget_results],
            "steps_used": [r.avg_steps for r in result.budget_results],
        },
    }
    save_results_json(raw_data, os.path.join(output_dir, "fig2_raw_data.json"))

    # Scaling fit
    fit_data = {
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
        seed=args.seed,
        output_dir=args.output_dir,
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
