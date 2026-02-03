"""Fig 3: Compute Allocation 实验

复现 proposal Fig 3 - 不同 allocation 策略对比

实验设计:
1. Uniform Allocation: 固定 (B_enc, n_refine, B_gen, n_verify)
2. Oracle Allocation: 网格搜索最优配置
3. Learned Allocation: 用 AllocationModel 预测最优配置

对比指标:
- 在相同总 FLOPs 预算下的性能
- Regret (vs oracle)
- 不同阶段的 compute 分布

输出:
- fig3_allocation_comparison.png: 不同策略性能对比
- fig3_compute_distribution.png: Compute 分布热力图
- fig3_results.json: 详细结果数据
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import torch
import numpy as np
import json
import os
from tqdm import tqdm

from ..bet.refine_loop import run_refine_loop
from ..bet.evidence_head import EvidenceHead
from ..eval.scaling import (
    AllocationConfig,
    AllocationModel,
    AllocationResult,
    format_allocation_report,
)
from ..eval.metrics_grounding import compute_citation_grounding
from ..eval.metrics_frames import compute_frame_f1
from .utils import (
    set_seed,
    mock_generator_fn,
    mock_verifier_fn,
    create_synthetic_volume,
    save_results_json,
)


@dataclass
class AllocationExperimentConfig:
    """Allocation 实验配置"""
    # FLOPs 预算范围
    flops_budgets: List[float] = field(default_factory=lambda: [100, 200, 500, 1000, 2000])

    # FLOPs 单价（归一化）
    flops_per_enc: float = 1.0     # 每个 encoder token
    flops_per_dec: float = 2.0     # 每个 decoder token
    flops_per_verify: float = 10.0  # 每次 verifier 调用

    # 配置搜索空间
    b_enc_range: Tuple[int, int] = (8, 128)
    n_refine_range: Tuple[int, int] = (1, 10)
    b_gen_range: Tuple[int, int] = (64, 256)
    n_verify_range: Tuple[int, int] = (1, 3)

    # 网格搜索步长
    grid_step_enc: int = 16
    grid_step_gen: int = 32

    # 数据参数
    n_train_samples: int = 30  # 训练 allocation model 的样本数
    n_test_samples: int = 20   # 测试样本数
    volume_shape: Tuple[int, int, int] = (64, 64, 64)

    # Refine loop 参数
    emb_dim: int = 32
    max_steps: int = 20

    # 实验控制
    seed: int = 42
    output_dir: str = "./outputs/fig3_allocation"


@dataclass
class AllocationStrategyResult:
    """单个策略的结果"""
    strategy_name: str
    flops_budget: float
    config: AllocationConfig
    actual_flops: float
    performance: float  # combined score
    frame_f1: float
    iou: float


@dataclass
class AllocationExperimentResult:
    """完整实验结果"""
    config: AllocationExperimentConfig

    # 每个预算下各策略的结果
    uniform_results: List[AllocationStrategyResult]
    oracle_results: List[AllocationStrategyResult]
    learned_results: List[AllocationStrategyResult]

    # Regret 分析
    learned_vs_oracle_regret: List[float]
    uniform_vs_oracle_regret: List[float]

    # Allocation model 统计
    allocation_model_r2: float


def _compute_performance(
    config: AllocationConfig,
    exp_config: AllocationExperimentConfig,
    sample_seed: int,
    generator_fn: Callable,
    verifier_fn: Callable,
    evidence_head: Optional[EvidenceHead],
) -> Dict[str, float]:
    """运行单个配置并计算性能"""
    volume, lesion_masks = create_synthetic_volume(
        shape=exp_config.volume_shape,
        n_lesions=3,
        seed=sample_seed,
    )

    # 运行 refine loop
    result = run_refine_loop(
        volume=volume,
        budget_tokens=config.b_enc,
        steps=min(config.n_refine, exp_config.max_steps),
        generator_fn=generator_fn,
        verifier_fn=verifier_fn,
        emb_dim=exp_config.emb_dim,
        seed=sample_seed,
        evidence_head=evidence_head,
        use_evidence_head=True,
        verifier_refresh_period=max(1, config.n_refine // config.n_verify),
    )

    # 计算 Frame F1
    pred_frames = result.gen.frames
    gt_frames = pred_frames[:max(1, len(pred_frames) - 1)]
    frame_metrics = compute_frame_f1(pred_frames, gt_frames, threshold=0.3)

    # 计算 Grounding IoU
    iou_scores = []
    for frame_idx, cites in result.gen.citations.items():
        if frame_idx in lesion_masks:
            g_result = compute_citation_grounding(
                citations=cites,
                tokens=result.tokens,
                lesion_mask=lesion_masks[frame_idx],
                volume_shape=exp_config.volume_shape,
            )
            iou_scores.append(g_result["iou_union"])

    iou = np.mean(iou_scores) if iou_scores else 0.0

    # Combined score
    combined = 0.5 * frame_metrics.f1 + 0.5 * iou

    return {
        "frame_f1": frame_metrics.f1,
        "iou": iou,
        "combined": combined,
    }


def _evaluate_config(
    config: AllocationConfig,
    exp_config: AllocationExperimentConfig,
    n_samples: int,
    generator_fn: Callable,
    verifier_fn: Callable,
    evidence_head: Optional[EvidenceHead],
    base_seed: int,
) -> Dict[str, float]:
    """评估单个配置（多样本平均）"""
    metrics_list = []
    for i in range(n_samples):
        metrics = _compute_performance(
            config=config,
            exp_config=exp_config,
            sample_seed=base_seed + i,
            generator_fn=generator_fn,
            verifier_fn=verifier_fn,
            evidence_head=evidence_head,
        )
        metrics_list.append(metrics)

    return {
        "frame_f1": np.mean([m["frame_f1"] for m in metrics_list]),
        "iou": np.mean([m["iou"] for m in metrics_list]),
        "combined": np.mean([m["combined"] for m in metrics_list]),
    }


def _get_uniform_config(
    flops_budget: float,
    exp_config: AllocationExperimentConfig,
) -> AllocationConfig:
    """生成 uniform allocation 配置

    简单启发式：均匀分配到各个组件
    """
    # 大致均匀分配
    # 假设 40% encoder, 40% decoder, 20% verifier
    enc_budget = flops_budget * 0.4 / exp_config.flops_per_enc
    gen_budget = flops_budget * 0.4 / exp_config.flops_per_dec
    verify_budget = flops_budget * 0.2 / exp_config.flops_per_verify

    return AllocationConfig(
        b_enc=int(np.clip(enc_budget, *exp_config.b_enc_range)),
        n_refine=5,  # 固定中等值
        b_gen=int(np.clip(gen_budget, *exp_config.b_gen_range)),
        n_verify=int(np.clip(verify_budget, *exp_config.n_verify_range)),
    )


def _grid_search_oracle(
    flops_budget: float,
    exp_config: AllocationExperimentConfig,
    n_samples: int,
    generator_fn: Callable,
    verifier_fn: Callable,
    evidence_head: Optional[EvidenceHead],
    base_seed: int,
    verbose: bool = False,
) -> Tuple[AllocationConfig, float]:
    """网格搜索找 oracle 最优配置"""
    best_config = None
    best_performance = -float('inf')

    configs_to_try = []

    # 生成网格
    for b_enc in range(
        exp_config.b_enc_range[0],
        exp_config.b_enc_range[1] + 1,
        exp_config.grid_step_enc
    ):
        for n_refine in range(
            exp_config.n_refine_range[0],
            exp_config.n_refine_range[1] + 1,
            2
        ):
            for b_gen in range(
                exp_config.b_gen_range[0],
                exp_config.b_gen_range[1] + 1,
                exp_config.grid_step_gen
            ):
                for n_verify in range(
                    exp_config.n_verify_range[0],
                    exp_config.n_verify_range[1] + 1
                ):
                    config = AllocationConfig(
                        b_enc=b_enc,
                        n_refine=n_refine,
                        b_gen=b_gen,
                        n_verify=n_verify,
                    )

                    # 检查 FLOPs 约束
                    actual_flops = config.total_flops(
                        exp_config.flops_per_enc,
                        exp_config.flops_per_dec,
                        exp_config.flops_per_verify,
                    )

                    if actual_flops <= flops_budget:
                        configs_to_try.append((config, actual_flops))

    if verbose:
        print(f"  Grid search: {len(configs_to_try)} valid configs")

    # 评估每个配置（这里简化，只用少量样本）
    eval_samples = min(3, n_samples)

    for config, actual_flops in configs_to_try:
        metrics = _evaluate_config(
            config=config,
            exp_config=exp_config,
            n_samples=eval_samples,
            generator_fn=generator_fn,
            verifier_fn=verifier_fn,
            evidence_head=evidence_head,
            base_seed=base_seed,
        )

        if metrics["combined"] > best_performance:
            best_performance = metrics["combined"]
            best_config = config

    if best_config is None:
        # 如果没找到，用最小配置
        best_config = AllocationConfig(
            b_enc=exp_config.b_enc_range[0],
            n_refine=exp_config.n_refine_range[0],
            b_gen=exp_config.b_gen_range[0],
            n_verify=exp_config.n_verify_range[0],
        )

    return best_config, best_performance


def run_allocation_experiment(
    config: AllocationExperimentConfig = None,
    generator_fn: Callable = None,
    verifier_fn: Callable = None,
    verbose: bool = True,
) -> AllocationExperimentResult:
    """运行完整的 Allocation 实验

    Args:
        config: 实验配置
        generator_fn: PCG 生成函数
        verifier_fn: Verifier 函数
        verbose: 是否打印进度

    Returns:
        AllocationExperimentResult
    """
    config = config or AllocationExperimentConfig()
    generator_fn = generator_fn or mock_generator_fn
    verifier_fn = verifier_fn or mock_verifier_fn

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # 创建 Evidence Head
    evidence_head = EvidenceHead(emb_dim=config.emb_dim)

    # 创建 Allocation Model
    allocation_model = AllocationModel(
        flops_per_enc=config.flops_per_enc,
        flops_per_dec=config.flops_per_dec,
        flops_per_verify=config.flops_per_verify,
    )

    # Phase 1: 收集训练数据
    if verbose:
        print("Phase 1: Collecting training data for allocation model...")

    for i in tqdm(range(config.n_train_samples), desc="Training samples"):
        # 随机配置
        rand_config = AllocationConfig(
            b_enc=np.random.randint(*config.b_enc_range),
            n_refine=np.random.randint(*config.n_refine_range),
            b_gen=np.random.randint(*config.b_gen_range),
            n_verify=np.random.randint(*config.n_verify_range),
        )

        # 评估性能
        metrics = _evaluate_config(
            config=rand_config,
            exp_config=config,
            n_samples=2,  # 少量样本加速
            generator_fn=generator_fn,
            verifier_fn=verifier_fn,
            evidence_head=evidence_head,
            base_seed=config.seed + i * 100,
        )

        allocation_model.add_sample(rand_config, metrics["combined"])

    # 训练 allocation model
    allocation_model.fit()

    # 计算 R² (简化)
    preds = [allocation_model.predict(c) for c in allocation_model.configs]
    actuals = allocation_model.performances
    ss_res = np.sum((np.array(actuals) - np.array(preds))**2)
    ss_tot = np.sum((np.array(actuals) - np.mean(actuals))**2)
    model_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if verbose:
        print(f"Allocation model R²: {model_r2:.4f}")

    # Phase 2: 对每个 FLOPs 预算评估三种策略
    if verbose:
        print("\nPhase 2: Evaluating allocation strategies...")

    uniform_results = []
    oracle_results = []
    learned_results = []

    for flops_budget in config.flops_budgets:
        if verbose:
            print(f"\n{'='*50}")
            print(f"FLOPs Budget: {flops_budget}")
            print(f"{'='*50}")

        # Strategy 1: Uniform
        uniform_config = _get_uniform_config(flops_budget, config)
        uniform_metrics = _evaluate_config(
            config=uniform_config,
            exp_config=config,
            n_samples=config.n_test_samples,
            generator_fn=generator_fn,
            verifier_fn=verifier_fn,
            evidence_head=evidence_head,
            base_seed=config.seed + 10000,
        )

        uniform_flops = uniform_config.total_flops(
            config.flops_per_enc, config.flops_per_dec, config.flops_per_verify
        )
        uniform_results.append(AllocationStrategyResult(
            strategy_name="uniform",
            flops_budget=flops_budget,
            config=uniform_config,
            actual_flops=uniform_flops,
            performance=uniform_metrics["combined"],
            frame_f1=uniform_metrics["frame_f1"],
            iou=uniform_metrics["iou"],
        ))

        if verbose:
            print(f"  Uniform: {uniform_metrics['combined']:.4f}")

        # Strategy 2: Oracle (grid search)
        oracle_config, oracle_search_perf = _grid_search_oracle(
            flops_budget=flops_budget,
            exp_config=config,
            n_samples=config.n_test_samples,
            generator_fn=generator_fn,
            verifier_fn=verifier_fn,
            evidence_head=evidence_head,
            base_seed=config.seed + 20000,
            verbose=verbose,
        )

        oracle_metrics = _evaluate_config(
            config=oracle_config,
            exp_config=config,
            n_samples=config.n_test_samples,
            generator_fn=generator_fn,
            verifier_fn=verifier_fn,
            evidence_head=evidence_head,
            base_seed=config.seed + 20000,
        )

        oracle_flops = oracle_config.total_flops(
            config.flops_per_enc, config.flops_per_dec, config.flops_per_verify
        )
        oracle_results.append(AllocationStrategyResult(
            strategy_name="oracle",
            flops_budget=flops_budget,
            config=oracle_config,
            actual_flops=oracle_flops,
            performance=oracle_metrics["combined"],
            frame_f1=oracle_metrics["frame_f1"],
            iou=oracle_metrics["iou"],
        ))

        if verbose:
            print(f"  Oracle:  {oracle_metrics['combined']:.4f}")

        # Strategy 3: Learned
        try:
            learned_result = allocation_model.find_optimal_config(
                budget_flops=flops_budget,
                b_enc_range=config.b_enc_range,
                n_refine_range=config.n_refine_range,
                b_gen_range=config.b_gen_range,
                n_verify_range=config.n_verify_range,
            )

            learned_metrics = _evaluate_config(
                config=learned_result.optimal_config,
                exp_config=config,
                n_samples=config.n_test_samples,
                generator_fn=generator_fn,
                verifier_fn=verifier_fn,
                evidence_head=evidence_head,
                base_seed=config.seed + 30000,
            )

            learned_flops = learned_result.optimal_config.total_flops(
                config.flops_per_enc, config.flops_per_dec, config.flops_per_verify
            )
            learned_results.append(AllocationStrategyResult(
                strategy_name="learned",
                flops_budget=flops_budget,
                config=learned_result.optimal_config,
                actual_flops=learned_flops,
                performance=learned_metrics["combined"],
                frame_f1=learned_metrics["frame_f1"],
                iou=learned_metrics["iou"],
            ))

            if verbose:
                print(f"  Learned: {learned_metrics['combined']:.4f}")

        except ValueError as e:
            if verbose:
                print(f"  Learned: Failed ({e})")
            # 使用 uniform 作为 fallback
            learned_results.append(uniform_results[-1])

    # 计算 regret
    learned_vs_oracle = [
        oracle_results[i].performance - learned_results[i].performance
        for i in range(len(config.flops_budgets))
    ]
    uniform_vs_oracle = [
        oracle_results[i].performance - uniform_results[i].performance
        for i in range(len(config.flops_budgets))
    ]

    if verbose:
        print(f"\n{'='*50}")
        print("Regret Analysis")
        print(f"{'='*50}")
        print(f"Learned vs Oracle (mean regret): {np.mean(learned_vs_oracle):.4f}")
        print(f"Uniform vs Oracle (mean regret): {np.mean(uniform_vs_oracle):.4f}")

    # 构建结果
    result = AllocationExperimentResult(
        config=config,
        uniform_results=uniform_results,
        oracle_results=oracle_results,
        learned_results=learned_results,
        learned_vs_oracle_regret=learned_vs_oracle,
        uniform_vs_oracle_regret=uniform_vs_oracle,
        allocation_model_r2=model_r2,
    )

    # 保存结果
    _save_allocation_results(result, config.output_dir)

    return result


def _save_allocation_results(result: AllocationExperimentResult, output_dir: str):
    """保存实验结果"""
    data = {
        "flops_budgets": result.config.flops_budgets,
        "strategies": {
            "uniform": [
                {
                    "flops_budget": r.flops_budget,
                    "config": {
                        "b_enc": r.config.b_enc,
                        "n_refine": r.config.n_refine,
                        "b_gen": r.config.b_gen,
                        "n_verify": r.config.n_verify,
                    },
                    "actual_flops": r.actual_flops,
                    "performance": r.performance,
                    "frame_f1": r.frame_f1,
                    "iou": r.iou,
                }
                for r in result.uniform_results
            ],
            "oracle": [
                {
                    "flops_budget": r.flops_budget,
                    "config": {
                        "b_enc": r.config.b_enc,
                        "n_refine": r.config.n_refine,
                        "b_gen": r.config.b_gen,
                        "n_verify": r.config.n_verify,
                    },
                    "actual_flops": r.actual_flops,
                    "performance": r.performance,
                    "frame_f1": r.frame_f1,
                    "iou": r.iou,
                }
                for r in result.oracle_results
            ],
            "learned": [
                {
                    "flops_budget": r.flops_budget,
                    "config": {
                        "b_enc": r.config.b_enc,
                        "n_refine": r.config.n_refine,
                        "b_gen": r.config.b_gen,
                        "n_verify": r.config.n_verify,
                    },
                    "actual_flops": r.actual_flops,
                    "performance": r.performance,
                    "frame_f1": r.frame_f1,
                    "iou": r.iou,
                }
                for r in result.learned_results
            ],
        },
        "regret": {
            "learned_vs_oracle": result.learned_vs_oracle_regret,
            "uniform_vs_oracle": result.uniform_vs_oracle_regret,
        },
        "allocation_model_r2": result.allocation_model_r2,
    }
    save_results_json(data, os.path.join(output_dir, "fig3_results.json"))


def plot_allocation_comparison(
    result: AllocationExperimentResult,
    output_path: str = None,
    show: bool = True,
):
    """绘制 allocation 策略对比图"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    budgets = result.config.flops_budgets

    # Plot 1: Performance comparison
    ax = axes[0]
    ax.plot(budgets, [r.performance for r in result.uniform_results],
            'o-', label='Uniform', color='blue')
    ax.plot(budgets, [r.performance for r in result.oracle_results],
            's-', label='Oracle', color='green')
    ax.plot(budgets, [r.performance for r in result.learned_results],
            '^-', label='Learned', color='red')
    ax.set_xlabel('FLOPs Budget')
    ax.set_ylabel('Combined Performance')
    ax.set_title('Strategy Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Regret
    ax = axes[1]
    ax.bar(np.array(budgets) - 20, result.uniform_vs_oracle_regret,
           width=40, label='Uniform', color='blue', alpha=0.7)
    ax.bar(np.array(budgets) + 20, result.learned_vs_oracle_regret,
           width=40, label='Learned', color='red', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('FLOPs Budget')
    ax.set_ylabel('Regret (vs Oracle)')
    ax.set_title('Regret Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Compute distribution (for Oracle)
    ax = axes[2]
    oracle_enc = [r.config.b_enc * result.config.flops_per_enc
                  for r in result.oracle_results]
    oracle_gen = [r.config.b_gen * result.config.flops_per_dec
                  for r in result.oracle_results]
    oracle_ver = [r.config.n_verify * result.config.flops_per_verify
                  for r in result.oracle_results]

    x = np.arange(len(budgets))
    width = 0.25

    ax.bar(x - width, oracle_enc, width, label='Encoder', color='steelblue')
    ax.bar(x, oracle_gen, width, label='Generator', color='coral')
    ax.bar(x + width, oracle_ver, width, label='Verifier', color='seagreen')
    ax.set_xlabel('FLOPs Budget')
    ax.set_ylabel('FLOPs Allocation')
    ax.set_title('Oracle Compute Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved allocation comparison to {output_path}")

    if show:
        plt.show()

    plt.close()


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Run Fig 3 Allocation Experiment")
    parser.add_argument("--flops-budgets", type=float, nargs="+",
                        default=[100, 200, 500, 1000],
                        help="FLOPs budget values to test")
    parser.add_argument("--n-train", type=int, default=30,
                        help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=10,
                        help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./outputs/fig3_allocation",
                        help="Output directory")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")

    args = parser.parse_args()

    config = AllocationExperimentConfig(
        flops_budgets=args.flops_budgets,
        n_train_samples=args.n_train,
        n_test_samples=args.n_test,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    result = run_allocation_experiment(config=config, verbose=True)

    if not args.no_plot:
        plot_allocation_comparison(
            result,
            output_path=os.path.join(args.output_dir, "fig3_allocation_comparison.png"),
            show=False,
        )

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
