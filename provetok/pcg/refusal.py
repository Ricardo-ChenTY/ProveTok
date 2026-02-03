"""Refusal Calibration: 可校准的拒答机制

根据 proposal §4.3.5:
- 每个 frame 输出支持概率 q_k ∈ [0,1]
- 若 q_k < τ_refuse 则输出 uncertain/refuse
- τ_refuse 在开发集按约束 critical miss-rate ≤ δ 选择一次
- 在所有预算 B 上固定，防止"按预算调阈值刷表"

必须同时报告（反封嘴主表约束）：
- unsupported rate（越低越好）
- critical miss-rate（不得上升）
- refusal rate（拒答比例）
- refusal ECE / reliability（校准好坏）
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from ..types import Frame, Generation, Issue


# Critical findings（与 evidence_head.py 保持一致）
CRITICAL_FINDINGS = {
    "pneumothorax",
    "pleural_effusion",
    "effusion",  # alias
    "large_consolidation",
    "consolidation",
    "suspicious_nodule",
    "nodule",  # 可能是 suspicious
}


@dataclass
class RefusalDecision:
    """单个 frame 的拒答决策"""
    frame_idx: int
    q_k: float              # 支持概率
    should_refuse: bool     # 是否拒答
    is_critical: bool       # 是否为 critical finding
    reason: str             # 拒答原因


@dataclass
class CalibrationMetrics:
    """校准指标集合"""
    # 核心指标
    unsupported_rate: float      # verifier U1 issue rate
    critical_miss_rate: float    # critical findings 被错误拒答的比例
    refusal_rate: float          # 总拒答比例
    refusal_ece: float           # Expected Calibration Error

    # 分桶统计
    reliability_bins: Dict[str, float]  # bin -> (avg_q, avg_correct)
    critical_refusal_count: int
    total_critical_count: int

    def is_valid(self, max_miss_rate: float = 0.05) -> bool:
        """检查是否满足安全约束"""
        return self.critical_miss_rate <= max_miss_rate


class RefusalCalibrator:
    """拒答校准器

    核心功能:
    1. 根据 q_k 和 τ_refuse 决定是否拒答
    2. 在开发集上选择最优 τ_refuse
    3. 计算校准指标（ECE, reliability diagram）
    """

    def __init__(
        self,
        tau_refuse: float = 0.5,
        max_critical_miss_rate: float = 0.05,
        num_bins: int = 10,
    ):
        """
        Args:
            tau_refuse: 拒答阈值
            max_critical_miss_rate: critical miss-rate 上限 (δ)
            num_bins: ECE 计算的分桶数
        """
        self.tau_refuse = tau_refuse
        self.max_critical_miss_rate = max_critical_miss_rate
        self.num_bins = num_bins

    def should_refuse(self, q_k: float, is_critical: bool = False) -> bool:
        """决定是否拒答

        对于 critical findings，使用更低的阈值以减少漏报
        """
        if is_critical:
            # Critical findings 使用更保守的阈值
            effective_tau = self.tau_refuse * 0.7
        else:
            effective_tau = self.tau_refuse

        return q_k < effective_tau

    def decide_refusals(
        self,
        generation: Generation,
    ) -> List[RefusalDecision]:
        """对一个 Generation 的所有 frames 做拒答决策"""
        decisions = []

        for idx, frame in enumerate(generation.frames):
            q_k = generation.q.get(idx, 0.5)
            is_critical = self._is_critical_finding(frame)
            refuse = self.should_refuse(q_k, is_critical)

            reason = ""
            if refuse:
                reason = f"q_k={q_k:.3f} < tau={self.tau_refuse:.3f}"
                if is_critical:
                    reason += " (critical, conservative threshold)"

            decisions.append(RefusalDecision(
                frame_idx=idx,
                q_k=q_k,
                should_refuse=refuse,
                is_critical=is_critical,
                reason=reason,
            ))

        return decisions

    def _is_critical_finding(self, frame: Frame) -> bool:
        """判断是否为 critical finding"""
        finding = frame.finding.lower()
        return any(cf in finding for cf in CRITICAL_FINDINGS)

    def compute_calibration_metrics(
        self,
        generations: List[Generation],
        ground_truths: List[List[Frame]],
        issues_list: List[List[Issue]],
    ) -> CalibrationMetrics:
        """计算完整的校准指标

        Args:
            generations: 生成结果列表
            ground_truths: ground truth frames 列表
            issues_list: 每个 generation 的 verifier issues

        Returns:
            CalibrationMetrics
        """
        # 收集所有 (q_k, correct, is_critical) 三元组
        samples = []
        unsupported_count = 0
        total_frames = 0
        critical_refused = 0
        total_critical = 0

        for gen, gt_frames, issues in zip(generations, ground_truths, issues_list):
            # 统计 unsupported issues
            for issue in issues:
                if "unsupported" in issue.issue_type.lower():
                    unsupported_count += 1

            for idx, frame in enumerate(gen.frames):
                q_k = gen.q.get(idx, 0.5)
                is_critical = self._is_critical_finding(frame)
                refused = gen.refusal.get(idx, False)

                # 判断是否正确（简化：检查是否在 GT 中）
                correct = self._frame_in_gt(frame, gt_frames)

                samples.append({
                    "q_k": q_k,
                    "correct": correct,
                    "is_critical": is_critical,
                    "refused": refused,
                })

                total_frames += 1

                if is_critical:
                    total_critical += 1
                    if refused and correct:
                        # 错误拒答了正确的 critical finding
                        critical_refused += 1

        # 计算 ECE
        ece, bins = self._compute_ece(samples)

        # 计算各指标
        unsupported_rate = unsupported_count / max(total_frames, 1)
        critical_miss_rate = critical_refused / max(total_critical, 1)
        refusal_rate = sum(1 for s in samples if s["refused"]) / max(len(samples), 1)

        return CalibrationMetrics(
            unsupported_rate=unsupported_rate,
            critical_miss_rate=critical_miss_rate,
            refusal_rate=refusal_rate,
            refusal_ece=ece,
            reliability_bins=bins,
            critical_refusal_count=critical_refused,
            total_critical_count=total_critical,
        )

    def _frame_in_gt(self, frame: Frame, gt_frames: List[Frame]) -> bool:
        """检查 frame 是否在 ground truth 中（简化匹配）"""
        for gt in gt_frames:
            if (frame.finding == gt.finding and
                frame.polarity == gt.polarity):
                return True
        return False

    def _compute_ece(
        self,
        samples: List[Dict],
    ) -> Tuple[float, Dict[str, float]]:
        """计算 Expected Calibration Error

        ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

        Args:
            samples: list of {q_k, correct, ...}

        Returns:
            (ece, bin_stats)
        """
        if not samples:
            return 0.0, {}

        # 分桶
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_stats = {}
        total_ece = 0.0
        n = len(samples)

        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            bin_name = f"{lo:.1f}-{hi:.1f}"

            # 筛选落入该 bin 的样本
            bin_samples = [
                s for s in samples
                if lo <= s["q_k"] < hi or (i == self.num_bins - 1 and s["q_k"] == hi)
            ]

            if not bin_samples:
                bin_stats[bin_name] = {"avg_conf": 0, "avg_acc": 0, "count": 0}
                continue

            avg_conf = np.mean([s["q_k"] for s in bin_samples])
            avg_acc = np.mean([1 if s["correct"] else 0 for s in bin_samples])
            count = len(bin_samples)

            bin_stats[bin_name] = {
                "avg_conf": float(avg_conf),
                "avg_acc": float(avg_acc),
                "count": count,
            }

            # ECE 累加
            total_ece += (count / n) * abs(avg_acc - avg_conf)

        return total_ece, bin_stats

    def find_optimal_threshold(
        self,
        val_generations: List[Generation],
        val_ground_truths: List[List[Frame]],
        val_issues: List[List[Issue]],
        candidate_taus: Optional[List[float]] = None,
    ) -> Tuple[float, CalibrationMetrics]:
        """在验证集上找最优 τ_refuse

        约束: critical_miss_rate ≤ max_critical_miss_rate
        目标: 最小化 unsupported_rate

        Args:
            val_generations: 验证集生成结果
            val_ground_truths: 验证集 ground truth
            val_issues: 验证集 issues
            candidate_taus: 候选阈值列表

        Returns:
            (best_tau, best_metrics)
        """
        if candidate_taus is None:
            candidate_taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        best_tau = self.tau_refuse
        best_metrics = None
        best_unsupported = float('inf')

        for tau in candidate_taus:
            # 临时设置阈值
            old_tau = self.tau_refuse
            self.tau_refuse = tau

            # 重新计算 refusals
            updated_gens = []
            for gen in val_generations:
                decisions = self.decide_refusals(gen)
                new_refusal = {d.frame_idx: d.should_refuse for d in decisions}
                updated_gens.append(Generation(
                    frames=gen.frames,
                    citations=gen.citations,
                    q=gen.q,
                    refusal=new_refusal,
                ))

            # 计算指标
            metrics = self.compute_calibration_metrics(
                updated_gens, val_ground_truths, val_issues
            )

            # 检查约束
            if metrics.is_valid(self.max_critical_miss_rate):
                if metrics.unsupported_rate < best_unsupported:
                    best_unsupported = metrics.unsupported_rate
                    best_tau = tau
                    best_metrics = metrics

            self.tau_refuse = old_tau

        # 如果没有满足约束的，返回最保守的
        if best_metrics is None:
            self.tau_refuse = min(candidate_taus)
            best_tau = self.tau_refuse
            best_metrics = self.compute_calibration_metrics(
                val_generations, val_ground_truths, val_issues
            )

        self.tau_refuse = best_tau
        return best_tau, best_metrics


def apply_refusal_to_generation(
    generation: Generation,
    calibrator: RefusalCalibrator,
) -> Generation:
    """应用拒答决策到 Generation

    返回更新了 refusal 字段的新 Generation
    """
    decisions = calibrator.decide_refusals(generation)
    new_refusal = {d.frame_idx: d.should_refuse for d in decisions}

    return Generation(
        frames=generation.frames,
        citations=generation.citations,
        q=generation.q,
        refusal=new_refusal,
    )


def format_calibration_report(metrics: CalibrationMetrics) -> str:
    """格式化校准报告（用于日志/可视化）"""
    lines = [
        "=" * 50,
        "Refusal Calibration Report",
        "=" * 50,
        f"Unsupported Rate:     {metrics.unsupported_rate:.4f}",
        f"Critical Miss Rate:   {metrics.critical_miss_rate:.4f}",
        f"Refusal Rate:         {metrics.refusal_rate:.4f}",
        f"Refusal ECE:          {metrics.refusal_ece:.4f}",
        "",
        f"Critical Findings:    {metrics.critical_refusal_count}/{metrics.total_critical_count} refused",
        "",
        "Reliability Bins:",
    ]

    for bin_name, stats in sorted(metrics.reliability_bins.items()):
        if isinstance(stats, dict):
            lines.append(
                f"  {bin_name}: conf={stats.get('avg_conf', 0):.3f}, "
                f"acc={stats.get('avg_acc', 0):.3f}, n={stats.get('count', 0)}"
            )

    lines.append("=" * 50)
    return "\n".join(lines)
