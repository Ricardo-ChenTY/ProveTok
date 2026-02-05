"""Frame-level 评估指标

根据 proposal §7.1 Clinical correctness（结构化事实）:
- finding frame F1（含 laterality/location/negation/size bin 等槽）
- 匹配方式：按 finding type + coarse location + laterality 做 Hungarian matching
- slot-level micro/macro F1 统计
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..types import Frame


@dataclass
class FrameMatchResult:
    """单个 frame 的匹配结果"""
    pred_idx: int
    gt_idx: int
    finding_match: bool
    polarity_match: bool
    laterality_match: bool
    location_match: bool
    size_bin_match: bool
    severity_match: bool
    uncertainty_match: bool
    full_match: bool


@dataclass
class FrameMetrics:
    """Frame-level 指标汇总"""
    # Detection metrics
    precision: float
    recall: float
    f1: float

    # Slot-level metrics
    finding_accuracy: float
    polarity_accuracy: float
    laterality_accuracy: float
    location_accuracy: float
    size_bin_accuracy: float
    severity_accuracy: float
    uncertainty_accuracy: float

    # Counts
    num_pred: int
    num_gt: int
    num_matched: int

    # Per-finding breakdown
    per_finding_f1: Dict[str, float]


def compute_frame_similarity(pred: Frame, gt: Frame) -> float:
    """计算两个 frame 的相似度（用于 Hungarian matching）

    相似度 = weighted sum of slot matches
    """
    score = 0.0

    # Finding type match (highest weight)
    if pred.finding.lower() == gt.finding.lower():
        score += 0.4
    elif _is_similar_finding(pred.finding, gt.finding):
        score += 0.25

    # Coarse location match
    if pred.location == gt.location:
        score += 0.25
    elif pred.location == "unspecified" or gt.location == "unspecified":
        score += 0.1  # partial credit

    # Laterality match
    if pred.laterality == gt.laterality:
        score += 0.2
    elif pred.laterality == "unspecified" or gt.laterality == "unspecified":
        score += 0.1  # partial credit

    # Polarity match
    if pred.polarity == gt.polarity:
        score += 0.15

    return score


def _is_similar_finding(f1: str, f2: str) -> bool:
    """检查两个 finding type 是否相似（处理别名）"""
    f1, f2 = f1.lower(), f2.lower()

    # 别名映射
    aliases = {
        "effusion": {"pleural_effusion", "pleural effusion"},
        "nodule": {"lung_nodule", "pulmonary_nodule", "suspicious_nodule"},
        "atelectasis": {"collapse", "lung_collapse"},
        "consolidation": {"pneumonia", "infiltrate"},
    }

    for canonical, variants in aliases.items():
        if f1 == canonical or f1 in variants:
            if f2 == canonical or f2 in variants:
                return True

    return False


def hungarian_match_frames(
    preds: List[Frame],
    gts: List[Frame],
    threshold: float = 0.3,
) -> List[FrameMatchResult]:
    """使用 Hungarian algorithm 进行 frame 匹配

    Args:
        preds: 预测的 frames
        gts: ground truth frames
        threshold: 最小相似度阈值

    Returns:
        匹配结果列表
    """
    if not preds or not gts:
        return []

    # 构建 cost matrix (negative similarity for minimization)
    cost_matrix = np.zeros((len(preds), len(gts)))
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            cost_matrix[i, j] = -compute_frame_similarity(pred, gt)

    # Hungarian matching
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    results = []
    for pred_idx, gt_idx in zip(row_indices, col_indices):
        similarity = -cost_matrix[pred_idx, gt_idx]

        if similarity >= threshold:
            pred, gt = preds[pred_idx], gts[gt_idx]
            results.append(FrameMatchResult(
                pred_idx=pred_idx,
                gt_idx=gt_idx,
                finding_match=pred.finding.lower() == gt.finding.lower(),
                polarity_match=pred.polarity == gt.polarity,
                laterality_match=pred.laterality == gt.laterality,
                location_match=pred.location == gt.location,
                size_bin_match=pred.size_bin == gt.size_bin,
                severity_match=pred.severity == gt.severity,
                uncertainty_match=pred.uncertain == gt.uncertain,
                full_match=(
                    pred.finding.lower() == gt.finding.lower() and
                    pred.polarity == gt.polarity and
                    pred.laterality == gt.laterality and
                    pred.location == gt.location and
                    pred.size_bin == gt.size_bin and
                    pred.severity == gt.severity and
                    pred.uncertain == gt.uncertain
                ),
            ))

    return results


def frame_f1(pred: List[Frame], gt: List[Frame]) -> float:
    """计算 Frame F1（向后兼容接口）"""
    metrics = compute_frame_f1(pred, gt)
    return metrics.f1


def compute_frame_f1(
    pred_frames: List[Frame],
    gt_frames: List[Frame],
    threshold: float = 0.3,
) -> FrameMetrics:
    """计算 Frame-level F1 指标

    Args:
        pred_frames: 预测的 frames
        gt_frames: ground truth frames
        threshold: 匹配阈值

    Returns:
        FrameMetrics
    """
    matches = hungarian_match_frames(pred_frames, gt_frames, threshold)

    num_pred = len(pred_frames)
    num_gt = len(gt_frames)
    num_matched = len(matches)

    # Detection metrics
    precision = num_matched / num_pred if num_pred > 0 else 0.0
    recall = num_matched / num_gt if num_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Slot-level accuracy (among matched pairs)
    if num_matched > 0:
        finding_acc = sum(m.finding_match for m in matches) / num_matched
        polarity_acc = sum(m.polarity_match for m in matches) / num_matched
        laterality_acc = sum(m.laterality_match for m in matches) / num_matched
        location_acc = sum(m.location_match for m in matches) / num_matched
        size_acc = sum(m.size_bin_match for m in matches) / num_matched
        severity_acc = sum(m.severity_match for m in matches) / num_matched
        uncertainty_acc = sum(m.uncertainty_match for m in matches) / num_matched
    else:
        finding_acc = polarity_acc = laterality_acc = 0.0
        location_acc = size_acc = severity_acc = uncertainty_acc = 0.0

    # Per-finding F1
    per_finding_f1 = _compute_per_finding_f1(pred_frames, gt_frames, matches)

    return FrameMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        finding_accuracy=finding_acc,
        polarity_accuracy=polarity_acc,
        laterality_accuracy=laterality_acc,
        location_accuracy=location_acc,
        size_bin_accuracy=size_acc,
        severity_accuracy=severity_acc,
        uncertainty_accuracy=uncertainty_acc,
        num_pred=num_pred,
        num_gt=num_gt,
        num_matched=num_matched,
        per_finding_f1=per_finding_f1,
    )


def _compute_per_finding_f1(
    preds: List[Frame],
    gts: List[Frame],
    matches: List[FrameMatchResult],
) -> Dict[str, float]:
    """计算每个 finding type 的 F1"""
    # 收集所有 finding types
    all_findings: Set[str] = set()
    for f in preds + gts:
        all_findings.add(f.finding.lower())

    per_finding = {}

    for finding in all_findings:
        # 该 finding 的 pred 和 gt 数量
        pred_count = sum(1 for f in preds if f.finding.lower() == finding)
        gt_count = sum(1 for f in gts if f.finding.lower() == finding)

        # 该 finding 的 match 数量
        matched_pred_indices = {m.pred_idx for m in matches if m.finding_match}
        matched_count = sum(
            1 for i, f in enumerate(preds)
            if f.finding.lower() == finding and i in matched_pred_indices
        )

        # 计算 F1
        p = matched_count / pred_count if pred_count > 0 else 0.0
        r = matched_count / gt_count if gt_count > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_finding[finding] = f1

    return per_finding


def aggregate_frame_metrics(
    metrics_list: List[FrameMetrics],
) -> FrameMetrics:
    """聚合多个样本的 Frame metrics

    使用 micro averaging
    """
    if not metrics_list:
        return FrameMetrics(
            precision=0.0, recall=0.0, f1=0.0,
            finding_accuracy=0.0,
            polarity_accuracy=0.0,
            laterality_accuracy=0.0,
            location_accuracy=0.0,
            size_bin_accuracy=0.0,
            severity_accuracy=0.0,
            uncertainty_accuracy=0.0,
            num_pred=0, num_gt=0, num_matched=0,
            per_finding_f1={},
        )

    total_pred = sum(m.num_pred for m in metrics_list)
    total_gt = sum(m.num_gt for m in metrics_list)
    total_matched = sum(m.num_matched for m in metrics_list)

    # Micro precision/recall/F1
    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Weighted slot accuracy
    if total_matched > 0:
        finding_acc = sum(m.finding_accuracy * m.num_matched for m in metrics_list) / total_matched
        polarity_acc = sum(m.polarity_accuracy * m.num_matched for m in metrics_list) / total_matched
        laterality_acc = sum(m.laterality_accuracy * m.num_matched for m in metrics_list) / total_matched
        location_acc = sum(m.location_accuracy * m.num_matched for m in metrics_list) / total_matched
        size_acc = sum(m.size_bin_accuracy * m.num_matched for m in metrics_list) / total_matched
        severity_acc = sum(m.severity_accuracy * m.num_matched for m in metrics_list) / total_matched
        uncertainty_acc = sum(m.uncertainty_accuracy * m.num_matched for m in metrics_list) / total_matched
    else:
        finding_acc = polarity_acc = laterality_acc = 0.0
        location_acc = size_acc = severity_acc = uncertainty_acc = 0.0

    # Aggregate per-finding F1
    all_findings: Set[str] = set()
    for m in metrics_list:
        all_findings.update(m.per_finding_f1.keys())

    per_finding_f1 = {}
    for finding in all_findings:
        f1_values = [m.per_finding_f1.get(finding, 0.0) for m in metrics_list if finding in m.per_finding_f1]
        per_finding_f1[finding] = np.mean(f1_values) if f1_values else 0.0

    return FrameMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        finding_accuracy=finding_acc,
        polarity_accuracy=polarity_acc,
        laterality_accuracy=laterality_acc,
        location_accuracy=location_acc,
        size_bin_accuracy=size_acc,
        severity_accuracy=severity_acc,
        uncertainty_accuracy=uncertainty_acc,
        num_pred=total_pred,
        num_gt=total_gt,
        num_matched=total_matched,
        per_finding_f1=per_finding_f1,
    )


def format_frame_metrics(metrics: FrameMetrics) -> str:
    """格式化 Frame metrics 报告"""
    lines = [
        "=" * 50,
        "Frame-level Metrics",
        "=" * 50,
        f"Precision:     {metrics.precision:.4f}",
        f"Recall:        {metrics.recall:.4f}",
        f"F1:            {metrics.f1:.4f}",
        "",
        f"Pred/GT/Match: {metrics.num_pred}/{metrics.num_gt}/{metrics.num_matched}",
        "",
        "Slot Accuracy (among matches):",
        f"  Finding:     {metrics.finding_accuracy:.4f}",
        f"  Polarity:    {metrics.polarity_accuracy:.4f}",
        f"  Laterality:  {metrics.laterality_accuracy:.4f}",
        f"  Location:    {metrics.location_accuracy:.4f}",
        f"  Size Bin:    {metrics.size_bin_accuracy:.4f}",
        f"  Severity:    {metrics.severity_accuracy:.4f}",
        f"  Uncertainty: {metrics.uncertainty_accuracy:.4f}",
        "",
        "Per-Finding F1:",
    ]

    for finding, f1 in sorted(metrics.per_finding_f1.items()):
        lines.append(f"  {finding}: {f1:.4f}")

    lines.append("=" * 50)
    return "\n".join(lines)
