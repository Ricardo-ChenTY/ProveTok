"""Grounding 评估指标（ReXGroundingCT）

根据 proposal §7.1:
- sentence → citation → 3D segmentation
- hit-rate：是否存在被引用 token 的 Ω 与 lesion mask 有 overlap（≥阈值）
- IoU / Dice：对 cited cells 的 union 与 lesion mask 计算
- 同时报告 max-over-citations 与 union-over-citations 两种口径
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
import torch

from ..types import Token
from ..grid.cells import Cell, cell_bounds


@dataclass
class GroundingMetrics:
    """Grounding 指标汇总"""
    # 主指标
    hit_rate: float          # 是否有任何 citation 命中 lesion
    iou_max: float           # max IoU over citations
    iou_union: float         # union of citations vs lesion
    dice_max: float          # max Dice over citations
    dice_union: float        # union of citations vs lesion

    # 详细统计
    num_samples: int
    num_hits: int
    avg_overlap_ratio: float  # 平均 overlap 体素比例


def compute_iou(
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> float:
    """计算两个 3D mask 的 IoU

    IoU = intersection / union
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_dice(
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> float:
    """计算两个 3D mask 的 Dice coefficient

    Dice = 2 * intersection / (|mask1| + |mask2|)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    if total == 0:
        return 0.0

    return float(2 * intersection / total)


def tokens_to_mask(
    tokens: List[Token],
    volume_shape: Tuple[int, int, int],
    cells: Optional[List[Cell]] = None,
) -> np.ndarray:
    """将 tokens 转换为 3D binary mask

    Args:
        tokens: Token 列表
        volume_shape: (D, H, W) volume 形状
        cells: 可选，提供 Cell 对象用于边界计算

    Returns:
        3D binary mask (D, H, W)
    """
    mask = np.zeros(volume_shape, dtype=bool)

    for token in tokens:
        # 解析 cell_id 获取边界
        cell = _parse_cell_from_id(token.cell_id)
        if cell is None:
            continue

        slices = cell_bounds(cell, volume_shape)
        mask[slices[0], slices[1], slices[2]] = True

    return mask


def _parse_cell_from_id(cell_id: str) -> Optional[Cell]:
    """从 cell_id 字符串解析 Cell 对象

    cell_id 格式: "L{level}:({ix},{iy},{iz})"
    """
    try:
        # 解析格式 "L0:(0,0,0)"
        level_part, coord_part = cell_id.split(":")
        level = int(level_part[1:])

        # 解析坐标
        coord_str = coord_part.strip("()")
        coords = [int(x) for x in coord_str.split(",")]

        return Cell(level=level, ix=coords[0], iy=coords[1], iz=coords[2])
    except Exception:
        return None


def compute_citation_grounding(
    citations: List[int],
    tokens: List[Token],
    lesion_mask: np.ndarray,
    volume_shape: Tuple[int, int, int],
    overlap_threshold: float = 0.1,
) -> Dict[str, float]:
    """计算单个样本的 citation grounding 指标

    Args:
        citations: cited token ids
        tokens: 所有 tokens
        lesion_mask: ground truth lesion segmentation (D, H, W)
        volume_shape: volume 形状
        overlap_threshold: hit 判定阈值

    Returns:
        Dict with hit, iou_max, iou_union, dice_max, dice_union
    """
    if not citations:
        return {
            "hit": 0.0,
            "iou_max": 0.0,
            "iou_union": 0.0,
            "dice_max": 0.0,
            "dice_union": 0.0,
            "overlap_ratio": 0.0,
        }

    # 获取 cited tokens
    token_map = {t.token_id: t for t in tokens}
    cited_tokens = [token_map[tid] for tid in citations if tid in token_map]

    if not cited_tokens:
        return {
            "hit": 0.0,
            "iou_max": 0.0,
            "iou_union": 0.0,
            "dice_max": 0.0,
            "dice_union": 0.0,
            "overlap_ratio": 0.0,
        }

    # 计算每个 citation 的 mask 和指标
    ious = []
    dices = []
    overlaps = []

    for token in cited_tokens:
        token_mask = tokens_to_mask([token], volume_shape)
        iou = compute_iou(token_mask, lesion_mask)
        dice = compute_dice(token_mask, lesion_mask)

        ious.append(iou)
        dices.append(dice)

        # overlap ratio
        intersection = np.logical_and(token_mask, lesion_mask).sum()
        token_volume = token_mask.sum()
        overlap_ratio = intersection / token_volume if token_volume > 0 else 0.0
        overlaps.append(overlap_ratio)

    # Union mask
    union_mask = tokens_to_mask(cited_tokens, volume_shape)
    iou_union = compute_iou(union_mask, lesion_mask)
    dice_union = compute_dice(union_mask, lesion_mask)

    # Hit rate: 是否有任何 citation 与 lesion 有足够 overlap
    hit = 1.0 if max(overlaps) >= overlap_threshold else 0.0

    return {
        "hit": hit,
        "iou_max": max(ious) if ious else 0.0,
        "iou_union": iou_union,
        "dice_max": max(dices) if dices else 0.0,
        "dice_union": dice_union,
        "overlap_ratio": np.mean(overlaps) if overlaps else 0.0,
    }


def compute_grounding_metrics(
    samples: List[Dict],
) -> GroundingMetrics:
    """聚合多个样本的 grounding 指标

    Args:
        samples: list of dicts from compute_citation_grounding

    Returns:
        GroundingMetrics
    """
    if not samples:
        return GroundingMetrics(
            hit_rate=0.0,
            iou_max=0.0,
            iou_union=0.0,
            dice_max=0.0,
            dice_union=0.0,
            num_samples=0,
            num_hits=0,
            avg_overlap_ratio=0.0,
        )

    n = len(samples)
    hits = sum(s["hit"] for s in samples)

    return GroundingMetrics(
        hit_rate=hits / n,
        iou_max=np.mean([s["iou_max"] for s in samples]),
        iou_union=np.mean([s["iou_union"] for s in samples]),
        dice_max=np.mean([s["dice_max"] for s in samples]),
        dice_union=np.mean([s["dice_union"] for s in samples]),
        num_samples=n,
        num_hits=int(hits),
        avg_overlap_ratio=np.mean([s["overlap_ratio"] for s in samples]),
    )


def compute_mask_sanity(
    old_tokens: List[Token],
    new_tokens: List[Token],
    lesion_mask: np.ndarray,
    volume_shape: Tuple[int, int, int],
) -> Dict[str, float]:
    """Mask Sanity Check (proposal §7.4)

    检验 refine 新增的 tokens 的 Ω 与 lesion mask overlap 是否上升
    证明 refine 真在"追证据"

    Args:
        old_tokens: refine 前的 tokens
        new_tokens: refine 后新增的 tokens
        lesion_mask: ground truth lesion mask
        volume_shape: volume 形状

    Returns:
        Dict with old_overlap, new_overlap, improvement
    """
    old_mask = tokens_to_mask(old_tokens, volume_shape)
    new_mask = tokens_to_mask(new_tokens, volume_shape)

    old_iou = compute_iou(old_mask, lesion_mask)
    new_iou = compute_iou(new_mask, lesion_mask)

    old_dice = compute_dice(old_mask, lesion_mask)
    new_dice = compute_dice(new_mask, lesion_mask)

    return {
        "old_iou": old_iou,
        "new_iou": new_iou,
        "iou_improvement": new_iou - old_iou,
        "old_dice": old_dice,
        "new_dice": new_dice,
        "dice_improvement": new_dice - old_dice,
    }


def format_grounding_metrics(metrics: GroundingMetrics) -> str:
    """格式化 grounding metrics 报告"""
    lines = [
        "=" * 50,
        "Grounding Metrics (ReXGroundingCT)",
        "=" * 50,
        f"Hit Rate:           {metrics.hit_rate:.4f}",
        "",
        "IoU:",
        f"  Max (per-citation): {metrics.iou_max:.4f}",
        f"  Union (all-citation): {metrics.iou_union:.4f}",
        "",
        "Dice:",
        f"  Max (per-citation): {metrics.dice_max:.4f}",
        f"  Union (all-citation): {metrics.dice_union:.4f}",
        "",
        f"Samples: {metrics.num_samples} (hits: {metrics.num_hits})",
        f"Avg Overlap Ratio: {metrics.avg_overlap_ratio:.4f}",
        "=" * 50,
    ]
    return "\n".join(lines)


# ============================================================
# Counterfactual Experiments (proposal §7.4)
# ============================================================

def omega_permutation_test(
    tokens: List[Token],
    citations: Dict[int, List[int]],
    lesion_masks: Dict[int, np.ndarray],
    volume_shape: Tuple[int, int, int],
    n_permutations: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Ω-permutation 反事实实验

    随机置换 Ω_i (cell_id)，保持 token embedding 不变
    检验 grounding/correctness 是否显著下降

    Args:
        tokens: 原始 tokens
        citations: frame_idx -> cited token_ids
        lesion_masks: frame_idx -> lesion mask
        volume_shape: volume 形状
        n_permutations: 置换次数
        seed: 随机种子

    Returns:
        Dict with original_score, permuted_mean, permuted_std, p_value
    """
    rng = np.random.RandomState(seed)

    # 计算原始 grounding 分数
    original_samples = []
    for frame_idx, cites in citations.items():
        if frame_idx in lesion_masks:
            result = compute_citation_grounding(
                cites, tokens, lesion_masks[frame_idx], volume_shape
            )
            original_samples.append(result["iou_union"])

    original_score = np.mean(original_samples) if original_samples else 0.0

    # 进行多次置换
    permuted_scores = []
    cell_ids = [t.cell_id for t in tokens]

    for _ in range(n_permutations):
        # 置换 cell_ids
        perm_indices = rng.permutation(len(cell_ids))
        permuted_cell_ids = [cell_ids[i] for i in perm_indices]

        # 创建置换后的 tokens
        permuted_tokens = []
        for i, t in enumerate(tokens):
            permuted_tokens.append(Token(
                token_id=t.token_id,
                cell_id=permuted_cell_ids[i],  # 置换后的 cell_id
                level=t.level,
                embedding=t.embedding,  # embedding 保持不变
                score=t.score,
                uncertainty=t.uncertainty,
            ))

        # 计算置换后的 grounding 分数
        perm_samples = []
        for frame_idx, cites in citations.items():
            if frame_idx in lesion_masks:
                result = compute_citation_grounding(
                    cites, permuted_tokens, lesion_masks[frame_idx], volume_shape
                )
                perm_samples.append(result["iou_union"])

        if perm_samples:
            permuted_scores.append(np.mean(perm_samples))

    # 计算统计量
    permuted_mean = np.mean(permuted_scores) if permuted_scores else 0.0
    permuted_std = np.std(permuted_scores) if permuted_scores else 0.0

    # p-value: 原始分数优于置换分数的比例
    p_value = np.mean([1 if ps >= original_score else 0 for ps in permuted_scores])

    return {
        "original_score": original_score,
        "permuted_mean": permuted_mean,
        "permuted_std": permuted_std,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def citation_swap_test(
    generations: List,  # List[Generation]
    tokens: List[Token],
    lesion_masks: Dict[int, np.ndarray],
    volume_shape: Tuple[int, int, int],
    n_swaps: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Citation-swap 反事实实验

    在同一报告内随机交换 C_k（保持 |C_k| 分布不变）
    检验 unsupported/overclaim 是否激增

    Args:
        generations: Generation 列表
        tokens: 所有 tokens
        lesion_masks: 可选的 lesion masks
        volume_shape: volume 形状
        n_swaps: 交换次数
        seed: 随机种子

    Returns:
        Dict with original metrics, swapped metrics, degradation
    """
    rng = np.random.RandomState(seed)

    # 计算原始 grounding
    original_samples = []
    for gen in generations:
        for frame_idx, cites in gen.citations.items():
            if frame_idx in lesion_masks:
                result = compute_citation_grounding(
                    cites, tokens, lesion_masks[frame_idx], volume_shape
                )
                original_samples.append(result["iou_union"])

    original_score = np.mean(original_samples) if original_samples else 0.0

    # 进行多次 citation swap
    swapped_scores = []

    for _ in range(n_swaps):
        swap_samples = []

        for gen in generations:
            # 收集所有 citations
            all_citations = list(gen.citations.values())
            if len(all_citations) < 2:
                continue

            # 随机交换 citations（保持长度分布）
            perm = rng.permutation(len(all_citations))
            swapped_citations = {
                i: all_citations[perm[i]]
                for i in range(len(all_citations))
            }

            for frame_idx, cites in swapped_citations.items():
                if frame_idx in lesion_masks:
                    result = compute_citation_grounding(
                        cites, tokens, lesion_masks[frame_idx], volume_shape
                    )
                    swap_samples.append(result["iou_union"])

        if swap_samples:
            swapped_scores.append(np.mean(swap_samples))

    swapped_mean = np.mean(swapped_scores) if swapped_scores else 0.0

    return {
        "original_score": original_score,
        "swapped_mean": swapped_mean,
        "degradation": original_score - swapped_mean,
        "degradation_ratio": (original_score - swapped_mean) / original_score if original_score > 0 else 0.0,
    }
