"""BET Refine Loop: 预算约束下的证据 Tokenization 迭代优化

根据 proposal Algorithm 1: Deterministic Greedy BET-Refine
1. 构建/更新 evidence graph
2. 对每个可 split cell 计算 Δ(c)
3. 选择 c* = argmax Δ(c)，tie-break: 最小 cell_id
4. 若 Δ(c*) < ε 或 |S| >= B_enc: 停止
5. split c* 为 8 children
6. 每 M 步运行一次 PCG+verifier 刷新 issues
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import torch

from ..grid.cells import root_cell, split, Cell
from ..types import Token, Generation, Issue
from .tokenize import encode_tokens
from .allocator import pick_cell_to_split
from .evidence_head import (
    EvidenceHead,
    EvidenceScore,
    compute_delta,
    rank_cells_by_delta,
    select_cell_to_split_by_delta,
)


@dataclass
class RefineTrace:
    """单步 refine 的详细跟踪信息"""
    step: int
    num_cells: int
    num_tokens: int
    num_frames: int
    num_issues: int
    issues: List[Dict[str, Any]]
    split_cell: Optional[str] = None
    delta: Optional[float] = None
    delta_breakdown: Optional[Dict[str, float]] = None


@dataclass
class RefineResult:
    """Refine loop 的完整结果"""
    tokens: List[Token]
    gen: Generation
    issues: List[Issue]
    trace: List[RefineTrace]
    # 额外的可审计信息
    final_cells: List[Cell] = field(default_factory=list)
    total_steps: int = 0
    stopped_reason: str = ""


def run_refine_loop(
    volume: torch.Tensor,
    budget_tokens: int,
    steps: int,
    generator_fn: Callable[[List[Token]], Generation],
    verifier_fn: Callable[[Generation, List[Token]], List[Issue]],
    emb_dim: int = 32,
    seed: int = 0,
    # Evidence Head 相关参数
    evidence_head: Optional[EvidenceHead] = None,
    use_evidence_head: bool = True,
    epsilon: float = 0.01,
    max_depth: int = 4,
    lambda_uncertainty: float = 0.3,
    verifier_refresh_period: int = 1,  # M: 每 M 步刷新 verifier
) -> RefineResult:
    """运行 BET refine loop

    Args:
        volume: 3D CT volume tensor
        budget_tokens: 最大 token 数量 (B_enc)
        steps: 最大迭代步数
        generator_fn: PCG 生成函数
        verifier_fn: Verifier 验证函数
        emb_dim: Token embedding 维度
        seed: 随机种子
        evidence_head: Evidence Head 模型（可选，如果 None 则使用原始 allocator）
        use_evidence_head: 是否使用 Evidence Head 选择 cell
        epsilon: 停机阈值（Δ < ε 时停止）
        max_depth: Cell 最大深度
        lambda_uncertainty: 不确定性项的权重
        verifier_refresh_period: Verifier 刷新周期

    Returns:
        RefineResult 包含 tokens, generation, issues 和详细 trace
    """
    cells: List[Cell] = [root_cell()]
    trace: List[RefineTrace] = []
    issues: List[Issue] = []
    gen: Generation = None  # type: ignore
    stopped_reason = "max_steps"

    # 如果使用 Evidence Head 但未提供，则创建默认的
    if use_evidence_head and evidence_head is None:
        evidence_head = EvidenceHead(
            emb_dim=emb_dim,
            lambda_uncertainty=lambda_uncertainty,
        )

    for step in range(steps):
        # 1. 编码当前 cells 为 tokens
        tokens = encode_tokens(volume, cells, emb_dim=emb_dim, seed=seed)

        # 2. 运行 PCG 生成
        gen = generator_fn(tokens)

        # 3. 运行 Verifier（根据 refresh period）
        should_refresh_verifier = (step % verifier_refresh_period == 0) or (step == 0)
        if should_refresh_verifier:
            issues = verifier_fn(gen, tokens)

        # 4. 记录 trace
        trace_entry = RefineTrace(
            step=step,
            num_cells=len(cells),
            num_tokens=len(tokens),
            num_frames=len(gen.frames),
            num_issues=len(issues),
            issues=[
                dict(
                    frame_idx=i.frame_idx,
                    type=i.issue_type,  # IssueType is already a string Literal
                    rule=i.rule_id,
                    severity=i.severity,
                )
                for i in issues
            ],
        )

        # 5. 检查停止条件
        if len(tokens) >= budget_tokens:
            stopped_reason = "budget_reached"
            trace.append(trace_entry)
            break

        if not issues:
            stopped_reason = "no_issues"
            trace.append(trace_entry)
            break

        # 6. 选择要 split 的 cell
        if use_evidence_head and evidence_head is not None:
            # 使用 Evidence Head 计算 Δ(c)
            cell_embeddings = {t.cell_id: t.embedding for t in tokens}

            ranked_scores = rank_cells_by_delta(
                cells=cells,
                cell_embeddings=cell_embeddings,
                evidence_head=evidence_head,
                current_issues=issues,
                max_depth=max_depth,
                epsilon=epsilon,
            )

            if not ranked_scores:
                stopped_reason = "no_splittable_cells"
                trace.append(trace_entry)
                break

            best_score = ranked_scores[0]
            c_star = best_score.cell

            # 检查 Δ < ε
            if best_score.delta < epsilon:
                stopped_reason = f"delta_below_epsilon (Δ={best_score.delta:.4f} < ε={epsilon})"
                trace.append(trace_entry)
                break

            # 记录 Δ 信息
            trace_entry.split_cell = c_star.id()
            trace_entry.delta = best_score.delta
            trace_entry.delta_breakdown = {
                "issue_reduction": best_score.issue_reduction,
                "uncertainty": best_score.uncertainty,
            }
        else:
            # 使用原始 allocator
            c_star = pick_cell_to_split(cells, tokens, issues)
            if c_star is not None:
                trace_entry.split_cell = c_star.id()

        trace.append(trace_entry)

        if c_star is None:
            stopped_reason = "no_cell_to_split"
            break

        # 7. Split 选中的 cell
        cells = [c for c in cells if c.id() != c_star.id()] + split(c_star)

    # 最终编码和验证
    tokens = encode_tokens(volume, cells, emb_dim=emb_dim, seed=seed)
    gen = generator_fn(tokens)
    issues = verifier_fn(gen, tokens)

    return RefineResult(
        tokens=tokens,
        gen=gen,
        issues=issues,
        trace=trace,
        final_cells=cells,
        total_steps=len(trace),
        stopped_reason=stopped_reason,
    )


def run_refine_loop_with_grounding(
    volume: torch.Tensor,
    budget_tokens: int,
    steps: int,
    generator_fn: Callable[[List[Token]], Generation],
    verifier_fn: Callable[[Generation, List[Token]], List[Issue]],
    lesion_masks: Optional[Dict[int, Any]] = None,
    emb_dim: int = 32,
    seed: int = 0,
    evidence_head: Optional[EvidenceHead] = None,
    epsilon: float = 0.01,
    max_depth: int = 4,
) -> RefineResult:
    """带 grounding 监督的 refine loop（M2 阶段）

    Args:
        lesion_masks: frame_idx -> 3D lesion mask，用于 grounding consistency
        其他参数同 run_refine_loop

    Returns:
        RefineResult，trace 中包含 grounding 相关信息
    """
    # 基础 refine loop
    result = run_refine_loop(
        volume=volume,
        budget_tokens=budget_tokens,
        steps=steps,
        generator_fn=generator_fn,
        verifier_fn=verifier_fn,
        emb_dim=emb_dim,
        seed=seed,
        evidence_head=evidence_head,
        use_evidence_head=True,
        epsilon=epsilon,
        max_depth=max_depth,
    )

    # 如果有 lesion masks，计算 grounding 信息
    if lesion_masks is not None:
        from ..eval.metrics_grounding import compute_citation_grounding

        volume_shape = tuple(volume.shape)
        grounding_scores = []

        for frame_idx, cites in result.gen.citations.items():
            if frame_idx in lesion_masks:
                score = compute_citation_grounding(
                    citations=cites,
                    tokens=result.tokens,
                    lesion_mask=lesion_masks[frame_idx],
                    volume_shape=volume_shape,
                )
                grounding_scores.append(score)

        # 添加到 result（可以扩展 RefineResult 或使用 trace）
        if result.trace:
            result.trace[-1].delta_breakdown = result.trace[-1].delta_breakdown or {}
            result.trace[-1].delta_breakdown["grounding_iou"] = (
                sum(s["iou_union"] for s in grounding_scores) / len(grounding_scores)
                if grounding_scores else 0.0
            )

    return result


# 向后兼容的简化接口
def refine_loop_simple(
    volume: torch.Tensor,
    budget_tokens: int,
    steps: int,
    generator_fn: Callable[[List[Token]], Generation],
    verifier_fn: Callable[[Generation, List[Token]], List[Issue]],
    emb_dim: int = 32,
    seed: int = 0,
) -> RefineResult:
    """简化接口，不使用 Evidence Head"""
    return run_refine_loop(
        volume=volume,
        budget_tokens=budget_tokens,
        steps=steps,
        generator_fn=generator_fn,
        verifier_fn=verifier_fn,
        emb_dim=emb_dim,
        seed=seed,
        use_evidence_head=False,
    )
