"""Evidence Head: 计算每个 cell 的边际证据收益 Δ(c)

根据 proposal §4.2.2:
Δ(c) = Σ_{u∈I} w_u · Δ̂_issue(u,c)  [verifier-driven]
     + λ · H(p(critical findings | c))  [uncertainty]

这个模块负责:
1. 预测每个 cell 对 verifier issues 的边际改善
2. 计算 finding-level 的不确定性熵
3. 综合得到 Δ(c) 用于 greedy allocator 决策
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..types import Token, Issue, IssueType
from ..grid.cells import Cell


@dataclass(frozen=True)
class EvidenceScore:
    """Cell 的证据评分结果"""
    cell: Cell
    delta: float              # 总体边际收益 Δ(c)
    issue_reduction: float    # verifier-driven 项
    uncertainty: float        # 不确定性项
    slot_probs: Dict[str, float]  # 各 finding type 的预测概率


# Severity 权重（根据 proposal: critical > non-critical）
SEVERITY_WEIGHTS = {
    3: 1.0,   # critical
    2: 0.5,   # non-critical major
    1: 0.2,   # non-critical minor
}

# Critical findings（根据 proposal §4.3.4）
CRITICAL_FINDINGS = {
    "pneumothorax",
    "pleural_effusion",
    "large_consolidation",
    "suspicious_nodule",
}


class EvidenceHead(nn.Module):
    """Evidence Head: 预测 cell 的证据潜力

    输入: cell 的 pooled 特征
    输出:
    - finding type 概率分布 (用于计算 uncertainty)
    - 预测的 issue 改善量 (用于 verifier-driven term)
    """

    def __init__(
        self,
        emb_dim: int = 32,
        num_findings: int = 8,
        hidden_dim: int = 64,
        lambda_uncertainty: float = 0.3,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_findings = num_findings
        self.lambda_uncertainty = lambda_uncertainty

        # Finding type 分类头
        self.finding_head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_findings),
        )

        # Issue reduction 预测头（预测 split 后 issue 减少量）
        self.issue_head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4 种 issue type: U1, O1, I1, M1
            nn.Softplus(),  # 保证非负
        )

        # Finding type 名称映射
        self.finding_names = [
            "nodule", "effusion", "atelectasis", "consolidation",
            "pneumothorax", "cardiomegaly", "mass", "normal"
        ]

    def forward(self, cell_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cell_embedding: (D,) 或 (B, D) 的 cell 特征

        Returns:
            finding_probs: (num_findings,) 或 (B, num_findings) 的概率分布
            issue_reduction: (4,) 或 (B, 4) 的预测 issue 减少量
        """
        if cell_embedding.dim() == 1:
            cell_embedding = cell_embedding.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Finding 概率
        finding_logits = self.finding_head(cell_embedding)
        finding_probs = F.softmax(finding_logits, dim=-1)

        # Issue reduction 预测
        issue_reduction = self.issue_head(cell_embedding)

        if squeeze:
            finding_probs = finding_probs.squeeze(0)
            issue_reduction = issue_reduction.squeeze(0)

        return finding_probs, issue_reduction

    def compute_uncertainty(self, finding_probs: torch.Tensor) -> float:
        """计算 finding 分布的熵（不确定性）

        H(p) = -Σ p_i log(p_i)

        对 critical findings 加权
        """
        eps = 1e-8
        probs = finding_probs.detach().cpu().numpy()

        # 基础熵
        entropy = -np.sum(probs * np.log(probs + eps))

        # Critical findings 的边际概率（越高说明越需要细化）
        critical_mass = 0.0
        for i, name in enumerate(self.finding_names):
            if name in CRITICAL_FINDINGS or any(cf in name for cf in CRITICAL_FINDINGS):
                critical_mass += probs[i]

        # 不确定性 = 熵 + critical 概率加权
        uncertainty = entropy + 0.5 * critical_mass
        return float(uncertainty)


def compute_delta(
    cell: Cell,
    cell_embedding: torch.Tensor,
    evidence_head: EvidenceHead,
    current_issues: List[Issue],
    lambda_uncertainty: float = 0.3,
) -> EvidenceScore:
    """计算单个 cell 的边际证据收益 Δ(c)

    Δ(c) = Σ_{u∈I} w_u · Δ̂_issue(u,c) + λ · H(p(critical | c))

    Args:
        cell: 目标 cell
        cell_embedding: cell 的特征向量
        evidence_head: 证据预测头
        current_issues: 当前的 verifier issues
        lambda_uncertainty: 不确定性项的权重

    Returns:
        EvidenceScore 包含 Δ(c) 及各分量
    """
    # 1. 获取预测
    with torch.no_grad():
        finding_probs, issue_pred = evidence_head(cell_embedding)

    # 2. 计算 verifier-driven 项
    # 基于当前 issues 和预测的 issue reduction
    issue_reduction_total = 0.0

    # 统计当前各类型 issue 数量
    ISSUE_TYPE_LIST = ["U1_unsupported", "O1_overclaim", "I1_inconsistency", "M1_missing_slot"]
    issue_counts = {it: 0.0 for it in ISSUE_TYPE_LIST}
    for issue in current_issues:
        if issue.issue_type in issue_counts:
            issue_counts[issue.issue_type] += SEVERITY_WEIGHTS.get(issue.severity, 0.5)

    # issue_pred: [U1, O1, I1, M1] 的预测减少量
    issue_pred_np = issue_pred.detach().cpu().numpy()

    for i, it in enumerate(ISSUE_TYPE_LIST):
        # 预测的减少量 * 当前该类型 issue 的加权数量
        issue_reduction_total += issue_pred_np[i] * issue_counts[it]

    # 3. 计算 uncertainty 项
    uncertainty = evidence_head.compute_uncertainty(finding_probs)

    # 4. 综合计算 Δ(c)
    delta = issue_reduction_total + lambda_uncertainty * uncertainty

    # 5. 构造 slot_probs
    probs_np = finding_probs.detach().cpu().numpy()
    slot_probs = {
        name: float(probs_np[i])
        for i, name in enumerate(evidence_head.finding_names)
    }

    return EvidenceScore(
        cell=cell,
        delta=delta,
        issue_reduction=issue_reduction_total,
        uncertainty=uncertainty,
        slot_probs=slot_probs,
    )


def rank_cells_by_delta(
    cells: List[Cell],
    cell_embeddings: Dict[str, torch.Tensor],
    evidence_head: EvidenceHead,
    current_issues: List[Issue],
    max_depth: int = 4,
    epsilon: float = 0.01,
) -> List[EvidenceScore]:
    """对所有可 split 的 cell 按 Δ 排序

    Args:
        cells: 当前所有 cells
        cell_embeddings: cell_id -> embedding 的映射
        evidence_head: 证据预测头
        current_issues: 当前 issues
        max_depth: 最大深度限制
        epsilon: 停机阈值

    Returns:
        按 Δ 降序排列的 EvidenceScore 列表（已过滤 < epsilon 的）
    """
    scores = []

    for cell in cells:
        # 检查是否可以 split
        if cell.level >= max_depth:
            continue

        cell_id = cell.id()
        if cell_id not in cell_embeddings:
            continue

        emb = cell_embeddings[cell_id]
        score = compute_delta(
            cell=cell,
            cell_embedding=emb,
            evidence_head=evidence_head,
            current_issues=current_issues,
        )

        if score.delta >= epsilon:
            scores.append(score)

    # 按 Δ 降序排列，tie-break 用 cell_id 字典序
    scores.sort(key=lambda s: (-s.delta, s.cell.id()))

    return scores


def select_cell_to_split_by_delta(
    cells: List[Cell],
    cell_embeddings: Dict[str, torch.Tensor],
    evidence_head: EvidenceHead,
    current_issues: List[Issue],
    max_depth: int = 4,
    epsilon: float = 0.01,
) -> Optional[Cell]:
    """选择 Δ 最大的 cell 进行 split

    这是 Algorithm 1 (Deterministic Greedy BET-Refine) 的核心
    """
    ranked = rank_cells_by_delta(
        cells=cells,
        cell_embeddings=cell_embeddings,
        evidence_head=evidence_head,
        current_issues=current_issues,
        max_depth=max_depth,
        epsilon=epsilon,
    )

    if not ranked:
        return None

    return ranked[0].cell
