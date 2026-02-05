"""Evidence Graph: Token → Slot Value 映射

根据 proposal §4.3.1:
E(i) = {(type=nodule, p), (loc=LLL, p), (lat=left, p), (sizebin=3-5mm, p), ...}

全局合法域: V_slot = ∪_i E(i)
constrained decoding 只允许输出域内槽值

这个模块负责:
1. 为每个 token 计算其支持的 slot value 及置信度
2. 构建全局合法域 V_slot
3. 支持 constrained decoding 的查询接口
"""
from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import Token
from .schema import FINDINGS, LATERALITY, POLARITY, LOCATIONS, SIZE_BINS, SEVERITY_LEVELS


@dataclass(frozen=True)
class SlotValue:
    """单个 slot value 及其置信度"""
    slot: str           # slot name: finding_type, location, laterality, size_bin, severity
    value: str          # slot value
    confidence: float   # 置信度 [0, 1]


@dataclass
class TokenEvidence:
    """单个 token 的证据图条目"""
    token_id: int
    cell_id: str
    slot_values: List[SlotValue] = field(default_factory=list)

    def get_slot(self, slot_name: str) -> List[SlotValue]:
        """获取指定 slot 的所有候选值"""
        return [sv for sv in self.slot_values if sv.slot == slot_name]

    def best_value(self, slot_name: str) -> Optional[SlotValue]:
        """获取指定 slot 的最高置信度值"""
        candidates = self.get_slot(slot_name)
        if not candidates:
            return None
        return max(candidates, key=lambda x: x.confidence)

    def supports(self, slot: str, value: str, threshold: float = 0.3) -> bool:
        """检查是否支持特定 slot value"""
        for sv in self.slot_values:
            if sv.slot == slot and sv.value == value and sv.confidence >= threshold:
                return True
        return False


@dataclass
class EvidenceGraph:
    """全局证据图"""
    entries: Dict[int, TokenEvidence] = field(default_factory=dict)  # token_id -> evidence

    def add_entry(self, entry: TokenEvidence):
        self.entries[entry.token_id] = entry

    def get_valid_domain(self, slot: str) -> Set[str]:
        """获取某个 slot 的全局合法域 V_slot"""
        values = set()
        for entry in self.entries.values():
            for sv in entry.slot_values:
                if sv.slot == slot and sv.confidence > 0.1:  # 最小阈值
                    values.add(sv.value)
        return values

    def get_supporting_tokens(
        self,
        slot: str,
        value: str,
        threshold: float = 0.3
    ) -> List[int]:
        """获取支持特定 slot value 的所有 token ids"""
        supporting = []
        for token_id, entry in self.entries.items():
            if entry.supports(slot, value, threshold):
                supporting.append(token_id)
        return supporting

    def check_citation_support(
        self,
        citations: List[int],
        slot: str,
        value: str,
        threshold: float = 0.3
    ) -> Tuple[bool, float]:
        """检查 citations 是否支持特定 slot value

        Returns:
            (is_supported, max_confidence)
        """
        max_conf = 0.0
        for tid in citations:
            if tid in self.entries:
                for sv in self.entries[tid].slot_values:
                    if sv.slot == slot and sv.value == value:
                        max_conf = max(max_conf, sv.confidence)

        return (max_conf >= threshold, max_conf)


class EvidenceGraphBuilder(nn.Module):
    """从 token embeddings 构建 Evidence Graph

    对每个 token，预测其支持的各类 slot values
    """

    def __init__(
        self,
        emb_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.emb_dim = emb_dim

        # Finding type 分类器
        self.finding_classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(FINDINGS)),
        )

        # Location 分类器
        self.location_classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(LOCATIONS)),
        )

        # Laterality 分类器
        self.laterality_classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(LATERALITY)),
        )

        # Size bin 分类器
        self.size_classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(SIZE_BINS)),
        )

        # Severity 分类器
        self.severity_classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(SEVERITY_LEVELS)),
        )

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: (N, D) token embeddings

        Returns:
            Dict of slot -> (N, num_values) probability distributions
        """
        return {
            "finding_type": F.softmax(self.finding_classifier(embeddings), dim=-1),
            "location": F.softmax(self.location_classifier(embeddings), dim=-1),
            "laterality": F.softmax(self.laterality_classifier(embeddings), dim=-1),
            "size_bin": F.softmax(self.size_classifier(embeddings), dim=-1),
            "severity": F.softmax(self.severity_classifier(embeddings), dim=-1),
        }

    def build_graph(
        self,
        tokens: List[Token],
        top_k: int = 3,
        min_conf: float = 0.1,
    ) -> EvidenceGraph:
        """从 tokens 构建完整的 Evidence Graph

        Args:
            tokens: Token 列表
            top_k: 每个 slot 保留的最大候选数
            min_conf: 最小置信度阈值

        Returns:
            EvidenceGraph
        """
        if not tokens:
            return EvidenceGraph()

        # 收集 embeddings
        embeddings = torch.stack([t.embedding for t in tokens])

        # 获取各 slot 的概率分布
        with torch.no_grad():
            slot_probs = self(embeddings)

        graph = EvidenceGraph()

        # Slot value 名称映射
        slot_values_map = {
            "finding_type": FINDINGS,
            "location": LOCATIONS,
            "laterality": LATERALITY,
            "size_bin": SIZE_BINS,
            "severity": SEVERITY_LEVELS,
        }

        for i, token in enumerate(tokens):
            slot_values = []

            for slot_name, probs in slot_probs.items():
                probs_i = probs[i].cpu().numpy()
                value_names = slot_values_map[slot_name]

                # 获取 top-k 候选
                top_indices = probs_i.argsort()[-top_k:][::-1]

                for idx in top_indices:
                    conf = float(probs_i[idx])
                    if conf >= min_conf:
                        slot_values.append(SlotValue(
                            slot=slot_name,
                            value=value_names[idx],
                            confidence=conf,
                        ))

            entry = TokenEvidence(
                token_id=token.token_id,
                cell_id=token.cell_id,
                slot_values=slot_values,
            )
            graph.add_entry(entry)

        return graph


def compute_support_score(
    graph: EvidenceGraph,
    citations: List[int],
    finding_type: str,
    location: Optional[str] = None,
    laterality: Optional[str] = None,
) -> float:
    """计算 citations 对特定 claim 的支持分数

    用于 verifier 判断 unsupported/overclaim

    Args:
        graph: Evidence Graph
        citations: cited token ids
        finding_type: claimed finding type
        location: claimed location (optional)
        laterality: claimed laterality (optional)

    Returns:
        support score in [0, 1]
    """
    if not citations:
        return 0.0

    scores = []

    # Finding type 支持
    _, finding_conf = graph.check_citation_support(
        citations, "finding_type", finding_type
    )
    scores.append(finding_conf)

    # Location 支持（如果指定）
    if location and location != "unspecified":
        _, loc_conf = graph.check_citation_support(
            citations, "location", location
        )
        scores.append(loc_conf)

    # Laterality 支持（如果指定）
    if laterality and laterality != "unspecified":
        _, lat_conf = graph.check_citation_support(
            citations, "laterality", laterality
        )
        scores.append(lat_conf)

    # 返回平均支持分数
    return sum(scores) / len(scores) if scores else 0.0


def get_constrained_vocab(
    graph: EvidenceGraph,
    slot: str,
) -> Set[str]:
    """获取 constrained decoding 的合法词表

    根据 proposal: constrained decoding 只允许输出域内槽值
    """
    valid = graph.get_valid_domain(slot)

    # 总是允许 "unspecified"
    valid.add("unspecified")

    return valid
