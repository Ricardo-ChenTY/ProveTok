"""Rule-based Verifier

根据 proposal §4.3.4 Verifier-in-the-Loop:
检测四类问题：
- U1_unsupported: 没有足够证据支持的声明
- O1_overclaim: 证据不足以支持声明的强度/specificity
- I1_inconsistency: 内部逻辑不一致
- M1_missing_slot: 缺少必要的槽位信息

每个规则返回 Issue，包含：
- severity: 1-3 (1=minor, 2=moderate, 3=critical)
- rule_id: 用于审计追踪
- evidence_trace: 详细证据信息
"""
from __future__ import annotations
from typing import Any, List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import torch

from ..types import Issue, Token, Generation, Frame, IssueType
from .taxonomy import RULE_SET_VERSION


def build_evidence_trace(
    token_ids: List[int],
    token_map: Dict[int, Token],
    *,
    rule_inputs: Optional[Dict[str, Any]] = None,
    rule_outputs: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a minimal, auditable evidence_trace schema.

    Locked keys:
    - token_ids: cited token ids
    - token_cell_ids: cited cell_id list (same order as token_ids when possible)
    - rule_inputs: deterministic rule parameters / thresholds
    - rule_outputs: deterministic intermediate values used by the rule
    """
    token_cell_ids = [token_map[tid].cell_id for tid in token_ids if tid in token_map]
    trace: Dict[str, Any] = {
        "token_ids": list(token_ids),
        "token_cell_ids": token_cell_ids,
        "rule_inputs": rule_inputs or {},
        "rule_outputs": rule_outputs or {},
    }
    if extra:
        trace.update(extra)
    return trace


# ============================================================
# 解剖学知识库
# ============================================================

# 肺部位置与侧性的映射
LOCATION_LATERALITY_MAP = {
    # 左肺
    "LUL": "left",   # Left Upper Lobe
    "LLL": "left",   # Left Lower Lobe
    "lingula": "left",
    "left_lung": "left",
    "left_hilum": "left",
    # 右肺
    "RUL": "right",  # Right Upper Lobe
    "RML": "right",  # Right Middle Lobe
    "RLL": "right",  # Right Lower Lobe
    "right_lung": "right",
    "right_hilum": "right",
    # 双侧/中央
    "mediastinum": "bilateral",
    "trachea": "bilateral",
    "heart": "bilateral",
    "spine": "bilateral",
}

# 位置兼容性（哪些位置可以同时出现）
COMPATIBLE_LOCATIONS = {
    "left": {"LUL", "LLL", "lingula", "left_lung", "left_hilum"},
    "right": {"RUL", "RML", "RLL", "right_lung", "right_hilum"},
    "bilateral": {"mediastinum", "trachea", "heart", "spine"},
}

# Finding 类型的严重程度基线
FINDING_SEVERITY_MAP = {
    "nodule": 2,
    "mass": 3,
    "consolidation": 2,
    "opacity": 1,
    "effusion": 2,
    "atelectasis": 1,
    "pneumothorax": 3,
    "cardiomegaly": 2,
    "fracture": 2,
}

# Size bins 的阈值（mm）
SIZE_THRESHOLDS = {
    "tiny": (0, 3),
    "small": (3, 6),
    "medium": (6, 10),
    "large": (10, 30),
    "massive": (30, float('inf')),
}


# ============================================================
# 规则基类
# ============================================================

@dataclass
class RuleConfig:
    """规则配置"""
    enabled: bool = True
    severity_override: Optional[int] = None
    threshold: float = 0.5


class VerificationRule(ABC):
    """验证规则基类"""

    def __init__(self, rule_id: str, issue_type: IssueType, default_severity: int = 2, severity: int = None):
        self.rule_id = rule_id
        self.issue_type = issue_type
        self.default_severity = severity if severity is not None else default_severity
        self.config = RuleConfig()

    @abstractmethod
    def check(
        self,
        frame_idx: int,
        frame: Frame,
        generation: Generation,
        tokens: List[Token],
        token_map: Dict[int, Token],
    ) -> Optional[Issue]:
        """检查单个 frame，返回 Issue 或 None"""
        pass

    def _make_issue(
        self,
        frame_idx: int,
        message: str,
        evidence_trace: Dict,
        severity: Optional[int] = None,
    ) -> Issue:
        """创建 Issue"""
        return Issue(
            frame_idx=frame_idx,
            issue_type=self.issue_type,
            severity=severity or self.config.severity_override or self.default_severity,
            rule_id=self.rule_id,
            message=message,
            evidence_trace=evidence_trace,
        )


# ============================================================
# U1: Unsupported 规则
# ============================================================

class U1_NoCitation(VerificationRule):
    """正向声明没有 citation"""

    def __init__(self):
        super().__init__("U1.0", "U1_unsupported", severity=3)

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if frame.polarity in ("present", "positive"):
            citations = generation.citations.get(frame_idx, [])
            if not citations:
                return self._make_issue(
                    frame_idx=frame_idx,
                    message="Positive claim without any citations.",
                    evidence_trace=build_evidence_trace(
                        [],
                        token_map,
                        rule_inputs={},
                        rule_outputs={},
                    ),
                )
        return None


class U1_LowScore(VerificationRule):
    """Citation 的 evidence score 太低"""

    def __init__(self, min_score: float = 0.35):
        super().__init__("U1.1", "U1_unsupported", severity=2)
        self.min_score = min_score

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if frame.polarity not in ("present", "positive"):
            return None

        citations = generation.citations.get(frame_idx, [])
        if not citations:
            return None  # U1.0 会处理

        scores = [token_map[tid].score for tid in citations if tid in token_map]
        cell_ids = [token_map[tid].cell_id for tid in citations if tid in token_map]

        if scores and max(scores) < self.min_score:
            return self._make_issue(
                frame_idx=frame_idx,
                message=f"Cited evidence has low support score (max={max(scores):.3f} < {self.min_score}).",
                evidence_trace=build_evidence_trace(
                    citations,
                    token_map,
                    rule_inputs={"min_score": self.min_score},
                    rule_outputs={"scores": scores, "max_score": max(scores) if scores else None},
                    extra={"cell_ids": cell_ids},
                ),
            )
        return None


class U1_HighUncertainty(VerificationRule):
    """Citation 的 uncertainty 太高"""

    def __init__(self, max_uncertainty: float = 0.7):
        super().__init__("U1.2", "U1_unsupported", severity=2)
        self.max_uncertainty = max_uncertainty

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if frame.polarity not in ("present", "positive"):
            return None

        citations = generation.citations.get(frame_idx, [])
        if not citations:
            return None

        uncertainties = [token_map[tid].uncertainty for tid in citations if tid in token_map]

        if uncertainties and min(uncertainties) > self.max_uncertainty:
            return self._make_issue(
                frame_idx=frame_idx,
                message=f"Cited evidence has high uncertainty (min={min(uncertainties):.3f} > {self.max_uncertainty}).",
                evidence_trace=build_evidence_trace(
                    citations,
                    token_map,
                    rule_inputs={"max_uncertainty": self.max_uncertainty},
                    rule_outputs={"uncertainties": uncertainties, "min_uncertainty": min(uncertainties) if uncertainties else None},
                ),
            )
        return None


class U1_InsufficientCoverage(VerificationRule):
    """Citation 覆盖的空间范围不足"""

    def __init__(self, min_coverage_ratio: float = 0.1):
        super().__init__("U1.3", "U1_unsupported", severity=1)
        self.min_coverage_ratio = min_coverage_ratio

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if frame.polarity not in ("present", "positive"):
            return None

        citations = generation.citations.get(frame_idx, [])
        if not citations:
            return None

        # 计算 citation 覆盖的总体积比例
        cited_levels = [token_map[tid].level for tid in citations if tid in token_map]
        if not cited_levels:
            return None

        # 简化计算：level 越高，覆盖越小
        # level 0: 覆盖 1/1, level 1: 1/8, level 2: 1/64, ...
        coverage = sum(1 / (8 ** level) for level in cited_levels)

        if coverage < self.min_coverage_ratio:
            return self._make_issue(
                frame_idx=frame_idx,
                message=f"Cited evidence covers insufficient volume (coverage={coverage:.3f}).",
                evidence_trace=build_evidence_trace(
                    citations,
                    token_map,
                    rule_inputs={"min_coverage_ratio": self.min_coverage_ratio},
                    rule_outputs={"levels": cited_levels, "coverage": coverage},
                ),
                severity=1,
            )
        return None


class U1_CitationRelevance(VerificationRule):
    """Citations should be relevant to the claim (ToyPCG-style attention proxy).

    This is a deterministic, auditable proxy rule:
    - Build a query vector from `frame.finding`
    - Compute attention over evidence-token embeddings (optionally score-biased)
    - If cited tokens are not among the top-K for that claim, flag as unsupported

    Notes:
    - Designed to make `cite_swap` counterfactual detectable.
    - When embeddings are unavailable/non-tensor, this rule is skipped.
    """

    def __init__(
        self,
        *,
        min_recall_at_k: float = 0.5,
        min_attention_mass: float = 0.2,
        query_seed: int = 0,
        score_bias: float = 0.0,
    ):
        super().__init__("U1.4", "U1_unsupported", severity=2)
        self.min_recall_at_k = float(min_recall_at_k)
        self.min_attention_mass = float(min_attention_mass)
        self.query_seed = int(query_seed)
        self.score_bias = float(score_bias)

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if frame.polarity not in ("present", "positive"):
            return None

        citations = generation.citations.get(frame_idx, [])
        if not citations:
            return None

        if not tokens:
            return None

        # Build embedding matrix (N,D).
        rows: List[torch.Tensor] = []
        scores: List[float] = []
        token_id_to_row: Dict[int, int] = {}
        for row, t in enumerate(tokens):
            if not isinstance(t.embedding, torch.Tensor) or t.embedding.dim() != 1:
                return None
            rows.append(t.embedding.to(dtype=torch.float32))
            scores.append(float(t.score))
            token_id_to_row[int(t.token_id)] = row

        if not rows:
            return None

        T = torch.stack(rows, dim=0)  # (N,D)
        score_vec = torch.tensor(scores, dtype=T.dtype, device=T.device)

        from ..pcg.toy_queries import toy_query_vector

        q = toy_query_vector(frame.finding, emb_dim=int(T.shape[1]), seed=self.query_seed, device=T.device, dtype=T.dtype)
        logits = (T @ q)  # (N,)
        if self.score_bias != 0.0:
            logits = logits + self.score_bias * score_vec
        att = torch.softmax(logits, dim=0)

        cited_rows = [token_id_to_row[int(tid)] for tid in citations if int(tid) in token_id_to_row]
        if not cited_rows:
            return None

        k = min(max(len(cited_rows), 1), int(att.numel()))
        topk_rows = torch.topk(att, k=k).indices.tolist()

        cited_set = set(cited_rows)
        topk_set = set(topk_rows)
        recall_at_k = float(len(cited_set & topk_set) / max(len(cited_set), 1))
        attention_mass = float(att[cited_rows].sum().item()) if cited_rows else 0.0

        if recall_at_k >= self.min_recall_at_k and attention_mass >= self.min_attention_mass:
            return None

        topk_token_ids = [int(tokens[r].token_id) for r in topk_rows if 0 <= r < len(tokens)]
        return self._make_issue(
            frame_idx=frame_idx,
            message=f"Citations appear irrelevant (recall@k={recall_at_k:.2f}, att_mass={attention_mass:.2f}).",
            evidence_trace=build_evidence_trace(
                citations,
                token_map,
                rule_inputs={
                    "min_recall_at_k": self.min_recall_at_k,
                    "min_attention_mass": self.min_attention_mass,
                    "query_seed": self.query_seed,
                    "score_bias": self.score_bias,
                },
                rule_outputs={
                    "k": int(k),
                    "topk_token_ids": topk_token_ids,
                    "recall_at_k": recall_at_k,
                    "attention_mass_cited": attention_mass,
                },
            ),
        )


# ============================================================
# O1: Overclaim 规则
# ============================================================

class O1_CoarseLevelOnly(VerificationRule):
    """具体声明只有粗粒度证据"""

    def __init__(self, max_coarse_level: int = 0):
        super().__init__("O1.0", "O1_overclaim", severity=2)
        self.max_coarse_level = max_coarse_level

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if frame.polarity not in ("present", "positive"):
            return None

        # 具体声明：指定了 laterality
        if frame.laterality not in ("left", "right", "bilateral"):
            return None

        citations = generation.citations.get(frame_idx, [])
        if not citations:
            return None

        levels = [token_map[tid].level for tid in citations if tid in token_map]
        cell_ids = [token_map[tid].cell_id for tid in citations if tid in token_map]

        if levels and max(levels) <= self.max_coarse_level:
            return self._make_issue(
                frame_idx=frame_idx,
                message=f"Specific claim (laterality={frame.laterality}) supported only by coarse-level evidence.",
                evidence_trace=build_evidence_trace(
                    citations,
                    token_map,
                    rule_inputs={"max_coarse_level": self.max_coarse_level},
                    rule_outputs={"levels": levels, "max_level": max(levels) if levels else None},
                    extra={"cell_ids": cell_ids, "laterality": frame.laterality},
                ),
            )
        return None


class O1_SeverityMismatch(VerificationRule):
    """声明的严重程度与证据不匹配"""

    def __init__(self):
        super().__init__("O1.1", "O1_overclaim", severity=2)

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if frame.polarity not in ("present", "positive"):
            return None

        # 获取 finding 的基线严重程度
        finding_severity = FINDING_SEVERITY_MAP.get(frame.finding.lower(), 1)

        # 如果声明的置信度很高但证据 score 不高
        citations = generation.citations.get(frame_idx, [])
        if not citations:
            return None

        scores = [token_map[tid].score for tid in citations if tid in token_map]
        if not scores:
            return None

        avg_score = np.mean(scores)

        # 高严重程度 finding + 高置信度 需要高 score 证据
        if finding_severity >= 2 and frame.confidence > 0.8 and avg_score < 0.5:
            return self._make_issue(
                frame_idx=frame_idx,
                message=f"High-confidence claim for serious finding ({frame.finding}) lacks strong evidence.",
                evidence_trace=build_evidence_trace(
                    citations,
                    token_map,
                    rule_inputs={},
                    rule_outputs={
                        "finding": frame.finding,
                        "finding_severity": finding_severity,
                        "confidence": frame.confidence,
                        "avg_evidence_score": avg_score,
                    },
                ),
            )
        return None


class O1_SizeClaim(VerificationRule):
    """Size 声明需要足够精细的证据"""

    def __init__(self, min_level_for_size: int = 2):
        super().__init__("O1.2", "O1_overclaim", severity=2)
        self.min_level_for_size = min_level_for_size

    def check(self, frame_idx, frame, generation, tokens, token_map):
        # 这个规则需要 Frame 有 size 属性，目前 types.py 没有
        # 暂时跳过，或者检查 finding 名称中是否有 size 暗示
        return None


# ============================================================
# I1: Inconsistency 规则
# ============================================================

class I1_BilateralCitation(VerificationRule):
    """Bilateral 声明需要来自两侧的证据"""

    def __init__(self, min_citations: int = 2):
        super().__init__("I1.0", "I1_inconsistency", severity=2)
        self.min_citations = min_citations

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if frame.laterality != "bilateral":
            return None

        citations = generation.citations.get(frame_idx, [])
        cell_ids = [token_map[tid].cell_id for tid in citations if tid in token_map]

        if len(citations) < self.min_citations:
            return self._make_issue(
                frame_idx=frame_idx,
                message=f"Bilateral claim requires >= {self.min_citations} citations (got {len(citations)}).",
                evidence_trace=build_evidence_trace(
                    citations,
                    token_map,
                    rule_inputs={"min_citations": self.min_citations},
                    rule_outputs={"num_citations": len(citations)},
                    extra={"cell_ids": cell_ids},
                ),
            )
        return None


class I1_AnatomicalMismatch(VerificationRule):
    """声明的位置与 citation 的解剖位置不匹配"""

    def __init__(self):
        super().__init__("I1.1", "I1_inconsistency", severity=2)

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if frame.polarity not in ("present", "positive"):
            return None

        # 需要从 cell_id 推断解剖位置
        # 简化：检查 laterality 与 cell 位置是否一致
        citations = generation.citations.get(frame_idx, [])
        if not citations:
            return None

        from ..grid.cells import parse_cell_id

        # 从 cell_id 提取位置信息
        cell_ids = [token_map[tid].cell_id for tid in citations if tid in token_map]

        # 简化实现：检查 cell 的 x 坐标判断左右
        left_count = 0
        right_count = 0

        for cell_id in cell_ids:
            cell = parse_cell_id(cell_id)
            if cell is None:
                continue
            if cell.ix == 0 and cell.level == 0:
                left_count += 1
                right_count += 1  # level 0 covers whole volume
            elif cell.ix % 2 == 0:
                left_count += 1
            else:
                right_count += 1

        # 检查一致性
        if frame.laterality == "left" and left_count == 0 and right_count > 0:
            return self._make_issue(
                frame_idx=frame_idx,
                message="Left laterality claim but citations are from right side.",
                evidence_trace=build_evidence_trace(
                    citations,
                    token_map,
                    rule_inputs={},
                    rule_outputs={"left_count": left_count, "right_count": right_count},
                    extra={"laterality": frame.laterality, "cell_ids": cell_ids},
                ),
            )
        elif frame.laterality == "right" and right_count == 0 and left_count > 0:
            return self._make_issue(
                frame_idx=frame_idx,
                message="Right laterality claim but citations are from left side.",
                evidence_trace=build_evidence_trace(
                    citations,
                    token_map,
                    rule_inputs={},
                    rule_outputs={"left_count": left_count, "right_count": right_count},
                    extra={"laterality": frame.laterality, "cell_ids": cell_ids},
                ),
            )

        return None


class I1_ConflictingFindings(VerificationRule):
    """同一位置的冲突 findings"""

    def __init__(self):
        super().__init__("I1.2", "I1_inconsistency", severity=2)

    def check(self, frame_idx, frame, generation, tokens, token_map):
        # 检查同一 generation 中是否有冲突的 findings
        # 例如：同一位置同时有 "nodule present" 和 "no nodule"

        if generation.refusal.get(frame_idx, False):
            return None

        # 获取当前 frame 的 citations
        current_citations = set(generation.citations.get(frame_idx, []))
        if not current_citations:
            return None

        # 检查其他 frames 是否与当前 frame 冲突
        for other_idx, other_frame in enumerate(generation.frames):
            if other_idx == frame_idx:
                continue

            if generation.refusal.get(other_idx, False):
                continue

            other_citations = set(generation.citations.get(other_idx, []))

            # 如果 citations 有重叠
            overlap = current_citations & other_citations
            if not overlap:
                continue

            # 检查是否为同一 finding 的冲突 polarity
            if (frame.finding.lower() == other_frame.finding.lower() and
                frame.polarity != other_frame.polarity):
                return self._make_issue(
                    frame_idx=frame_idx,
                    message=f"Conflicting findings: {frame.finding} is both {frame.polarity} and {other_frame.polarity}.",
                    evidence_trace=build_evidence_trace(
                        list(overlap),
                        token_map,
                        rule_inputs={},
                        rule_outputs={
                            "current_frame": frame_idx,
                            "other_frame": other_idx,
                            "finding": frame.finding,
                            "polarities": [frame.polarity, other_frame.polarity],
                        },
                    ),
                )

        return None


class I1_PolarityConfidence(VerificationRule):
    """Negative 声明不应该有高置信度除非有足够证据"""

    def __init__(self, negative_max_confidence: float = 0.9):
        super().__init__("I1.3", "I1_inconsistency", severity=1)
        self.negative_max_confidence = negative_max_confidence

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        # Negative 声明的置信度不应过高（因为"absence of evidence is not evidence of absence"）
        if frame.polarity in ("absent", "negative") and frame.confidence > self.negative_max_confidence:
            citations = generation.citations.get(frame_idx, [])

            # 除非有大量高质量证据覆盖该区域
            if len(citations) < 5:  # 简化阈值
                return self._make_issue(
                    frame_idx=frame_idx,
                    message=f"High confidence ({frame.confidence:.2f}) for negative finding without extensive coverage.",
                    evidence_trace=build_evidence_trace(
                        citations,
                        token_map,
                        rule_inputs={"negative_max_confidence": self.negative_max_confidence},
                        rule_outputs={"polarity": frame.polarity, "confidence": frame.confidence, "num_citations": len(citations)},
                    ),
                    severity=1,
                )
        return None


class I1_TextRoundTrip(VerificationRule):
    """Dual-channel protocol: narrative text must round-trip to the same findings table.

    If `generation.text` is empty, this rule is skipped.
    """

    def __init__(self):
        super().__init__("I1.4", "I1_inconsistency", severity=2)

    def check(self, frame_idx, frame, generation, tokens, token_map):
        # This is a generation-level check; run once.
        if frame_idx != 0:
            return None

        if not getattr(generation, "text", "").strip():
            return None

        from ..pcg.narrative import parse_generation_text, roundtrip_equal

        try:
            frames2, citations2, q2, refusal2 = parse_generation_text(generation.text)
        except Exception as e:
            all_token_ids = sorted({int(t) for cites in generation.citations.values() for t in cites})
            return self._make_issue(
                frame_idx=frame_idx,
                message=f"Narrative text is not parseable: {e}",
                evidence_trace=build_evidence_trace(
                    all_token_ids,
                    token_map,
                    rule_outputs={"parse_error": str(e)},
                ),
                severity=2,
            )

        parsed = Generation(frames=frames2, citations=citations2, q=q2, refusal=refusal2, text="")
        original = Generation(
            frames=generation.frames,
            citations=generation.citations,
            q=generation.q,
            refusal=generation.refusal,
            text="",
        )

        if not roundtrip_equal(original, parsed):
            all_token_ids = sorted({int(t) for cites in generation.citations.values() for t in cites})
            return self._make_issue(
                frame_idx=frame_idx,
                message="Narrative text does not round-trip to the same findings table.",
                evidence_trace=build_evidence_trace(
                    all_token_ids,
                    token_map,
                    rule_outputs={
                        "original_frames": [f.finding for f in generation.frames],
                        "parsed_frames": [f.finding for f in frames2],
                    },
                ),
                severity=2,
            )

        return None


# ============================================================
# M1: Missing Slot 规则
# ============================================================

class M1_MissingLaterality(VerificationRule):
    """正向 finding 缺少 laterality"""

    def __init__(self):
        super().__init__("M1.0", "M1_missing_slot", severity=1)

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if frame.polarity in ("present", "positive") and frame.laterality == "unspecified":
            citations = generation.citations.get(frame_idx, [])
            cell_ids = [token_map[tid].cell_id for tid in citations if tid in token_map]

            return self._make_issue(
                frame_idx=frame_idx,
                message="Positive finding must specify laterality.",
                evidence_trace=build_evidence_trace(
                    citations,
                    token_map,
                    rule_inputs={},
                    rule_outputs={"finding": frame.finding},
                    extra={"cell_ids": cell_ids},
                ),
            )
        return None


class M1_MissingFinding(VerificationRule):
    """Frame 缺少 finding type"""

    def __init__(self):
        super().__init__("M1.1", "M1_missing_slot", severity=2)

    def check(self, frame_idx, frame, generation, tokens, token_map):
        if generation.refusal.get(frame_idx, False):
            return None

        if not frame.finding or frame.finding.lower() in ("unknown", "unspecified", ""):
            return self._make_issue(
                frame_idx=frame_idx,
                message="Frame is missing finding type.",
                evidence_trace=build_evidence_trace(
                    [],
                    token_map,
                    rule_outputs={"finding": frame.finding},
                ),
            )
        return None


class M1_LowConfidenceNoExplanation(VerificationRule):
    """低置信度声明应该有解释（或被 refusal）"""

    def __init__(self, low_confidence_threshold: float = 0.5):
        super().__init__("M1.2", "M1_missing_slot", severity=1)
        self.low_confidence_threshold = low_confidence_threshold

    def check(self, frame_idx, frame, generation, tokens, token_map):
        # 如果置信度低但没有 refusal，可能需要更多解释
        if frame.confidence < self.low_confidence_threshold:
            if not generation.refusal.get(frame_idx, False):
                return self._make_issue(
                    frame_idx=frame_idx,
                    message=f"Low confidence ({frame.confidence:.2f}) finding should either be refused or have more evidence.",
                    evidence_trace=build_evidence_trace(
                        [],
                        token_map,
                        rule_inputs={"low_confidence_threshold": self.low_confidence_threshold},
                        rule_outputs={
                            "confidence": frame.confidence,
                            "refusal": generation.refusal.get(frame_idx, False),
                        },
                    ),
                    severity=1,
                )
        return None


# ============================================================
# Verifier 主类
# ============================================================

class RuleBasedVerifier:
    """基于规则的 Verifier

    Usage:
        verifier = RuleBasedVerifier()
        verifier.add_rule(U1_NoCitation())
        issues = verifier.verify(generation, tokens)
    """

    def __init__(self):
        self.rules: List[VerificationRule] = []

    def add_rule(self, rule: VerificationRule):
        """添加规则"""
        self.rules.append(rule)
        return self

    def add_default_rules(self):
        """添加默认规则集"""
        # U1: Unsupported
        self.add_rule(U1_NoCitation())
        self.add_rule(U1_LowScore(min_score=0.35))
        self.add_rule(U1_HighUncertainty(max_uncertainty=0.7))
        self.add_rule(U1_InsufficientCoverage(min_coverage_ratio=0.05))
        self.add_rule(U1_CitationRelevance())

        # O1: Overclaim
        self.add_rule(O1_CoarseLevelOnly(max_coarse_level=0))
        self.add_rule(O1_SeverityMismatch())

        # I1: Inconsistency
        self.add_rule(I1_BilateralCitation(min_citations=2))
        self.add_rule(I1_AnatomicalMismatch())
        self.add_rule(I1_ConflictingFindings())
        self.add_rule(I1_PolarityConfidence())
        self.add_rule(I1_TextRoundTrip())

        # M1: Missing Slot
        self.add_rule(M1_MissingLaterality())
        self.add_rule(M1_MissingFinding())
        self.add_rule(M1_LowConfidenceNoExplanation())

        return self

    def verify(self, generation: Generation, tokens: List[Token]) -> List[Issue]:
        """验证 generation，返回所有发现的 issues"""
        issues: List[Issue] = []
        token_map = {t.token_id: t for t in tokens}

        for frame_idx, frame in enumerate(generation.frames):
            for rule in self.rules:
                if not rule.config.enabled:
                    continue

                issue = rule.check(
                    frame_idx=frame_idx,
                    frame=frame,
                    generation=generation,
                    tokens=tokens,
                    token_map=token_map,
                )

                if issue is not None:
                    issues.append(issue)

        return issues

    def get_issues_by_type(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """按类型分组 issues"""
        result: Dict[str, List[Issue]] = {}
        for issue in issues:
            if issue.issue_type not in result:
                result[issue.issue_type] = []
            result[issue.issue_type].append(issue)
        return result

    def get_issues_by_severity(self, issues: List[Issue]) -> Dict[int, List[Issue]]:
        """按严重程度分组 issues"""
        result: Dict[int, List[Issue]] = {}
        for issue in issues:
            if issue.severity not in result:
                result[issue.severity] = []
            result[issue.severity].append(issue)
        return result


# ============================================================
# 向后兼容的 verify 函数
# ============================================================

_default_verifier: Optional[RuleBasedVerifier] = None


def get_default_verifier() -> RuleBasedVerifier:
    """获取默认 verifier（单例）"""
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = RuleBasedVerifier().add_default_rules()
    return _default_verifier


def verify(gen: Generation, tokens: List[Token]) -> List[Issue]:
    """向后兼容的 verify 函数"""
    return get_default_verifier().verify(gen, tokens)


def create_verifier(rules: Optional[List[str]] = None) -> RuleBasedVerifier:
    """创建自定义 verifier

    Args:
        rules: 要启用的规则 ID 列表，如 ["U1.0", "U1.1", "M1.0"]
               如果为 None，则启用所有默认规则

    Returns:
        RuleBasedVerifier 实例
    """
    verifier = RuleBasedVerifier().add_default_rules()

    if rules is not None:
        rule_set = set(rules)
        for rule in verifier.rules:
            rule.config.enabled = rule.rule_id in rule_set

    return verifier
