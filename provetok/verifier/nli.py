"""NLI-based Verifier

使用 Natural Language Inference 模型验证 claim-evidence 一致性。

根据 proposal:
- 将 claim (frame) 和 evidence (cited tokens) 转换为文本
- 使用 NLI 模型判断：entailment / contradiction / neutral
- contradiction → 返回 Issue
- neutral + 高置信度 claim → 可能是 overclaim

支持的 NLI 模型:
1. HuggingFace transformers (默认: microsoft/deberta-v3-base-mnli)
2. OpenAI API (可选)
3. 本地规则 fallback (无需模型)
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from ..types import Issue, Token, Generation, Frame, IssueType


@dataclass
class NLIResult:
    """NLI 推理结果"""
    premise: str        # evidence 文本
    hypothesis: str     # claim 文本
    label: str          # "entailment", "contradiction", "neutral"
    scores: Dict[str, float]  # 各类别的概率
    confidence: float   # 最高概率


# ============================================================
# 文本转换工具
# ============================================================

def frame_to_claim(frame: Frame) -> str:
    """将 Frame 转换为 claim 文本"""
    polarity_map = {
        "present": "There is",
        "positive": "There is",
        "absent": "There is no",
        "negative": "There is no",
    }

    laterality_map = {
        "left": "in the left lung",
        "right": "in the right lung",
        "bilateral": "in both lungs",
        "unspecified": "",
    }

    polarity_text = polarity_map.get(frame.polarity, "There is")
    laterality_text = laterality_map.get(frame.laterality, "")

    claim = f"{polarity_text} {frame.finding} {laterality_text}".strip()
    claim = claim.rstrip(".") + "."

    return claim


def tokens_to_evidence(tokens: List[Token], volume_shape: Tuple[int, int, int] = None) -> str:
    """将 tokens 转换为 evidence 描述文本"""
    if not tokens:
        return "No evidence available."

    # 统计 token 信息
    levels = [t.level for t in tokens]
    scores = [t.score for t in tokens]
    uncertainties = [t.uncertainty for t in tokens]

    # 解析 cell 位置
    locations = []
    for t in tokens:
        from ..grid.cells import parse_cell_id

        cell = parse_cell_id(t.cell_id)
        if cell is None:
            locations.append("unknown region")
            continue
        # 简化位置描述（仅用于 NLI scaffold）
        if cell.ix == 0 and t.level == 0:
            locations.append("whole volume")
        elif cell.ix % 2 == 0:
            locations.append("left region")
        else:
            locations.append("right region")

    # 生成描述
    parts = []

    # 位置信息
    unique_locations = list(set(locations))
    if "whole volume" in unique_locations:
        parts.append("The evidence covers the whole volume.")
    else:
        parts.append(f"The evidence is from {', '.join(unique_locations)}.")

    # 质量信息
    avg_score = np.mean(scores)
    max_score = max(scores)
    if max_score > 0.7:
        parts.append(f"Strong evidence detected (score={max_score:.2f}).")
    elif max_score > 0.4:
        parts.append(f"Moderate evidence detected (score={max_score:.2f}).")
    else:
        parts.append(f"Weak evidence detected (score={max_score:.2f}).")

    # 精度信息
    max_level = max(levels)
    if max_level >= 3:
        parts.append("Evidence is at high resolution.")
    elif max_level >= 1:
        parts.append("Evidence is at moderate resolution.")
    else:
        parts.append("Evidence is at low resolution only.")

    return " ".join(parts)


# ============================================================
# NLI Backend 抽象
# ============================================================

class NLIBackend(ABC):
    """NLI 后端抽象类"""

    @abstractmethod
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """预测 premise 和 hypothesis 的关系"""
        pass

    def batch_predict(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """批量预测"""
        return [self.predict(p, h) for p, h in pairs]


class RuleBasedNLI(NLIBackend):
    """基于规则的 NLI（不需要模型）

    简单启发式：
    - 检查关键词匹配
    - 检查否定词
    - 检查位置一致性
    """

    def __init__(self):
        self.positive_keywords = {"strong", "detected", "present", "visible", "seen"}
        self.negative_keywords = {"no", "absent", "not", "without", "negative"}
        self.location_left = {"left"}
        self.location_right = {"right"}
        self.location_bilateral = {"both", "bilateral"}

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()

        # 检查否定一致性
        premise_negative = any(w in premise_lower for w in self.negative_keywords)
        hypothesis_negative = any(w in hypothesis_lower for w in self.negative_keywords)

        # 检查位置一致性
        premise_left = any(w in premise_lower for w in self.location_left)
        premise_right = any(w in premise_lower for w in self.location_right)
        hypothesis_left = any(w in hypothesis_lower for w in self.location_left)
        hypothesis_right = any(w in hypothesis_lower for w in self.location_right)

        # 简单规则判断
        scores = {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}

        # 否定不一致 → contradiction
        if premise_negative != hypothesis_negative:
            if "strong" in premise_lower or "detected" in premise_lower:
                # 有强证据但 claim 说没有
                if hypothesis_negative:
                    scores = {"entailment": 0.1, "contradiction": 0.7, "neutral": 0.2}
            elif "weak" in premise_lower or "no evidence" in premise_lower:
                # 没有证据但 claim 说有
                if not hypothesis_negative:
                    scores = {"entailment": 0.1, "contradiction": 0.6, "neutral": 0.3}

        # 位置不一致 → contradiction
        if hypothesis_left and premise_right and not premise_left:
            scores = {"entailment": 0.1, "contradiction": 0.7, "neutral": 0.2}
        elif hypothesis_right and premise_left and not premise_right:
            scores = {"entailment": 0.1, "contradiction": 0.7, "neutral": 0.2}

        # 证据强 + claim positive → entailment
        if "strong" in premise_lower and not hypothesis_negative:
            scores = {"entailment": 0.6, "contradiction": 0.1, "neutral": 0.3}

        label = max(scores, key=scores.get)
        confidence = scores[label]

        return NLIResult(
            premise=premise,
            hypothesis=hypothesis,
            label=label,
            scores=scores,
            confidence=confidence,
        )


class TransformersNLI(NLIBackend):
    """使用 HuggingFace Transformers 的 NLI"""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base-mnli",
        device: str = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """懒加载模型"""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = self._model.to(self.device)
            self._model.eval()

        except ImportError:
            raise ImportError(
                "transformers is required for TransformersNLI. "
                "Install with: pip install transformers torch"
            )

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        import torch

        self._load_model()

        # Tokenize
        inputs = self._tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # 获取标签映射（不同模型可能不同）
        id2label = self._model.config.id2label
        scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}

        # 标准化标签名
        label_map = {
            "ENTAILMENT": "entailment",
            "CONTRADICTION": "contradiction",
            "NEUTRAL": "neutral",
        }
        scores = {label_map.get(k.upper(), k.lower()): v for k, v in scores.items()}

        label = max(scores, key=scores.get)
        confidence = scores[label]

        return NLIResult(
            premise=premise,
            hypothesis=hypothesis,
            label=label,
            scores=scores,
            confidence=confidence,
        )


class OpenAINLI(NLIBackend):
    """使用 OpenAI API 的 NLI"""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        try:
            import openai
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        if self.api_key:
            openai.api_key = self.api_key

        prompt = f"""Given the following evidence and claim, determine their relationship.

Evidence: {premise}

Claim: {hypothesis}

Classify the relationship as one of:
- ENTAILMENT: The evidence supports the claim
- CONTRADICTION: The evidence contradicts the claim
- NEUTRAL: The evidence neither supports nor contradicts the claim

Respond with only the classification label."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0,
            )

            label_raw = response.choices[0].message.content.strip().lower()

            # 解析标签
            if "entailment" in label_raw:
                label = "entailment"
            elif "contradiction" in label_raw:
                label = "contradiction"
            else:
                label = "neutral"

            # OpenAI 不返回概率，使用固定值
            scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            scores[label] = 0.9

            return NLIResult(
                premise=premise,
                hypothesis=hypothesis,
                label=label,
                scores=scores,
                confidence=0.9,
            )

        except Exception as e:
            # 失败时返回 neutral
            return NLIResult(
                premise=premise,
                hypothesis=hypothesis,
                label="neutral",
                scores={"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34},
                confidence=0.34,
            )


# ============================================================
# NLI Verifier
# ============================================================

class NLIVerifier:
    """基于 NLI 的 Verifier

    Usage:
        verifier = NLIVerifier()  # 默认使用规则
        verifier = NLIVerifier(backend="transformers")  # 使用模型
        issues = verifier.verify(generation, tokens)
    """

    def __init__(
        self,
        backend: Union[str, NLIBackend] = "rule",
        contradiction_threshold: float = 0.6,
        neutral_high_confidence_threshold: float = 0.8,
    ):
        """
        Args:
            backend: "rule", "transformers", "openai" 或 NLIBackend 实例
            contradiction_threshold: 判定为 contradiction 的阈值
            neutral_high_confidence_threshold: neutral + 高置信度 claim 的阈值
        """
        if isinstance(backend, NLIBackend):
            self.backend = backend
        elif backend == "rule":
            self.backend = RuleBasedNLI()
        elif backend == "transformers":
            self.backend = TransformersNLI()
        elif backend == "openai":
            self.backend = OpenAINLI()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.contradiction_threshold = contradiction_threshold
        self.neutral_high_confidence_threshold = neutral_high_confidence_threshold

    def verify(self, generation: Generation, tokens: List[Token]) -> List[Issue]:
        """验证 generation"""
        issues = []
        token_map = {t.token_id: t for t in tokens}

        for frame_idx, frame in enumerate(generation.frames):
            if generation.refusal.get(frame_idx, False):
                continue

            # 获取 cited tokens
            citations = generation.citations.get(frame_idx, [])
            cited_tokens = [token_map[tid] for tid in citations if tid in token_map]

            if not cited_tokens:
                continue  # 规则 verifier 会处理

            # 转换为文本
            claim = frame_to_claim(frame)
            evidence = tokens_to_evidence(cited_tokens)

            # NLI 推理
            result = self.backend.predict(evidence, claim)

            # 检查 contradiction
            if (result.label == "contradiction" and
                result.scores.get("contradiction", 0) >= self.contradiction_threshold):
                issues.append(Issue(
                    frame_idx=frame_idx,
                    issue_type="I1_inconsistency",
                    severity=3,
                    rule_id="NLI.contradiction",
                    message=f"NLI detected contradiction between claim and evidence.",
                    evidence_trace={
                        "claim": claim,
                        "evidence": evidence,
                        "nli_label": result.label,
                        "nli_scores": result.scores,
                    },
                ))

            # 检查 neutral + 高置信度
            elif (result.label == "neutral" and
                  frame.confidence > self.neutral_high_confidence_threshold):
                issues.append(Issue(
                    frame_idx=frame_idx,
                    issue_type="O1_overclaim",
                    severity=2,
                    rule_id="NLI.neutral_overclaim",
                    message=f"High confidence claim ({frame.confidence:.2f}) but evidence is neutral.",
                    evidence_trace={
                        "claim": claim,
                        "evidence": evidence,
                        "claim_confidence": frame.confidence,
                        "nli_label": result.label,
                        "nli_scores": result.scores,
                    },
                ))

        return issues


# ============================================================
# 组合 Verifier
# ============================================================

class CombinedVerifier:
    """组合 Rule-based 和 NLI Verifier"""

    def __init__(
        self,
        use_rules: bool = True,
        use_nli: bool = True,
        nli_backend: str = "rule",
    ):
        self.use_rules = use_rules
        self.use_nli = use_nli

        if use_rules:
            from .rules import RuleBasedVerifier
            self.rule_verifier = RuleBasedVerifier().add_default_rules()
        else:
            self.rule_verifier = None

        if use_nli:
            self.nli_verifier = NLIVerifier(backend=nli_backend)
        else:
            self.nli_verifier = None

    def verify(self, generation: Generation, tokens: List[Token]) -> List[Issue]:
        """运行所有 verifier"""
        issues = []

        if self.rule_verifier:
            issues.extend(self.rule_verifier.verify(generation, tokens))

        if self.nli_verifier:
            issues.extend(self.nli_verifier.verify(generation, tokens))

        # 去重（同一 frame 的相同类型 issue）
        seen = set()
        unique_issues = []
        for issue in issues:
            key = (issue.frame_idx, issue.issue_type, issue.rule_id)
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)

        return unique_issues


# ============================================================
# 便捷函数
# ============================================================

def create_nli_verifier(
    backend: str = "rule",
    **kwargs,
) -> NLIVerifier:
    """创建 NLI Verifier"""
    return NLIVerifier(backend=backend, **kwargs)


def create_combined_verifier(
    use_rules: bool = True,
    use_nli: bool = True,
    nli_backend: str = "rule",
) -> CombinedVerifier:
    """创建组合 Verifier"""
    return CombinedVerifier(
        use_rules=use_rules,
        use_nli=use_nli,
        nli_backend=nli_backend,
    )
