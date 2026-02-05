"""Verifier 模块

提供两种验证方式：
1. Rule-based: 基于解剖学知识和逻辑规则
2. NLI-based: 使用 NLI 模型判断 claim-evidence 一致性

Usage:
    # 简单使用（默认规则）
    from provetok.verifier import verify
    issues = verify(generation, tokens)

    # 自定义规则
    from provetok.verifier import create_verifier
    verifier = create_verifier(rules=["U1.0", "U1.1", "M1.0"])
    issues = verifier.verify(generation, tokens)

    # 使用 NLI
    from provetok.verifier import create_nli_verifier
    verifier = create_nli_verifier(backend="transformers")
    issues = verifier.verify(generation, tokens)

    # 组合使用
    from provetok.verifier import create_combined_verifier
    verifier = create_combined_verifier(use_rules=True, use_nli=True)
    issues = verifier.verify(generation, tokens)
"""
from .rules import (
    # 主函数
    verify,
    create_verifier,
    get_default_verifier,
    # 主类
    RuleBasedVerifier,
    VerificationRule,
    RuleConfig,
    # U1 规则
    U1_NoCitation,
    U1_LowScore,
    U1_HighUncertainty,
    U1_InsufficientCoverage,
    U1_CitationRelevance,
    # O1 规则
    O1_CoarseLevelOnly,
    O1_SeverityMismatch,
    O1_SizeClaim,
    # I1 规则
    I1_BilateralCitation,
    I1_AnatomicalMismatch,
    I1_ConflictingFindings,
    I1_PolarityConfidence,
    # M1 规则
    M1_MissingLaterality,
    M1_MissingFinding,
    M1_LowConfidenceNoExplanation,
    # 知识库
    LOCATION_LATERALITY_MAP,
    COMPATIBLE_LOCATIONS,
    FINDING_SEVERITY_MAP,
    SIZE_THRESHOLDS,
)

from .nli import (
    # NLI 结果
    NLIResult,
    # 文本转换
    frame_to_claim,
    tokens_to_evidence,
    # NLI 后端
    NLIBackend,
    RuleBasedNLI,
    TransformersNLI,
    OpenAINLI,
    # NLI Verifier
    NLIVerifier,
    CombinedVerifier,
    # 便捷函数
    create_nli_verifier,
    create_combined_verifier,
)

__all__ = [
    # 主函数
    "verify",
    "create_verifier",
    "get_default_verifier",
    # Rule-based
    "RuleBasedVerifier",
    "VerificationRule",
    "RuleConfig",
    # 规则类
    "U1_NoCitation",
    "U1_LowScore",
    "U1_HighUncertainty",
    "U1_InsufficientCoverage",
    "U1_CitationRelevance",
    "O1_CoarseLevelOnly",
    "O1_SeverityMismatch",
    "O1_SizeClaim",
    "I1_BilateralCitation",
    "I1_AnatomicalMismatch",
    "I1_ConflictingFindings",
    "I1_PolarityConfidence",
    "M1_MissingLaterality",
    "M1_MissingFinding",
    "M1_LowConfidenceNoExplanation",
    # 知识库
    "LOCATION_LATERALITY_MAP",
    "COMPATIBLE_LOCATIONS",
    "FINDING_SEVERITY_MAP",
    "SIZE_THRESHOLDS",
    # NLI
    "NLIResult",
    "frame_to_claim",
    "tokens_to_evidence",
    "NLIBackend",
    "RuleBasedNLI",
    "TransformersNLI",
    "OpenAINLI",
    "NLIVerifier",
    "CombinedVerifier",
    "create_nli_verifier",
    "create_combined_verifier",
]
