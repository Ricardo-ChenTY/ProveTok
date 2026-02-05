"""Training Stages: M0 → M1 → M2 → M3

根据 proposal §8.1:
  M0: Frame-extraction pipeline (frozen encoder, train frame extractor)
  M1: BET tokenizer + PCG head (train BET allocator + PCG slot classifiers)
  M2: + Grounding reward (add grounding loss, finetune with RL/SCST)
  M3: + LLM integration (full pipeline with LLM backend)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class StageConfig:
    """单个训练阶段的配置"""
    name: str
    description: str

    # 模块冻结控制
    freeze_encoder: bool = True
    freeze_pcg_head: bool = False
    freeze_evidence_head: bool = False
    freeze_llm: bool = True

    # 训练参数
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    batch_size: int = 4
    grad_accum_steps: int = 1

    # Loss 权重
    loss_weights: Dict[str, float] = field(default_factory=dict)

    # BET 参数
    budget_tokens: int = 128
    bet_steps: int = 5
    max_depth: int = 4
    epsilon: float = 0.01

    # Verifier 控制
    use_verifier: bool = True
    verifier_refresh_period: int = 1

    # 数据
    dataset_type: str = "synthetic"
    vol_shape: List[int] = field(default_factory=lambda: [64, 128, 128])

    # 其他
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 50


STAGE_CONFIGS: Dict[str, StageConfig] = {
    "M0": StageConfig(
        name="M0",
        description="Frame-extraction pipeline: 训练 frame extractor，冻结 encoder",
        freeze_encoder=True,
        freeze_pcg_head=True,
        freeze_evidence_head=True,
        freeze_llm=True,
        lr=1e-3,
        max_steps=2000,
        loss_weights={"frame_extraction": 1.0},
        use_verifier=False,
        budget_tokens=64,
        bet_steps=0,  # 不做 BET refinement
    ),
    "M1": StageConfig(
        name="M1",
        description="BET tokenizer + PCG head: 训练 BET allocator 和 PCG slot classifiers",
        freeze_encoder=True,
        freeze_pcg_head=False,
        freeze_evidence_head=False,
        freeze_llm=True,
        lr=1e-4,
        max_steps=5000,
        loss_weights={
            "finding": 1.0,
            "polarity": 1.0,
            "laterality": 0.5,
            "location": 0.5,
            "size_bin": 0.5,
            "severity": 0.3,
            "uncertainty": 0.3,
            "verifier": 0.3,
        },
        use_verifier=True,
        budget_tokens=128,
        bet_steps=5,
    ),
    "M2": StageConfig(
        name="M2",
        description="+ Grounding reward: 加入 grounding loss，finetune",
        freeze_encoder=True,
        freeze_pcg_head=False,
        freeze_evidence_head=False,
        freeze_llm=True,
        lr=5e-5,
        max_steps=3000,
        loss_weights={
            "finding": 1.0,
            "polarity": 1.0,
            "laterality": 0.5,
            "location": 0.5,
            "size_bin": 0.5,
            "severity": 0.3,
            "uncertainty": 0.3,
            "verifier": 0.3,
            "grounding": 0.5,
        },
        use_verifier=True,
        budget_tokens=128,
        bet_steps=5,
    ),
    "M3": StageConfig(
        name="M3",
        description="+ LLM integration: 完整 pipeline，接入 LLM backend",
        freeze_encoder=True,
        freeze_pcg_head=False,
        freeze_evidence_head=False,
        freeze_llm=False,  # 可能用 LoRA 微调 LLM
        lr=2e-5,
        max_steps=10000,
        loss_weights={
            "finding": 1.0,
            "polarity": 1.0,
            "laterality": 0.5,
            "location": 0.5,
            "size_bin": 0.5,
            "severity": 0.3,
            "uncertainty": 0.3,
            "verifier": 0.3,
            "grounding": 0.5,
            "language": 1.0,
        },
        use_verifier=True,
        budget_tokens=256,
        bet_steps=5,
    ),
}


def get_stage_config(stage: str) -> StageConfig:
    """获取训练阶段配置"""
    if stage not in STAGE_CONFIGS:
        available = ", ".join(STAGE_CONFIGS.keys())
        raise ValueError(f"Unknown stage: {stage}. Available: {available}")
    return STAGE_CONFIGS[stage]
