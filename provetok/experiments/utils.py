"""实验通用工具函数"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import torch
import numpy as np
import json
import os
from datetime import datetime

from ..types import Token, Generation, Issue, Frame
from ..grid.cells import Cell


@dataclass
class ExperimentConfig:
    """实验基础配置"""
    name: str
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./outputs"

    def save(self, path: str):
        """保存配置到 JSON"""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


def set_seed(seed: int):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_output_dir(base_dir: str, exp_name: str) -> str:
    """创建实验输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ============================================================
# Mock 函数（用于测试，实际使用时替换为真实模型）
# ============================================================

def mock_generator_fn(tokens: List[Token]) -> Generation:
    """Mock PCG 生成器

    根据 types.py 的定义:
    - Frame: finding, polarity, laterality, confidence
    - Generation: frames, citations, q, refusal
    """
    num_frames = min(len(tokens) // 4 + 1, 5)
    frames = []
    citations = {}
    q = {}
    refusal = {}

    findings = ["nodule", "opacity", "effusion", "atelectasis", "consolidation"]
    polarities = ["positive", "negative"]
    lateralities = ["left", "right", "bilateral", "unspecified"]

    for i in range(num_frames):
        frame = Frame(
            finding=findings[i % len(findings)],
            polarity=polarities[i % len(polarities)],
            laterality=lateralities[i % len(lateralities)],
            confidence=0.7 + np.random.random() * 0.3,
        )
        frames.append(frame)

        # 随机分配 citations
        if tokens:
            n_citations = min(2 + i, len(tokens))
            citations[i] = [tokens[j].token_id for j in range(n_citations)]

        # acceptance probability
        q[i] = 0.8 + np.random.random() * 0.2
        refusal[i] = False

    return Generation(
        frames=frames,
        citations=citations,
        q=q,
        refusal=refusal,
    )


def mock_verifier_fn(gen: Generation, tokens: List[Token]) -> List[Issue]:
    """Mock Verifier

    根据 types.py 的定义:
    - Issue: frame_idx, issue_type (Literal), severity, rule_id, message, evidence_trace
    - IssueType = Literal["U1_unsupported", "O1_overclaim", "I1_inconsistency", "M1_missing_slot"]
    """
    issue_types = ["U1_unsupported", "O1_overclaim", "I1_inconsistency", "M1_missing_slot"]

    issues = []
    # 模拟一些 issues（随机）
    for i, frame in enumerate(gen.frames):
        if np.random.random() < 0.3:  # 30% 概率有 issue
            issue = Issue(
                frame_idx=i,
                issue_type=issue_types[np.random.randint(len(issue_types))],
                severity=np.random.randint(1, 4),
                rule_id=f"R{np.random.randint(1, 5)}",
                message=f"Potential issue in frame {i}",
                evidence_trace={"source": "mock"},
            )
            issues.append(issue)

    return issues


def create_synthetic_volume(
    shape: Tuple[int, int, int] = (64, 64, 64),
    n_lesions: int = 3,
    seed: int = 42,
) -> Tuple[torch.Tensor, Dict[int, np.ndarray]]:
    """创建合成 CT volume 和 lesion masks

    Args:
        shape: volume 形状
        n_lesions: lesion 数量
        seed: 随机种子

    Returns:
        (volume, lesion_masks) - volume 和每个 lesion 的 mask
    """
    rng = np.random.RandomState(seed)

    # 创建背景
    volume = rng.randn(*shape).astype(np.float32) * 0.1

    # 添加 lesions
    lesion_masks = {}
    for i in range(n_lesions):
        # 随机位置和大小
        center = [rng.randint(s // 4, 3 * s // 4) for s in shape]
        radius = rng.randint(5, 15)

        # 创建球形 lesion
        mask = np.zeros(shape, dtype=bool)
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt(
            (zz - center[0])**2 +
            (yy - center[1])**2 +
            (xx - center[2])**2
        )
        mask[dist <= radius] = True

        lesion_masks[i] = mask

        # 在 volume 中添加高亮
        volume[mask] += 0.5 + rng.random() * 0.3

    return torch.from_numpy(volume).unsqueeze(0), lesion_masks


# ============================================================
# 指标计算工具
# ============================================================

def compute_nlg_metrics(generation: Generation, reference: str = None) -> Dict[str, float]:
    """计算 NLG 指标（简化版）

    实际使用时应该用 BLEU, ROUGE 等标准库

    根据 types.py 的 Generation 定义:
    - frames, citations, q, refusal
    """
    n_frames = len(generation.frames)

    # 计算平均 confidence 和 accept probability
    avg_confidence = np.mean([f.confidence for f in generation.frames]) if generation.frames else 0.0
    avg_q = np.mean(list(generation.q.values())) if generation.q else 0.0
    refusal_rate = sum(generation.refusal.values()) / len(generation.refusal) if generation.refusal else 0.0

    # Mock metrics
    return {
        "n_frames": n_frames,
        "avg_confidence": avg_confidence,
        "avg_accept_prob": avg_q,
        "refusal_rate": refusal_rate,
    }


def aggregate_metrics(
    metrics_list: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """聚合多个样本的指标，返回 (mean, std)"""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    result = {}

    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            result[key] = (np.mean(values), np.std(values))

    return result


def save_results_json(results: Dict, path: str):
    """保存结果到 JSON"""
    # 处理 numpy 类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)


def load_results_json(path: str) -> Dict:
    """从 JSON 加载结果"""
    with open(path, "r") as f:
        return json.load(f)
