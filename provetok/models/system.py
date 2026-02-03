"""ProveTok System: 统一封装所有模块

将 BET + PCG Head + Evidence Head + Verifier + LLM Backend
组合为一个可训练/推理的系统。

使用方式:
    system = ProveTokSystem.from_config(cfg)
    # 训练模式
    output = system(batch)
    output.loss.backward()
    # 推理模式
    result = system.inference(volume)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from ..types import Token, Frame, Generation, Issue
from ..grid.cells import root_cell, split, Cell
from ..bet.tokenize import encode_tokens
from ..bet.evidence_head import EvidenceHead, rank_cells_by_delta
from ..bet.refine_loop import run_refine_loop, RefineResult
from .pcg_head import PCGHead
from .llm_backend import BaseLLMBackend, DummyLLM, create_llm_backend


@dataclass
class Output:
    loss: torch.Tensor
    logs: Dict[str, float]
    artifact: Dict[str, Any]


class ProveTokSystem(nn.Module):
    """ProveTok 端到端系统

    组件:
    - PCG Head: slot classification + citation attention + accept prob
    - Evidence Head: Δ(c) 预测用于 BET allocation
    - LLM Backend: 文本生成（可选，M3 阶段）
    """

    def __init__(
        self,
        emb_dim: int = 32,
        num_findings: int = 3,
        budget_tokens: int = 128,
        bet_steps: int = 5,
        max_depth: int = 4,
        epsilon: float = 0.01,
        llm_backend: Optional[BaseLLMBackend] = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.budget_tokens = budget_tokens
        self.bet_steps = bet_steps
        self.max_depth = max_depth
        self.epsilon = epsilon

        # 核心模块
        self.pcg_head = PCGHead(emb_dim=emb_dim, num_findings=num_findings)
        self.evidence_head = EvidenceHead(emb_dim=emb_dim)

        # LLM Backend (不是 nn.Module，单独管理)
        self.llm_backend = llm_backend or DummyLLM()

    def forward(self, batch: Dict[str, Any]) -> Output:
        """训练 forward pass

        Args:
            batch: 包含 volume, frames, sample_id 等

        Returns:
            Output(loss, logs, artifact)
        """
        volumes = batch["volume"]  # (B, D, H, W)
        gt_frames_batch = batch["frames"]

        device = next(self.parameters()).device
        volumes = volumes.to(device)

        B = volumes.shape[0]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        all_pcg_losses: Dict[str, float] = {}

        for b in range(B):
            vol = volumes[b]
            gt_frames = gt_frames_batch[b]

            # BET tokenization (使用 root cell，训练时不做 refinement)
            cells = [root_cell()]
            tokens = encode_tokens(vol, cells, emb_dim=self.emb_dim)

            if not tokens:
                continue

            token_embs = torch.stack([t.embedding for t in tokens]).to(device)

            # PCG Head
            pcg_out = self.pcg_head(token_embs)
            pcg_losses = self.pcg_head.compute_loss(pcg_out, gt_frames)

            total_loss = total_loss + pcg_losses["total"] / B

            # 收集 logs
            for k, v in pcg_losses.items():
                key = f"pcg_{k}"
                all_pcg_losses[key] = all_pcg_losses.get(key, 0) + v.item() / B

        logs: Dict[str, float] = {"loss": total_loss.item()}
        logs.update(all_pcg_losses)

        return Output(loss=total_loss, logs=logs, artifact={})

    @torch.no_grad()
    def inference(
        self,
        volume: torch.Tensor,
        use_refinement: bool = True,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """推理模式: volume → tokens → generation + verification

        Args:
            volume: (D, H, W) 单个 CT volume
            use_refinement: 是否使用 BET refinement loop
            seed: 随机种子

        Returns:
            Dict with generation, tokens, issues, trace
        """
        self.eval()
        device = next(self.parameters()).device

        if use_refinement:
            def generator_fn(tokens: List[Token]) -> Generation:
                if not tokens:
                    return Generation(frames=[], citations={}, q={}, refusal={})
                token_embs = torch.stack([t.embedding for t in tokens]).to(device)
                return self.pcg_head.decode(token_embs, tokens)

            def verifier_fn(gen: Generation, tokens: List[Token]) -> List[Issue]:
                from ..verifier.rules import create_verifier
                v = create_verifier()
                return v.verify(gen, tokens)

            result = run_refine_loop(
                volume=volume,
                budget_tokens=self.budget_tokens,
                steps=self.bet_steps,
                generator_fn=generator_fn,
                verifier_fn=verifier_fn,
                emb_dim=self.emb_dim,
                seed=seed,
                evidence_head=self.evidence_head,
                use_evidence_head=True,
                epsilon=self.epsilon,
                max_depth=self.max_depth,
            )

            return {
                "generation": result.gen,
                "tokens": result.tokens,
                "issues": result.issues,
                "trace": result.trace,
                "final_cells": result.final_cells,
                "stopped_reason": result.stopped_reason,
            }
        else:
            cells = [root_cell()]
            tokens = encode_tokens(volume, cells, emb_dim=self.emb_dim, seed=seed)
            token_embs = torch.stack([t.embedding for t in tokens]).to(device)
            gen = self.pcg_head.decode(token_embs, tokens)

            return {
                "generation": gen,
                "tokens": tokens,
                "issues": [],
                "trace": [],
                "final_cells": cells,
                "stopped_reason": "no_refinement",
            }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ProveTokSystem":
        """从配置字典创建系统"""
        llm_backend = None
        if "llm" in cfg:
            llm_backend = create_llm_backend(**cfg["llm"])

        return cls(
            emb_dim=cfg.get("emb_dim", 32),
            num_findings=cfg.get("num_findings", 3),
            budget_tokens=cfg.get("budget_tokens", 128),
            bet_steps=cfg.get("bet_steps", 5),
            max_depth=cfg.get("max_depth", 4),
            epsilon=cfg.get("epsilon", 0.01),
            llm_backend=llm_backend,
        )
