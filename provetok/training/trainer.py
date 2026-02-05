"""Trainer: 统一训练循环

支持 M0-M3 各阶段的训练，通过 StageConfig 控制行为。
在没有 LLM 的情况下 (M0-M2) 可以完整运行。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from .stages import StageConfig, get_stage_config
from ..types import Token, Frame, Generation, Issue
from ..data.dataset import make_dataloader
from ..models.pcg_head import PCGHead
from ..bet.tokenize import encode_tokens
from ..bet.refine_loop import run_refine_loop
from ..bet.evidence_head import EvidenceHead
from ..grid.cells import root_cell, split, Cell, parse_cell_id, cell_bounds
from ..pcg.schema import POLARITY, LATERALITY, LOCATIONS, SIZE_BINS, SEVERITY_LEVELS, UNCERTAINTY


@dataclass
class TrainerConfig:
    """Trainer 配置"""
    stage: str = "M1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./outputs"
    seed: int = 42
    emb_dim: int = 32

    # 数据相关
    dataset_cfg: Dict[str, Any] = field(default_factory=lambda: {
        "dataset_type": "synthetic",
        "num_samples": 100,
        "vol_shape": [64, 128, 128],
        "batch_size": 4,
    })

    # 覆盖 stage config 的值
    overrides: Dict[str, Any] = field(default_factory=dict)


class Trainer:
    """ProveTok 统一训练器

    使用方式:
        cfg = TrainerConfig(stage="M1")
        trainer = Trainer(cfg)
        trainer.train()
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.stage_config = get_stage_config(config.stage)

        # 应用覆盖
        for k, v in config.overrides.items():
            if hasattr(self.stage_config, k):
                setattr(self.stage_config, k, v)

        self.device = torch.device(config.device)
        self.emb_dim = config.emb_dim

        # 初始化模块
        self._init_modules()
        self._init_optimizer()
        self._init_dataloader()

        # 训练状态
        self.global_step = 0
        self.best_metric = float("inf")
        self.logs: List[Dict[str, Any]] = []

    def _init_modules(self):
        """初始化各模块"""
        # PCG Head
        self.pcg_head = PCGHead(emb_dim=self.emb_dim).to(self.device)

        # Evidence Head
        self.evidence_head = EvidenceHead(emb_dim=self.emb_dim).to(self.device)

        # Deterministic verifier (non-trainable)
        from ..verifier.rules import create_verifier

        self.verifier = create_verifier()

        # 应用冻结
        sc = self.stage_config
        if sc.freeze_pcg_head:
            for p in self.pcg_head.parameters():
                p.requires_grad = False
        if sc.freeze_evidence_head:
            for p in self.evidence_head.parameters():
                p.requires_grad = False

    def _init_optimizer(self):
        """初始化优化器"""
        params = []
        for module in [self.pcg_head, self.evidence_head]:
            params.extend([p for p in module.parameters() if p.requires_grad])

        if not params:
            # 如果没有可训练参数，创建 dummy
            self.optimizer = None
            return

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.stage_config.lr,
            weight_decay=self.stage_config.weight_decay,
        )

        # Warmup + cosine scheduler
        total_steps = self.stage_config.max_steps
        warmup = self.stage_config.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return max(0.1, 0.5 * (1.0 + __import__("math").cos(3.14159 * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _init_dataloader(self):
        """初始化数据加载器"""
        self.train_loader = make_dataloader(self.config.dataset_cfg, split="train")
        self.val_loader = make_dataloader(self.config.dataset_cfg, split="val")

    def train(self) -> Dict[str, Any]:
        """执行训练循环

        Returns:
            训练结果摘要
        """
        sc = self.stage_config
        output_dir = Path(self.config.output_dir) / sc.name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Training Stage: {sc.name} ===")
        print(f"  Description: {sc.description}")
        print(f"  Max steps: {sc.max_steps}")
        print(f"  Device: {self.device}")
        print(f"  Trainable params: {self._count_params():,}")

        data_iter = iter(self.train_loader)
        t0 = time.time()

        for step in range(sc.max_steps):
            self.global_step = step

            # 获取 batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # 训练一步
            step_logs = self._train_step(batch)

            # Logging
            if step % sc.log_every == 0:
                elapsed = time.time() - t0
                step_logs["step"] = step
                step_logs["elapsed_s"] = elapsed
                step_logs["lr"] = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0
                self.logs.append(step_logs)
                print(f"  [Step {step:5d}] loss={step_logs.get('loss', 0):.4f} "
                      f"lr={step_logs.get('lr', 0):.2e} "
                      f"({elapsed:.1f}s)")

            # Eval
            if step > 0 and step % sc.eval_every == 0:
                eval_result = self._eval_step()
                print(f"  [Eval  {step:5d}] {eval_result}")

            # Save
            if step > 0 and step % sc.save_every == 0:
                self._save_checkpoint(output_dir / f"ckpt_step{step}.pt")

        # Final save
        self._save_checkpoint(output_dir / "ckpt_final.pt")

        # Save logs
        with open(output_dir / "train_logs.json", "w") as f:
            json.dump(self.logs, f, indent=2)

        total_time = time.time() - t0
        print(f"=== Training done: {sc.max_steps} steps in {total_time:.1f}s ===")

        return {
            "stage": sc.name,
            "total_steps": sc.max_steps,
            "total_time_s": total_time,
            "final_loss": self.logs[-1].get("loss", 0) if self.logs else 0,
        }

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """单步训练"""
        self.pcg_head.train()
        self.evidence_head.train()

        volumes = batch["volume"].to(self.device)  # (B, D, H, W)
        gt_frames_batch = batch["frames"]           # List[List[Frame]]
        lesion_masks_batch = batch.get("lesion_masks", [{} for _ in range(volumes.shape[0])])

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        log = {}

        # 对 batch 中每个样本分别处理（因为 cell 数量不固定）
        B = volumes.shape[0]
        sc = self.stage_config
        for b in range(B):
            vol = volumes[b]  # (D, H, W)
            gt_frames = gt_frames_batch[b]
            lesion_masks = lesion_masks_batch[b] if isinstance(lesion_masks_batch, list) else {}

            # 1. BET tokenization
            sample_seed = int(self.config.seed) + int(self.global_step) * 10_000 + int(b)
            vol_cpu = vol.detach().cpu()

            if int(sc.bet_steps) > 0:
                # BET refinement is a search procedure; run it without gradients and on CPU for determinism.
                @torch.no_grad()
                def _gen_fn(toks: List[Token]) -> Generation:
                    if not toks:
                        return Generation(frames=[], citations={}, q={}, refusal={}, text="")
                    token_embs = torch.stack([t.embedding for t in toks]).to(self.device)
                    return self.pcg_head.decode(token_embs, toks)

                def _ver_fn(gen: Generation, toks: List[Token]) -> List[Issue]:
                    return self.verifier.verify(gen, toks)

                refine = run_refine_loop(
                    volume=vol_cpu,
                    budget_tokens=int(sc.budget_tokens),
                    steps=int(sc.bet_steps),
                    generator_fn=_gen_fn,
                    verifier_fn=_ver_fn,
                    emb_dim=int(self.emb_dim),
                    seed=int(sample_seed),
                    evidence_head=None,          # keep allocation device-agnostic for training smoke
                    use_evidence_head=False,     # deterministic allocator (evidence traces + uncertainty)
                    epsilon=float(sc.epsilon),
                    max_depth=int(sc.max_depth),
                    verifier_refresh_period=int(sc.verifier_refresh_period),
                )
                tokens = refine.tokens
            else:
                cells = [root_cell()]
                tokens = encode_tokens(vol_cpu, cells, emb_dim=self.emb_dim, seed=sample_seed)

            if not tokens:
                continue

            # 2. 收集 token embeddings
            token_embs = torch.stack([t.embedding for t in tokens]).to(self.device)  # (N, D)

            # 3. PCG Head forward
            pcg_out = self.pcg_head(token_embs)

            # 4. 计算 PCG loss
            pcg_losses = self.pcg_head.compute_loss(pcg_out, gt_frames)

            # 4.1 额外 loss：citation 弱监督 / verifier-driven q_k / grounding consistency / evidence head
            extra_losses: Dict[str, torch.Tensor] = {}

            K = min(len(gt_frames), self.pcg_head.num_findings)
            N = token_embs.shape[0]

            if K > 0 and N > 0:
                attn = pcg_out.get("attn_weights")  # (K_all, N)
                q_k = pcg_out.get("q_k")            # (K_all,)

                # ----------------------------
                # Citation weak supervision
                # ----------------------------
                if attn is not None:
                    vol_shape = tuple(vol.shape)

                    def _best_token_index(frame_idx: int) -> int:
                        # If we have a pixel-level mask for this frame, pick token with max overlap.
                        if isinstance(lesion_masks, dict) and frame_idx in lesion_masks and lesion_masks[frame_idx] is not None:
                            mask = lesion_masks[frame_idx]
                            best_i = 0
                            best_overlap = -1.0
                            for i_tok, t in enumerate(tokens):
                                cell = parse_cell_id(t.cell_id)
                                if cell is None:
                                    continue
                                slc = cell_bounds(cell, vol_shape)
                                region = np.asarray(mask[slc[0], slc[1], slc[2]], dtype=bool)
                                denom = float(region.size)
                                overlap = float(region.sum()) / denom if denom > 0 else 0.0
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_i = i_tok
                            if best_overlap > 0.0:
                                return int(best_i)

                        # Fallback: max evidence score (deterministic)
                        return int(max(range(len(tokens)), key=lambda i_tok: float(tokens[i_tok].score)))

                    cite_losses = []
                    for k in range(K):
                        tgt = _best_token_index(k)
                        cite_losses.append(-torch.log(attn[k, tgt].clamp_min(1e-8)))
                    if cite_losses:
                        extra_losses["citation"] = torch.stack(cite_losses).mean()

                # ----------------------------
                # Verifier-driven q_k loss
                # ----------------------------
                if sc.use_verifier and q_k is not None:
                    with torch.no_grad():
                        frames_pred: List[Frame] = []
                        citations_pred: Dict[int, List[int]] = {}
                        q_pred: Dict[int, float] = {}
                        refusal_pred: Dict[int, bool] = {}

                        for k in range(K):
                            finding_idx = int(pcg_out["finding_logits"][k].argmax().item())
                            polarity_idx = int(pcg_out["polarity_logits"][k].argmax().item())
                            laterality_idx = int(pcg_out["laterality_logits"][k].argmax().item())
                            location_idx = int(pcg_out["location_logits"][k].argmax().item())
                            size_idx = int(pcg_out["size_logits"][k].argmax().item())
                            severity_idx = int(pcg_out["severity_logits"][k].argmax().item())
                            uncertainty_idx = int(pcg_out["uncertainty_logits"][k].argmax().item())

                            frames_pred.append(
                                Frame(
                                    finding=self.pcg_head.finding_vocab[finding_idx],
                                    polarity=POLARITY[polarity_idx],
                                    laterality=LATERALITY[laterality_idx],
                                    confidence=float(pcg_out["confidence"][k].item()),
                                    location=LOCATIONS[location_idx],
                                    size_bin=SIZE_BINS[size_idx],
                                    severity=SEVERITY_LEVELS[severity_idx],
                                    uncertain=(UNCERTAINTY[uncertainty_idx] == "uncertain"),
                                )
                            )

                            top_idx = torch.topk(attn[k], k=min(self.pcg_head.top_k_citations, attn.shape[1])).indices.tolist()
                            citations_pred[k] = [tokens[i_tok].token_id for i_tok in top_idx]
                            q_pred[k] = float(q_k[k].item())
                            refusal_pred[k] = False  # do not bypass verifier during supervision

                        gen_for_verifier = Generation(frames=frames_pred, citations=citations_pred, q=q_pred, refusal=refusal_pred, text="")
                        issues = self.verifier.verify(gen_for_verifier, tokens)

                        has_issue = torch.zeros(K, device=self.device)
                        for iss in issues:
                            if 0 <= iss.frame_idx < K:
                                has_issue[iss.frame_idx] = 1.0
                        q_targets = 1.0 - has_issue  # 1 if verifier-OK, else 0

                    extra_losses["verifier"] = F.binary_cross_entropy(
                        q_k[:K].clamp(1e-4, 1.0 - 1e-4),
                        q_targets,
                    )

                    # Evidence head losses (self-supervised):
                    # - entropy of finding_probs ~ token uncertainty
                    # - issue_head predicts issue-type counts blamed on tokens
                    finding_probs, issue_pred = self.evidence_head(token_embs)
                    entropy = -(finding_probs * (finding_probs + 1e-8).log()).sum(dim=-1)  # (N,)
                    tgt_uncert = torch.tensor([float(t.uncertainty) for t in tokens], device=self.device) * float(
                        np.log(max(finding_probs.shape[-1], 2))
                    )
                    extra_losses["evidence_uncertainty"] = F.mse_loss(entropy, tgt_uncert)

                    issue_targets = torch.zeros((N, 4), device=self.device)
                    cell_id_to_index = {t.cell_id: i_tok for i_tok, t in enumerate(tokens)}
                    type_to_idx = {
                        "U1_unsupported": 0,
                        "O1_overclaim": 1,
                        "I1_inconsistency": 2,
                        "M1_missing_slot": 3,
                    }
                    for iss in issues:
                        j = type_to_idx.get(iss.issue_type)
                        if j is None:
                            continue
                        trace = iss.evidence_trace or {}
                        for cid in trace.get("token_cell_ids", []):
                            i_tok = cell_id_to_index.get(cid)
                            if i_tok is None:
                                continue
                            issue_targets[i_tok, j] += float(iss.severity)

                    extra_losses["evidence_issue"] = F.mse_loss(issue_pred, issue_targets)

                # ----------------------------
                # Grounding consistency loss
                # ----------------------------
                if attn is not None and isinstance(lesion_masks, dict) and lesion_masks:
                    vol_shape = tuple(vol.shape)
                    grounding_losses = []
                    for k in range(K):
                        mask = lesion_masks.get(k)
                        if mask is None:
                            continue
                        overlaps = []
                        for t in tokens:
                            cell = parse_cell_id(t.cell_id)
                            if cell is None:
                                overlaps.append(0.0)
                                continue
                            slc = cell_bounds(cell, vol_shape)
                            region = np.asarray(mask[slc[0], slc[1], slc[2]], dtype=bool)
                            denom = float(region.size)
                            overlaps.append(float(region.sum()) / denom if denom > 0 else 0.0)
                        ov = torch.tensor(overlaps, device=self.device, dtype=attn.dtype)
                        ov = (ov + 1e-6) / (ov.sum() + 1e-6 * ov.numel())
                        grounding_losses.append(-(ov * torch.log(attn[k].clamp_min(1e-8))).sum())
                    if grounding_losses:
                        extra_losses["grounding"] = torch.stack(grounding_losses).mean()

            # 5. 加权求和
            sample_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            for loss_name, loss_val in pcg_losses.items():
                if loss_name == "total":
                    continue
                weight = sc.loss_weights.get(loss_name, 1.0)
                sample_loss = sample_loss + weight * loss_val

            for loss_name, loss_val in extra_losses.items():
                weight = sc.loss_weights.get(loss_name, 1.0)
                sample_loss = sample_loss + weight * loss_val

            total_loss = total_loss + sample_loss / B

        # Backward
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.pcg_head.parameters() if p.requires_grad] +
                [p for p in self.evidence_head.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self.optimizer.step()
            self.scheduler.step()

        log["loss"] = float(total_loss.item())
        return log

    def _eval_step(self) -> Dict[str, float]:
        """评估步骤"""
        self.pcg_head.eval()
        self.evidence_head.eval()

        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                volumes = batch["volume"].to(self.device)
                gt_frames_batch = batch["frames"]

                B = volumes.shape[0]
                for b in range(B):
                    vol = volumes[b]
                    gt_frames = gt_frames_batch[b]

                    sc = self.stage_config
                    sample_seed = int(self.config.seed) + int(self.global_step) * 10_000 + int(b)
                    vol_cpu = vol.detach().cpu()

                    if int(sc.bet_steps) > 0:
                        @torch.no_grad()
                        def _gen_fn(toks: List[Token]) -> Generation:
                            if not toks:
                                return Generation(frames=[], citations={}, q={}, refusal={}, text="")
                            token_embs = torch.stack([t.embedding for t in toks]).to(self.device)
                            return self.pcg_head.decode(token_embs, toks)

                        def _ver_fn(gen: Generation, toks: List[Token]) -> List[Issue]:
                            return self.verifier.verify(gen, toks)

                        refine = run_refine_loop(
                            volume=vol_cpu,
                            budget_tokens=int(sc.budget_tokens),
                            steps=int(sc.bet_steps),
                            generator_fn=_gen_fn,
                            verifier_fn=_ver_fn,
                            emb_dim=int(self.emb_dim),
                            seed=int(sample_seed),
                            evidence_head=None,
                            use_evidence_head=False,
                            epsilon=float(sc.epsilon),
                            max_depth=int(sc.max_depth),
                            verifier_refresh_period=int(sc.verifier_refresh_period),
                        )
                        tokens = refine.tokens
                    else:
                        cells = [root_cell()]
                        tokens = encode_tokens(vol_cpu, cells, emb_dim=self.emb_dim, seed=sample_seed)

                    if not tokens:
                        continue

                    token_embs = torch.stack([t.embedding for t in tokens]).to(self.device)
                    pcg_out = self.pcg_head(token_embs)
                    pcg_losses = self.pcg_head.compute_loss(pcg_out, gt_frames)
                    total_loss += pcg_losses["total"].item()
                    n_samples += 1

                # Only eval on first batch for speed
                break

        avg_loss = total_loss / max(n_samples, 1)
        return {"eval_loss": avg_loss}

    def _save_checkpoint(self, path: Path):
        """保存 checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self.global_step,
            "pcg_head": self.pcg_head.state_dict(),
            "evidence_head": self.evidence_head.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "config": self.config.__dict__,
            "stage": self.stage_config.name,
        }, path)

    def load_checkpoint(self, path: str):
        """加载 checkpoint"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.pcg_head.load_state_dict(ckpt["pcg_head"])
        self.evidence_head.load_state_dict(ckpt["evidence_head"])
        if self.optimizer and ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("step", 0)

    def _count_params(self) -> int:
        """统计可训练参数数量"""
        total = 0
        for module in [self.pcg_head, self.evidence_head]:
            total += sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total
