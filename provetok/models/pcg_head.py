"""PCG Head: Proof-Carrying Generation Head

根据 proposal §4.3:
- 从 token embeddings 生成 structured finding frames
- 每个 frame = (finding_type, polarity, laterality, confidence)
- Citation attention: 为每个 frame 选择 supporting tokens
- Constrained decoding: 只允许输出合法域 V_slot 内的值
- Accept probability q_k: 用于 refusal calibration

结构:
    token_embs (N, D) → pooled → slot classifiers → frames + citations
"""
from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import Token, Frame, Generation
from ..pcg.schema import (
    FINDINGS,
    LATERALITY,
    POLARITY,
    LOCATIONS,
    SIZE_BINS,
    SEVERITY_LEVELS,
    UNCERTAINTY,
)
from ..pcg.narrative import render_generation_text


class CitationAttention(nn.Module):
    """Citation Attention: 为每个 finding query 选择 supporting tokens

    根据 proposal §4.3.2:
    cite(k) = top-K( softmax(q_k^T · T / √d) )
    """

    def __init__(self, emb_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        queries: torch.Tensor,    # (K, D) finding queries
        token_embs: torch.Tensor,  # (N, D) token embeddings
        top_k: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pooled: (K, D) attended token features per finding
            attn_weights: (K, N) attention weights (用于 citation selection)
        """
        K = queries.shape[0]
        N = token_embs.shape[0]

        q = self.q_proj(queries)    # (K, D)
        k = self.k_proj(token_embs) # (N, D)
        v = self.v_proj(token_embs) # (N, D)

        # Multi-head attention
        q = q.view(K, self.num_heads, self.head_dim).transpose(0, 1)  # (H, K, d)
        k = k.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (H, N, d)
        v = v.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (H, N, d)

        scale = math.sqrt(self.head_dim)
        attn = torch.bmm(q, k.transpose(1, 2)) / scale  # (H, K, N)
        attn_weights = F.softmax(attn, dim=-1)            # (H, K, N)

        # Aggregate
        out = torch.bmm(attn_weights, v)  # (H, K, d)
        out = out.transpose(0, 1).contiguous().view(K, -1)  # (K, D)
        pooled = self.out_proj(out)

        # Average attention across heads for citation selection
        avg_attn = attn_weights.mean(dim=0)  # (K, N)

        return pooled, avg_attn


class SlotClassifier(nn.Module):
    """单个 slot 的分类器

    支持 constrained decoding: 只允许输出 valid_values 中的值
    """

    def __init__(self, emb_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (K, D) 或 (B, K, D)
            valid_mask: (num_classes,) bool mask for constrained decoding

        Returns:
            logits: (K, num_classes) 或 (B, K, num_classes)
        """
        logits = self.classifier(features)

        # Constrained decoding: mask invalid values
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float("-inf"))

        return logits


class AcceptProbHead(nn.Module):
    """Accept probability head: 预测 q_k

    根据 proposal §4.3.3:
    q_k = σ(MLP([h_k; max_cite_score; entropy]))
    """

    def __init__(self, emb_dim: int = 32):
        super().__init__()
        # 输入: frame feature + max citation score + slot entropy
        input_dim = emb_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        frame_features: torch.Tensor,  # (K, D)
        max_cite_scores: torch.Tensor,  # (K,)
        slot_entropies: torch.Tensor,   # (K,)
    ) -> torch.Tensor:
        """Returns: q_k (K,) accept probabilities"""
        extra = torch.stack([max_cite_scores, slot_entropies], dim=-1)  # (K, 2)
        combined = torch.cat([frame_features, extra], dim=-1)  # (K, D+2)
        return self.mlp(combined).squeeze(-1)  # (K,)


class PCGHead(nn.Module):
    """Proof-Carrying Generation Head

    从 BET token embeddings 生成:
    1. Finding frames (structured slots)
    2. Citations (token-level pointers)
    3. Accept probability q_k
    4. Refusal decision

    架构:
        token_embs → citation attention → slot classifiers → frames
                                        → accept prob head → q_k
    """

    def __init__(
        self,
        emb_dim: int = 32,
        num_findings: int = None,
        num_heads: int = 4,
        hidden_dim: int = 64,
        top_k_citations: int = 3,
        tau_refuse: float = 0.55,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.top_k_citations = top_k_citations
        self.tau_refuse = tau_refuse

        # 自动确定 finding 数量
        if num_findings is None:
            num_findings = len(FINDINGS)
        if num_findings > len(FINDINGS):
            raise ValueError(f"num_findings={num_findings} > len(FINDINGS)={len(FINDINGS)}")
        self.num_findings = num_findings
        self.finding_vocab = FINDINGS[:num_findings]

        # Finding query embeddings (可学习)
        self.finding_queries = nn.Parameter(
            torch.randn(num_findings, emb_dim) / math.sqrt(emb_dim)
        )

        # Citation attention
        self.citation_attn = CitationAttention(emb_dim, num_heads=num_heads)

        # Slot classifiers
        self.finding_classifier = SlotClassifier(emb_dim, num_findings, hidden_dim)
        self.polarity_classifier = SlotClassifier(emb_dim, len(POLARITY), hidden_dim)
        self.laterality_classifier = SlotClassifier(emb_dim, len(LATERALITY), hidden_dim)
        self.location_classifier = SlotClassifier(emb_dim, len(LOCATIONS), hidden_dim)
        self.size_classifier = SlotClassifier(emb_dim, len(SIZE_BINS), hidden_dim)
        self.severity_classifier = SlotClassifier(emb_dim, len(SEVERITY_LEVELS), hidden_dim)
        self.uncertainty_classifier = SlotClassifier(emb_dim, len(UNCERTAINTY), hidden_dim)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Accept probability head
        self.accept_head = AcceptProbHead(emb_dim)

    def forward(
        self,
        token_embs: torch.Tensor,
        constrained_vocab: Optional[Dict[str, Set[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            token_embs: (N, D) token embeddings from BET encoder
            constrained_vocab: slot -> valid values (for constrained decoding)

        Returns:
            Dict with:
                finding_logits: (K, num_findings)
                polarity_logits: (K, len(POLARITY))
                laterality_logits: (K, len(LATERALITY))
                confidence: (K,)
                attn_weights: (K, N)
                q_k: (K,)
                frame_features: (K, D)
        """
        N = token_embs.shape[0]
        K = self.num_findings

        # 1. Citation attention
        frame_features, attn_weights = self.citation_attn(
            self.finding_queries, token_embs, top_k=self.top_k_citations
        )

        # 2. Slot classification (with optional constraints)
        finding_mask = self._build_mask(constrained_vocab, "finding_type", self.finding_vocab)
        polarity_mask = self._build_mask(constrained_vocab, "polarity", POLARITY)
        laterality_mask = self._build_mask(constrained_vocab, "laterality", LATERALITY)
        location_mask = self._build_mask(constrained_vocab, "location", LOCATIONS)
        size_mask = self._build_mask(constrained_vocab, "size_bin", SIZE_BINS)
        severity_mask = self._build_mask(constrained_vocab, "severity", SEVERITY_LEVELS)
        uncertainty_mask = self._build_mask(constrained_vocab, "uncertainty", UNCERTAINTY)

        finding_logits = self.finding_classifier(frame_features, finding_mask)
        polarity_logits = self.polarity_classifier(frame_features, polarity_mask)
        laterality_logits = self.laterality_classifier(frame_features, laterality_mask)
        location_logits = self.location_classifier(frame_features, location_mask)
        size_logits = self.size_classifier(frame_features, size_mask)
        severity_logits = self.severity_classifier(frame_features, severity_mask)
        uncertainty_logits = self.uncertainty_classifier(frame_features, uncertainty_mask)

        # 3. Confidence
        confidence = self.confidence_head(frame_features).squeeze(-1)  # (K,)

        # 4. Accept probability
        max_cite_scores = attn_weights.max(dim=-1).values  # (K,)
        # Slot entropy as uncertainty measure
        slot_entropy = -(F.softmax(finding_logits, dim=-1) *
                         F.log_softmax(finding_logits + 1e-8, dim=-1)).sum(dim=-1)  # (K,)

        q_k = self.accept_head(frame_features, max_cite_scores, slot_entropy)

        return {
            "finding_logits": finding_logits,
            "polarity_logits": polarity_logits,
            "laterality_logits": laterality_logits,
            "location_logits": location_logits,
            "size_logits": size_logits,
            "severity_logits": severity_logits,
            "uncertainty_logits": uncertainty_logits,
            "confidence": confidence,
            "attn_weights": attn_weights,
            "q_k": q_k,
            "frame_features": frame_features,
        }

    def decode(
        self,
        token_embs: torch.Tensor,
        tokens: Optional[List[Token]] = None,
        constrained_vocab: Optional[Dict[str, Set[str]]] = None,
    ) -> Generation:
        """解码为 Generation 对象

        Args:
            token_embs: (N, D)
            tokens: Token list (用于获取 token_id)
            constrained_vocab: 合法域约束

        Returns:
            Generation
        """
        with torch.no_grad():
            out = self.forward(token_embs, constrained_vocab)

        K = self.num_findings
        frames: List[Frame] = []
        citations: Dict[int, List[int]] = {}
        q: Dict[int, float] = {}
        refusal: Dict[int, bool] = {}

        finding_probs = F.softmax(out["finding_logits"], dim=-1)
        polarity_probs = F.softmax(out["polarity_logits"], dim=-1)
        laterality_probs = F.softmax(out["laterality_logits"], dim=-1)
        location_probs = F.softmax(out["location_logits"], dim=-1)
        size_probs = F.softmax(out["size_logits"], dim=-1)
        severity_probs = F.softmax(out["severity_logits"], dim=-1)
        uncertainty_probs = F.softmax(out["uncertainty_logits"], dim=-1)

        for k in range(K):
            # Decode slots
            finding_idx = finding_probs[k].argmax().item()
            polarity_idx = polarity_probs[k].argmax().item()
            laterality_idx = laterality_probs[k].argmax().item()
            location_idx = location_probs[k].argmax().item()
            size_idx = size_probs[k].argmax().item()
            severity_idx = severity_probs[k].argmax().item()
            uncertainty_idx = uncertainty_probs[k].argmax().item()

            finding = self.finding_vocab[finding_idx]
            polarity = POLARITY[polarity_idx]
            laterality = LATERALITY[laterality_idx]
            location = LOCATIONS[location_idx]
            size_bin = SIZE_BINS[size_idx]
            severity = SEVERITY_LEVELS[severity_idx]
            uncertain = (UNCERTAINTY[uncertainty_idx] == "uncertain")
            conf = float(out["confidence"][k].item())

            frames.append(Frame(
                finding=finding,
                polarity=polarity,
                laterality=laterality,
                confidence=conf,
                location=location,
                size_bin=size_bin,
                severity=severity,
                uncertain=uncertain,
            ))

            # Citations: top-k tokens by attention
            attn_k = out["attn_weights"][k]  # (N,)
            top_indices = torch.topk(
                attn_k, k=min(self.top_k_citations, attn_k.shape[0])
            ).indices.tolist()

            if tokens is not None:
                citations[k] = [tokens[i].token_id for i in top_indices]
            else:
                citations[k] = top_indices

            # Accept probability
            q_k = float(out["q_k"][k].item())
            q[k] = q_k

            # Refusal decision
            refusal[k] = (q_k < self.tau_refuse and polarity == "present")

        tmp = Generation(frames=frames, citations=citations, q=q, refusal=refusal, text="")
        return Generation(frames=frames, citations=citations, q=q, refusal=refusal, text=render_generation_text(tmp))

    def _build_mask(
        self,
        constrained_vocab: Optional[Dict[str, Set[str]]],
        slot_name: str,
        all_values: List[str],
    ) -> Optional[torch.Tensor]:
        """构建 constrained decoding mask"""
        if constrained_vocab is None or slot_name not in constrained_vocab:
            return None

        valid = constrained_vocab[slot_name]
        mask = torch.tensor(
            [v in valid for v in all_values],
            dtype=torch.bool,
            device=self.finding_queries.device,
        )
        # 至少保留一个有效值
        if not mask.any():
            return None
        return mask

    def compute_loss(
        self,
        out: Dict[str, Any],
        gt_frames: List[Frame],
    ) -> Dict[str, torch.Tensor]:
        """计算训练 loss

        Args:
            out: forward() 的输出
            gt_frames: ground truth frames

        Returns:
            Dict of loss components
        """
        device = out["finding_logits"].device
        K = min(len(gt_frames), self.num_findings)

        losses = {}

        if K == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "total": zero,
                "finding": zero,
                "polarity": zero,
                "laterality": zero,
                "location": zero,
                "size_bin": zero,
                "severity": zero,
                "uncertainty": zero,
            }

        # Finding classification loss
        gt_finding_ids = []
        gt_polarity_ids = []
        gt_laterality_ids = []
        gt_location_ids = []
        gt_size_ids = []
        gt_severity_ids = []
        gt_uncertainty_ids = []

        for frame in gt_frames[:K]:
            gt_finding_ids.append(
                self.finding_vocab.index(frame.finding) if frame.finding in self.finding_vocab else 0
            )
            gt_polarity_ids.append(
                POLARITY.index(frame.polarity) if frame.polarity in POLARITY else 0
            )
            gt_laterality_ids.append(
                LATERALITY.index(frame.laterality) if frame.laterality in LATERALITY else len(LATERALITY) - 1
            )
            gt_location_ids.append(
                LOCATIONS.index(frame.location) if frame.location in LOCATIONS else LOCATIONS.index("unspecified")
            )
            gt_size_ids.append(
                SIZE_BINS.index(frame.size_bin) if frame.size_bin in SIZE_BINS else SIZE_BINS.index("unspecified")
            )
            gt_severity_ids.append(
                SEVERITY_LEVELS.index(frame.severity) if frame.severity in SEVERITY_LEVELS else SEVERITY_LEVELS.index("unspecified")
            )
            gt_uncertainty_ids.append(
                UNCERTAINTY.index("uncertain" if frame.uncertain else "certain")
            )

        gt_finding = torch.tensor(gt_finding_ids, device=device)
        gt_polarity = torch.tensor(gt_polarity_ids, device=device)
        gt_laterality = torch.tensor(gt_laterality_ids, device=device)
        gt_location = torch.tensor(gt_location_ids, device=device)
        gt_size = torch.tensor(gt_size_ids, device=device)
        gt_severity = torch.tensor(gt_severity_ids, device=device)
        gt_uncertainty = torch.tensor(gt_uncertainty_ids, device=device)

        losses["finding"] = F.cross_entropy(out["finding_logits"][:K], gt_finding)
        losses["polarity"] = F.cross_entropy(out["polarity_logits"][:K], gt_polarity)
        losses["laterality"] = F.cross_entropy(out["laterality_logits"][:K], gt_laterality)
        losses["location"] = F.cross_entropy(out["location_logits"][:K], gt_location)
        losses["size_bin"] = F.cross_entropy(out["size_logits"][:K], gt_size)
        losses["severity"] = F.cross_entropy(out["severity_logits"][:K], gt_severity)
        losses["uncertainty"] = F.cross_entropy(out["uncertainty_logits"][:K], gt_uncertainty)

        losses["total"] = (
            losses["finding"]
            + losses["polarity"]
            + losses["laterality"]
            + losses["location"]
            + losses["size_bin"]
            + losses["severity"]
            + losses["uncertainty"]
        )

        return losses
