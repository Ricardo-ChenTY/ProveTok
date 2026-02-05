from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CT2RepStrongConfig:
    emb_dim: int = 32
    num_findings: int = 10
    num_classes: int = 3  # none/present/absent
    dropout: float = 0.1


class CT2RepStrong(nn.Module):
    """A small learned baseline: query-attend over evidence tokens, classify finding polarity, emit citations."""

    def __init__(self, cfg: CT2RepStrongConfig):
        super().__init__()
        self.cfg = cfg
        emb_dim = int(cfg.emb_dim)
        self.queries = nn.Parameter(torch.randn(int(cfg.num_findings), emb_dim) / float(max(emb_dim, 1) ** 0.5))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.polarity_head = nn.Linear(emb_dim, int(cfg.num_classes))

    def forward(self, token_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (polarity_logits, attention_weights).

        Args:
            token_emb: (N,D)
        """
        if token_emb.dim() != 2:
            raise ValueError(f"Expected token_emb with shape (N,D), got {tuple(token_emb.shape)}")
        q = self.queries.to(device=token_emb.device, dtype=token_emb.dtype)  # (K,D)
        att_logits = q @ token_emb.T  # (K,N)
        att = torch.softmax(att_logits, dim=1)
        pooled = att @ token_emb  # (K,D)
        pooled = self.dropout(pooled)
        logits = self.polarity_head(pooled)  # (K,C)
        return logits, att

    def to_checkpoint(self) -> Dict[str, Any]:
        return {"cfg": asdict(self.cfg), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any]) -> "CT2RepStrong":
        cfg_dict = ckpt.get("cfg") or {}
        cfg = CT2RepStrongConfig(**{k: cfg_dict[k] for k in cfg_dict.keys() if k in CT2RepStrongConfig.__annotations__})
        model = cls(cfg)
        sd = ckpt.get("state_dict") or {}
        model.load_state_dict(sd)
        return model


def save_ct2rep_strong(path: str, model: CT2RepStrong, *, extra: Optional[Dict[str, Any]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"ct2rep_strong": model.to_checkpoint()}
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, str(p))


def load_ct2rep_strong(path: str, *, map_location: str | torch.device = "cpu") -> CT2RepStrong:
    ckpt = torch.load(str(path), map_location=map_location)
    payload = ckpt.get("ct2rep_strong") if isinstance(ckpt, dict) else None
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid ct2rep_strong checkpoint at {path!r}: missing 'ct2rep_strong' dict")
    return CT2RepStrong.from_checkpoint(payload)

