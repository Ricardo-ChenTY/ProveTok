from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LesionnessHeadConfig:
    emb_dim: int = 32
    hidden_dim: int = 64
    dropout: float = 0.1


class LesionnessHead(nn.Module):
    """A lightweight token-level classifier predicting whether a token overlaps a lesion.

    Input: token embedding (D,) or (N,D)
    Output: logits (N,) or scalar
    """

    def __init__(self, cfg: LesionnessHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(int(cfg.emb_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        logits = self.net(x).squeeze(-1)
        return logits.squeeze(0) if squeeze else logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def to_checkpoint(self) -> Dict[str, Any]:
        return {"cfg": asdict(self.cfg), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any]) -> "LesionnessHead":
        cfg_dict = ckpt.get("cfg") or {}
        cfg = LesionnessHeadConfig(**{k: cfg_dict[k] for k in cfg_dict.keys() if k in LesionnessHeadConfig.__annotations__})
        model = cls(cfg)
        sd = ckpt.get("state_dict") or {}
        model.load_state_dict(sd)
        return model


def save_lesionness_head(path: str, model: LesionnessHead, *, extra: Optional[Dict[str, Any]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"lesionness_head": model.to_checkpoint()}
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, str(p))


def load_lesionness_head(path: str, *, map_location: str | torch.device = "cpu") -> LesionnessHead:
    ckpt = torch.load(str(path), map_location=map_location)
    payload = ckpt.get("lesionness_head") if isinstance(ckpt, dict) else None
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid lesionness checkpoint at {path!r}: missing 'lesionness_head' dict")
    return LesionnessHead.from_checkpoint(payload)

