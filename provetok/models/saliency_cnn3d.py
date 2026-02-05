from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SaliencyCNN3DConfig:
    """Lightweight 3D saliency model producing a per-voxel lesion probability map.

    Output is a single-channel logits volume aligned with the (resized) input volume.
    """

    in_channels: int = 1
    base_channels: int = 16
    num_layers: int = 4
    dropout: float = 0.0


class SaliencyCNN3D(nn.Module):
    def __init__(self, cfg: SaliencyCNN3DConfig):
        super().__init__()
        self.cfg = cfg

        c0 = int(cfg.base_channels)
        layers: list[nn.Module] = []
        in_c = int(cfg.in_channels)
        for i in range(int(cfg.num_layers)):
            out_c = c0 if i == 0 else c0
            layers.append(nn.Conv3d(in_c, out_c, kernel_size=3, padding=1))
            layers.append(nn.SiLU())
            if float(cfg.dropout) > 0:
                layers.append(nn.Dropout3d(p=float(cfg.dropout)))
            in_c = out_c
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv3d(c0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect (B,1,D,H,W)
        if x.dim() != 5:
            raise ValueError(f"Expected input shape (B,C,D,H,W), got shape={tuple(x.shape)}")
        y = self.backbone(x)
        return self.head(y)  # (B,1,D,H,W)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def to_checkpoint(self) -> Dict[str, Any]:
        return {"cfg": asdict(self.cfg), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any]) -> "SaliencyCNN3D":
        cfg_dict = ckpt.get("cfg") or {}
        cfg = SaliencyCNN3DConfig(**{k: cfg_dict[k] for k in cfg_dict.keys() if k in SaliencyCNN3DConfig.__annotations__})
        model = cls(cfg)
        sd = ckpt.get("state_dict") or {}
        model.load_state_dict(sd)
        return model


def save_saliency_cnn3d(path: str, model: SaliencyCNN3D, *, extra: Optional[Dict[str, Any]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"saliency_cnn3d": model.to_checkpoint()}
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, str(p))


def load_saliency_cnn3d(path: str, *, map_location: str | torch.device = "cpu") -> SaliencyCNN3D:
    # PyTorch 2.6 changed `torch.load` default to `weights_only=True`, which can
    # reject checkpoints that include non-allowlisted globals in auxiliary metadata.
    # Prefer safe loading when available, but fall back to full unpickling for
    # backward compatibility with locally-produced artifacts.
    try:
        ckpt = torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch: no `weights_only` kwarg.
        ckpt = torch.load(str(path), map_location=map_location)
    except Exception:
        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    payload = ckpt.get("saliency_cnn3d") if isinstance(ckpt, dict) else None
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid saliency checkpoint at {path!r}: missing 'saliency_cnn3d' dict")
    return SaliencyCNN3D.from_checkpoint(payload)
