from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _pick_gn_groups(channels: int, max_groups: int) -> int:
    g = int(max(1, min(int(max_groups), int(channels))))
    while g > 1 and (int(channels) % int(g)) != 0:
        g -= 1
    return int(max(1, g))


class _ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, norm_groups: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv3d(int(in_ch), int(out_ch), kernel_size=3, padding=1)
        g = _pick_gn_groups(int(out_ch), int(norm_groups))
        self.norm = nn.GroupNorm(num_groups=int(g), num_channels=int(out_ch))
        self.act = nn.SiLU()
        self.drop = nn.Dropout3d(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, norm_groups: int, dropout: float):
        super().__init__()
        self.conv1 = _ConvGNAct(int(in_ch), int(out_ch), norm_groups=int(norm_groups), dropout=float(dropout))
        self.conv2 = nn.Conv3d(int(out_ch), int(out_ch), kernel_size=3, padding=1)
        g = _pick_gn_groups(int(out_ch), int(norm_groups))
        self.norm2 = nn.GroupNorm(num_groups=int(g), num_channels=int(out_ch))
        self.drop2 = nn.Dropout3d(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.skip = nn.Conv3d(int(in_ch), int(out_ch), kernel_size=1) if int(in_ch) != int(out_ch) else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.drop2(self.norm2(self.conv2(y)))
        y = y + self.skip(x)
        return self.act(y)


@dataclass(frozen=True)
class SaliencyResUNet3DConfig:
    """Residual 3D U-Net saliency model producing a per-voxel lesion probability map."""

    in_channels: int = 1
    base_channels: int = 16
    depth: int = 3
    norm_groups: int = 8
    dropout: float = 0.0


class SaliencyResUNet3D(nn.Module):
    def __init__(self, cfg: SaliencyResUNet3DConfig):
        super().__init__()
        self.cfg = cfg

        base = int(cfg.base_channels)
        depth = int(cfg.depth)
        if depth < 1:
            raise ValueError(f"depth must be >=1, got {depth}")

        self.down: nn.ModuleList = nn.ModuleList()
        self.pool: nn.ModuleList = nn.ModuleList()
        self.upconv: nn.ModuleList = nn.ModuleList()
        self.up: nn.ModuleList = nn.ModuleList()

        in_ch = int(cfg.in_channels)
        chs = [int(base * (2**i)) for i in range(int(depth))]
        for out_ch in chs:
            self.down.append(_ResBlock(in_ch, int(out_ch), norm_groups=int(cfg.norm_groups), dropout=float(cfg.dropout)))
            self.pool.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_ch = int(out_ch)

        bottleneck_ch = int(base * (2**int(depth)))
        self.bottleneck = _ResBlock(int(chs[-1]), int(bottleneck_ch), norm_groups=int(cfg.norm_groups), dropout=float(cfg.dropout))

        cur = int(bottleneck_ch)
        for skip_ch in reversed(chs):
            self.upconv.append(nn.ConvTranspose3d(int(cur), int(skip_ch), kernel_size=2, stride=2))
            self.up.append(_ResBlock(int(skip_ch) * 2, int(skip_ch), norm_groups=int(cfg.norm_groups), dropout=float(cfg.dropout)))
            cur = int(skip_ch)

        self.head = nn.Conv3d(int(cur), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected input shape (B,C,D,H,W), got shape={tuple(x.shape)}")

        skips = []
        y = x
        for down, pool in zip(self.down, self.pool):
            y = down(y)
            skips.append(y)
            y = pool(y)

        y = self.bottleneck(y)

        for i, (upconv, up) in enumerate(zip(self.upconv, self.up)):
            y = upconv(y)
            skip = skips[-(i + 1)]
            if y.shape[2:] != skip.shape[2:]:
                # Center-crop the larger tensor to match shapes.
                sz, sy, sx = (int(skip.shape[2]), int(skip.shape[3]), int(skip.shape[4]))
                yz, yy, yx = (int(y.shape[2]), int(y.shape[3]), int(y.shape[4]))
                tz, ty, tx = (min(sz, yz), min(sy, yy), min(sx, yx))

                def _center_crop(t: torch.Tensor, tz: int, ty: int, tx: int) -> torch.Tensor:
                    z0 = max(0, (int(t.shape[2]) - int(tz)) // 2)
                    y0 = max(0, (int(t.shape[3]) - int(ty)) // 2)
                    x0 = max(0, (int(t.shape[4]) - int(tx)) // 2)
                    return t[:, :, z0 : z0 + int(tz), y0 : y0 + int(ty), x0 : x0 + int(tx)]

                skip = _center_crop(skip, tz, ty, tx)
                y = _center_crop(y, tz, ty, tx)
            y = torch.cat([skip, y], dim=1)
            y = up(y)

        return self.head(y)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def to_checkpoint(self) -> Dict[str, Any]:
        return {"cfg": asdict(self.cfg), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any]) -> "SaliencyResUNet3D":
        cfg_dict = ckpt.get("cfg") or {}
        cfg = SaliencyResUNet3DConfig(**{k: cfg_dict[k] for k in cfg_dict.keys() if k in SaliencyResUNet3DConfig.__annotations__})
        model = cls(cfg)
        sd = ckpt.get("state_dict") or {}
        model.load_state_dict(sd)
        return model


def save_saliency_resunet3d(path: str, model: SaliencyResUNet3D, *, extra: Optional[Dict[str, Any]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"saliency_resunet3d": model.to_checkpoint()}
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, str(p))


def load_saliency_resunet3d(path: str, *, map_location: str | torch.device = "cpu") -> SaliencyResUNet3D:
    try:
        ckpt = torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        ckpt = torch.load(str(path), map_location=map_location)
    except Exception:
        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    payload = ckpt.get("saliency_resunet3d") if isinstance(ckpt, dict) else None
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid saliency checkpoint at {path!r}: missing 'saliency_resunet3d' dict")
    return SaliencyResUNet3D.from_checkpoint(payload)

