from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN3D(nn.Module):
    """Small 3D CNN feature extractor returning a feature map (B,C,D,H,W).

    This is a lightweight "real encoder" to replace the toy patch embedding path.
    - Designed to run on a single GPU (e.g. RTX A6000) with resized volumes.
    - Output channels default to emb_dim to keep TokenEncoder pooling stable.
    """

    def __init__(self, *, in_channels: int = 1, emb_dim: int = 32):
        super().__init__()
        c = int(emb_dim)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(c, c, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(c, c, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(c, c, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect (B,C,D,H,W)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape={tuple(x.shape)}")
        return self.net(x)

