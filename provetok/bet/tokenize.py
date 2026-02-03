from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from ..types import Token
from ..grid.cells import Cell, cell_bounds

# TODO: 替换为真实 3D encoder 输出 feature map，再对 cell 区域 pooling
def _toy_embed_patch(vol: torch.Tensor, slc: Tuple[slice, slice, slice], emb_dim: int, seed: int) -> torch.Tensor:
    patch = vol[slc[0], slc[1], slc[2]]
    mean = patch.mean()
    std = patch.std(unbiased=False)
    mn = patch.min()
    mx = patch.max()
    stats = torch.stack([mean, std, mn, mx]).float()

    g = torch.Generator(device=stats.device)
    g.manual_seed(seed + 12345)
    W = torch.randn((emb_dim, 4), generator=g, device=stats.device) / np.sqrt(4)
    emb = torch.tanh(W @ stats)
    return emb

def encode_tokens(volume: torch.Tensor, cells: List[Cell], emb_dim: int = 32, seed: int = 0) -> List[Token]:
    tokens: List[Token] = []
    token_id = 0
    for c in sorted(cells, key=lambda x: x.id()):
        slc = cell_bounds(c, shape=tuple(volume.shape))
        emb = _toy_embed_patch(volume, slc, emb_dim=emb_dim, seed=seed + (hash(c.id()) % (2**31 - 1)))
        patch = volume[slc[0], slc[1], slc[2]]
        var = patch.var(unbiased=False).item()
        score = float(1.0 / (1.0 + np.exp(-3.0 * (var - 0.5))))
        uncertainty = float(1.0 - score)
        tokens.append(Token(token_id=token_id, cell_id=c.id(), level=c.level, embedding=emb, score=score, uncertainty=uncertainty))
        token_id += 1
    return tokens
