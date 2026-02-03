from __future__ import annotations
import torch
import torch.nn as nn

class PCGHead(nn.Module):
    """占位 PCG 头：从 tokens/embeddings 生成 frames + citations。
    未来你可以在这里实现：
    - constrained decoding
    - multi-slot classification
    - citation attention / pointer network
    """
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, token_embs: torch.Tensor) -> torch.Tensor:
        # token_embs: (N,emb_dim)
        return self.mlp(token_embs)
