from __future__ import annotations
from typing import List, Dict
import math
import torch
from ..types import Token, Frame, Generation
from .schema import FINDINGS

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

class ToyPCG:
    """Toy Proof-Carrying Generator:
    - output frames (one per finding)
    - attach citations (top-k tokens by attention)
    - q_k: toy accept probability
    - refusal: if confidence too low for positive claim
    """
    def __init__(self, emb_dim: int = 32, topk: int = 3, seed: int = 0):
        self.emb_dim = emb_dim
        self.topk = topk
        g = torch.Generator()
        g.manual_seed(seed + 999)
        self.queries = torch.randn((len(FINDINGS), emb_dim), generator=g) / math.sqrt(emb_dim)

    def __call__(self, tokens: List[Token]) -> Generation:
        if not tokens:
            return Generation(frames=[], citations={}, q={}, refusal={})

        T = torch.stack([t.embedding for t in tokens], dim=0)  # (N,D)
        logits = (self.queries @ T.T)                           # (K,N)
        att = torch.softmax(logits, dim=1)

        frames: List[Frame] = []
        citations: Dict[int, List[int]] = {}
        q: Dict[int, float] = {}
        refusal: Dict[int, bool] = {}

        for k, finding in enumerate(FINDINGS):
            pooled = (att[k].unsqueeze(1) * T).sum(dim=0)
            conf_raw = float(torch.tanh(pooled.mean()).item())
            p_present = _sigmoid(3.0 * conf_raw)

            polarity = "present" if p_present >= 0.5 else "absent"
            confidence = p_present if polarity == "present" else (1.0 - p_present)

            top_ids = torch.topk(att[k], k=min(self.topk, att.shape[1])).indices.tolist()
            citations[k] = top_ids

            if polarity == "absent":
                laterality = "unspecified"
            else:
                laterality = "left" if (top_ids and top_ids[0] % 2 == 0) else ("right" if top_ids else "unspecified")

            q_k = float(max(0.05, min(0.95, 0.2 + 0.8 * confidence)))
            q[k] = q_k
            refusal[k] = (confidence < 0.55 and polarity == "present")

            frames.append(Frame(finding=finding, polarity=polarity, laterality=laterality, confidence=float(confidence)))

        return Generation(frames=frames, citations=citations, q=q, refusal=refusal)
