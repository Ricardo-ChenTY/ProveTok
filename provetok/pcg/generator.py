from __future__ import annotations
from typing import List, Dict
import math
import torch
from ..types import Token, Frame, Generation
from .schema import FINDINGS
from .narrative import render_generation_text
from .toy_queries import toy_queries

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

class ToyPCG:
    """Toy Proof-Carrying Generator:
    - output frames (one per finding)
    - attach citations (top-k tokens by attention)
    - q_k: toy accept probability
    - refusal: if confidence too low for positive claim
    """
    def __init__(
        self,
        emb_dim: int = 32,
        topk: int = 3,
        seed: int = 0,
        *,
        query_seed: int = 0,
        score_bias: float = 0.0,
        refusal_threshold: float = 0.55,
        citation_strategy: str = "attention",  # "attention" | "score" | "score_interleave" | "attn_score"
        polarity_strategy: str = "confidence",  # "confidence" | "support"
        q_strategy: str = "confidence",  # "confidence" | "support"
    ):
        self.emb_dim = emb_dim
        self.topk = topk
        self.seed = int(seed)
        self.query_seed = int(query_seed)
        self.score_bias = float(score_bias)
        self.refusal_threshold = float(refusal_threshold)
        self.citation_strategy = str(citation_strategy)
        self.polarity_strategy = str(polarity_strategy)
        self.q_strategy = str(q_strategy)
        # Queries are deterministic and shared with verifier rules (auditability).
        self.queries = toy_queries(FINDINGS, emb_dim=self.emb_dim, seed=self.query_seed)

    def __call__(self, tokens: List[Token]) -> Generation:
        if not tokens:
            return Generation(frames=[], citations={}, q={}, refusal={}, text="")

        T = torch.stack([t.embedding for t in tokens], dim=0)  # (N,D)
        logits = (self.queries.to(device=T.device, dtype=T.dtype) @ T.T)  # (K,N)
        scores = torch.tensor([float(t.score) for t in tokens], dtype=T.dtype, device=T.device)  # (N,)
        if self.score_bias != 0.0:
            logits = logits + self.score_bias * scores.unsqueeze(0)
        att = torch.softmax(logits, dim=1)

        citations: Dict[int, List[int]] = {}
        idx_to_token_id = [int(t.token_id) for t in tokens]
        token_by_id = {int(t.token_id): t for t in tokens}
        K = len(FINDINGS)
        pooled_mean = (att @ T).mean(dim=1)  # (K,)
        conf_raw = torch.tanh(pooled_mean).detach().cpu().tolist()

        polarity: List[str] = []
        confidence: List[float] = []
        for k in range(K):
            if self.polarity_strategy == "support":
                support = float((att[k] * scores).sum().detach().cpu().item())  # in [0,1] when scores are probabilities
                # Calibrate threshold lower than 0.5 because token scores are often
                # conservative (esp. heuristics); center at ~0.25 keeps positives
                # from collapsing to "all absent" while remaining score-sensitive.
                p_present = _sigmoid(6.0 * (support - 0.25))
            else:
                p_present = _sigmoid(3.0 * float(conf_raw[k]))
            pol = "present" if p_present >= 0.5 else "absent"
            polarity.append(pol)
            confidence.append(float(p_present if pol == "present" else (1.0 - p_present)))

        ranked_by_score: List[Token] = []
        if self.citation_strategy in ("score", "score_interleave"):
            ranked_by_score = sorted(tokens, key=lambda t: (-float(t.score), int(t.token_id)))

        if self.citation_strategy == "attn_score":
            # Select citations by attention * token.score, per frame.
            #
            # Motivation: score-only selection can become brittle as token count grows
            # (extreme-value false positives). Multiplying by attention keeps citations
            # query-conditioned while still benefiting from lesionness/scoring.
            scores_cpu = scores.detach().cpu().tolist()
            for k in range(K):
                a = att[k].detach().cpu().tolist()
                combined = [float(ai) * float(si) for ai, si in zip(a, scores_cpu)]
                idx = sorted(
                    range(len(combined)),
                    key=lambda i: (-float(combined[i]), int(idx_to_token_id[i])),
                )[: min(int(self.topk), len(combined))]
                citations[k] = [idx_to_token_id[i] for i in idx]
        elif self.citation_strategy == "score_interleave":
            # Score-interleave across *present* frames so that positive claims cite from
            # the highest-score pool instead of being diluted by absent findings.
            present = [k for k in range(K) if polarity[k] == "present"]
            if present and ranked_by_score:
                P = len(present)
                pool_size = min(len(ranked_by_score), int(self.topk) * P)
                pool = ranked_by_score[:pool_size]
                take = min(int(self.topk), len(pool))
                for j, k in enumerate(present):
                    cites = []
                    for t in range(take):
                        idx = j + t * P
                        if idx >= len(pool):
                            break
                        cites.append(int(pool[idx].token_id))
                    citations[k] = cites
            # Default: empty citations (no present frames or empty pool).
            for k in range(K):
                citations.setdefault(k, [])
        elif self.citation_strategy == "score":
            if ranked_by_score:
                top = [int(t.token_id) for t in ranked_by_score[: min(int(self.topk), len(ranked_by_score))]]
            else:
                top = []
            for k in range(K):
                citations[k] = list(top)
        else:
            for k in range(K):
                top_idx = torch.topk(att[k], k=min(int(self.topk), int(att.shape[1]))).indices.tolist()
                citations[k] = [idx_to_token_id[i] for i in top_idx if 0 <= int(i) < len(idx_to_token_id)]

        frames: List[Frame] = []
        q: Dict[int, float] = {}
        refusal: Dict[int, bool] = {}
        for k, finding in enumerate(FINDINGS):
            pol = polarity[k]
            conf = float(confidence[k])

            if pol == "absent":
                laterality = "unspecified"
            else:
                laterality = (
                    "left"
                    if (citations.get(k) and int(citations[k][0]) % 2 == 0)
                    else ("right" if citations.get(k) else "unspecified")
                )

            if self.q_strategy == "support":
                cite_scores = [float(token_by_id[tid].score) for tid in citations.get(k, []) if tid in token_by_id]
                support = max(cite_scores) if cite_scores else 0.0
                q_k = float(max(0.05, min(0.95, 0.2 + 0.8 * support)))
            else:
                q_k = float(max(0.05, min(0.95, 0.2 + 0.8 * conf)))
            q[k] = q_k
            refusal[k] = bool(conf < self.refusal_threshold and pol == "present")

            frames.append(Frame(finding=finding, polarity=pol, laterality=laterality, confidence=float(conf)))

        tmp = Generation(frames=frames, citations=citations, q=q, refusal=refusal, text="")
        return Generation(frames=frames, citations=citations, q=q, refusal=refusal, text=render_generation_text(tmp))
