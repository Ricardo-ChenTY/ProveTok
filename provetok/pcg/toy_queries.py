from __future__ import annotations

import hashlib
import math
from typing import Iterable, List, Optional

import torch


def _stable_int_hash(text: str) -> int:
    """Stable 32-bit hash for determinism across Python processes."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="big", signed=False)


def toy_query_vector(
    key: str,
    emb_dim: int,
    *,
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Deterministically generate a query vector for a string key.

    Shared by ToyPCG and verifier rules so that citation relevance is auditable.
    """
    emb_dim = int(emb_dim)
    g = torch.Generator()
    g.manual_seed(int(seed) + _stable_int_hash(str(key).lower()))
    q = torch.randn((emb_dim,), generator=g) / math.sqrt(max(emb_dim, 1))
    if dtype is not None:
        q = q.to(dtype=dtype)
    if device is not None:
        q = q.to(device=device)
    return q


def toy_queries(
    keys: Iterable[str],
    emb_dim: int,
    *,
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Stacked query vectors for a list of keys; shape=(K, emb_dim)."""
    qs: List[torch.Tensor] = [toy_query_vector(k, emb_dim, seed=seed, device=device, dtype=dtype) for k in keys]
    if not qs:
        return torch.empty((0, int(emb_dim)), device=device, dtype=dtype or torch.float32)
    return torch.stack(qs, dim=0)

