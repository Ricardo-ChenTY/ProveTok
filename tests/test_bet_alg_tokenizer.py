from __future__ import annotations

import torch

from provetok.baselines.tokenizers import BETAlgTokenizer


def test_bet_alg_tokenizer_deterministic_and_budgeted() -> None:
    torch.manual_seed(0)
    vol = torch.randn(16, 16, 16)

    tok = BETAlgTokenizer(max_depth=4, min_cover_tokens=8)
    out1 = tok.tokenize(vol, budget_tokens=32, emb_dim=16, seed=123)
    out2 = tok.tokenize(vol, budget_tokens=32, emb_dim=16, seed=123)

    assert len(out1) == 32
    assert len(out2) == 32
    # Token ids are stable octree-path ids (not sequential indices).
    ids1 = [int(t.token_id) for t in out1]
    ids2 = [int(t.token_id) for t in out2]
    assert ids1 == ids2
    assert len(set(ids1)) == len(ids1)
    assert [t.cell_id for t in out1] == [t.cell_id for t in out2]


def test_bet_alg_tokenizer_handles_overshoot_and_truncation() -> None:
    torch.manual_seed(1)
    vol = torch.randn(16, 16, 16)

    tok = BETAlgTokenizer(max_depth=6, min_cover_tokens=64)
    out = tok.tokenize(vol, budget_tokens=10, emb_dim=8, seed=0)
    assert len(out) == 10
    ids = [int(t.token_id) for t in out]
    assert len(set(ids)) == len(ids)
