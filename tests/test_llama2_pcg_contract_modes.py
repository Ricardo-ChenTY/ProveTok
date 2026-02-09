from __future__ import annotations

import torch

from provetok.pcg.llama2_pcg import Llama2PCGConfig, build_llama2_free_form_prompt, build_llama2_json_prompt
from provetok.types import Token


def _dummy_tokens(n: int = 4) -> list[Token]:
    toks: list[Token] = []
    for i in range(int(n)):
        toks.append(
            Token(
                token_id=int(i),
                cell_id=f"L1:{i%2},{(i//2)%2},0",
                level=1,
                embedding=torch.zeros(4),
                score=1.0 - 0.1 * float(i),
                uncertainty=0.1 * float(i),
            )
        )
    return toks


def test_build_llama2_json_prompt_respects_max_frames() -> None:
    cfg = Llama2PCGConfig(model_path="dummy", max_frames=3, contract_mode="full")
    prompt = build_llama2_json_prompt(_dummy_tokens(), cfg=cfg)
    assert "frames must contain at most 3 items" in prompt


def test_build_llama2_json_prompt_schema_only_requires_empty_citations() -> None:
    cfg = Llama2PCGConfig(model_path="dummy", max_frames=2, contract_mode="schema_only")
    prompt = build_llama2_json_prompt(_dummy_tokens(), cfg=cfg)
    assert "citations must be an empty JSON object" in prompt


def test_build_llama2_free_form_prompt_mentions_max_findings() -> None:
    cfg = Llama2PCGConfig(model_path="dummy", max_frames=5, contract_mode="free_form")
    prompt = build_llama2_free_form_prompt(_dummy_tokens(), cfg=cfg)
    assert "up to 5 findings" in prompt

