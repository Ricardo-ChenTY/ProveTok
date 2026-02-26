from __future__ import annotations

import torch

from provetok.pcg.llama2_pcg import (
    Llama2PCGConfig,
    _extract_inline_citations,
    build_llama2_free_form_prompt,
    build_llama2_inline_citation_prompt,
    build_llama2_json_prompt,
)
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


def test_build_llama2_inline_prompt_contains_citation_tags() -> None:
    cfg = Llama2PCGConfig(model_path="dummy", max_frames=2, contract_mode="inline_citation")
    prompt, tag_map = build_llama2_inline_citation_prompt(_dummy_tokens(3), cfg=cfg)
    assert "inline citation tags" in prompt.lower()
    assert "CIT_001" in prompt
    assert len(tag_map) == 3


def test_extract_inline_citations_aligns_sentence_order() -> None:
    cfg = Llama2PCGConfig(model_path="dummy", max_frames=2, topk_citations=2, contract_mode="inline_citation")
    text = (
        "Right nodule [CIT_001, CIT_002] suspicious. "
        "Left effusion [CIT_003]."
    )
    frames, citations, cleaned = _extract_inline_citations(
        text,
        tag_to_token_id={"CIT_001": 11, "CIT_002": 22, "CIT_003": 33},
        topk_citations=int(cfg.topk_citations),
        max_frames=int(cfg.max_frames),
    )
    assert len(frames) >= 1
    assert citations.get(0, []) == [11, 22]
    assert "CIT_001" not in cleaned


def test_extract_inline_citations_handles_fullwidth_punct_and_no_underscore_tags() -> None:
    text = "Right nodule【CIT001，CIT_002】 suspicious；Left opacity [cit003]。"
    frames, citations, cleaned = _extract_inline_citations(
        text,
        tag_to_token_id={"CIT_001": 7, "CIT_002": 8, "CIT_003": 9},
        topk_citations=3,
        max_frames=2,
    )
    assert len(frames) >= 1
    assert citations.get(0, []) == [7, 8]
    assert "CIT001" not in cleaned.upper()


def test_extract_inline_citations_ignores_unknown_tags_and_dedups() -> None:
    text = "Nodule [CIT_001, CIT_999, CIT_001, CIT_002]."
    frames, citations, _ = _extract_inline_citations(
        text,
        tag_to_token_id={"CIT_001": 11, "CIT_002": 22},
        topk_citations=5,
        max_frames=1,
    )
    assert len(frames) >= 1
    assert citations.get(0, []) == [11, 22]

