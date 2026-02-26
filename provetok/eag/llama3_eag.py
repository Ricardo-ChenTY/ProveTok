from __future__ import annotations

from dataclasses import dataclass

from ..pcg.llama2_pcg import (
    Llama2PCG,
    Llama2PCGConfig,
    _extract_inline_citations as extract_inline_citations,
    build_llama2_free_form_prompt as build_llama3_free_form_prompt,
    build_llama2_inline_citation_prompt as build_llama3_inline_citation_prompt,
    build_llama2_json_prompt as build_llama3_json_prompt,
    parse_llm_json,
    sanitize_generation_dict,
)


@dataclass
class Llama3EAGConfig(Llama2PCGConfig):
    """Alias config for Llama-3 EAG backend.

    The implementation intentionally reuses the existing generation core to
    avoid changing the current pipeline behavior.
    """


class Llama3EAG(Llama2PCG):
    """Llama-3 EAG adapter that reuses the proven LLM generation stack."""

    def __init__(self, cfg: Llama3EAGConfig):
        super().__init__(cfg)


__all__ = [
    "Llama3EAG",
    "Llama3EAGConfig",
    "build_llama3_json_prompt",
    "build_llama3_free_form_prompt",
    "build_llama3_inline_citation_prompt",
    "extract_inline_citations",
    "parse_llm_json",
    "sanitize_generation_dict",
]
