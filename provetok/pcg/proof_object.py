from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..types import Generation, Token, TokenRef
from .text_contract import enforce_pp_contract, parse_findings_text, render_findings, render_impression


def _token_to_record(tok: Token) -> Dict[str, Any]:
    return {
        "ref": str(getattr(tok, "ref", tok.cell_id)),
        "token_id": int(getattr(tok, "token_id", -1)),
        "cell_id": str(getattr(tok, "cell_id", "")),
        "level": int(getattr(tok, "level", 0)),
        "bounds_voxel": tuple(getattr(tok, "bounds_voxel", (0, 0, 0, 0, 0, 0))),
        "center_voxel": tuple(getattr(tok, "center_voxel", (0.0, 0.0, 0.0))),
        "bounds_mm": getattr(tok, "bounds_mm", None),
        "center_mm": getattr(tok, "center_mm", None),
        "score": float(getattr(tok, "score", 0.0)),
        "uncertainty": float(getattr(tok, "uncertainty", 0.0)),
    }


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


@dataclass(frozen=True)
class ProofObject:
    """pp.md-style proof-object attached to a report."""

    findings: List[str]
    impression: str
    citations_ref: Dict[int, List[TokenRef]]
    token_table: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": list(self.findings),
            "impression": str(self.impression),
            "citations_ref": {int(k): [str(x) for x in v] for k, v in (self.citations_ref or {}).items()},
            "token_table": list(self.token_table),
        }


def build_proof_object_from_generation(
    gen: Generation,
    tokens: Sequence[Token],
    *,
    k_max: int = 8,
    impression: Optional[str] = None,
) -> ProofObject:
    toks = list(tokens or [])
    gen_pp = enforce_pp_contract(gen, toks, k_max=int(k_max))
    findings_lines = render_findings(gen_pp, k_max=int(k_max))

    imp = impression
    if imp is None:
        imp = str(getattr(gen_pp, "impression", "") or "").strip()
    if not imp:
        imp = render_impression(findings_lines)

    citations_ref = {int(k): [str(x) for x in (v or [])] for k, v in (gen_pp.citations_ref or {}).items()}
    token_table = [_token_to_record(t) for t in toks]
    return ProofObject(findings=findings_lines, impression=str(imp), citations_ref=citations_ref, token_table=token_table)


def build_proof_object_from_findings_text(
    findings_text: str,
    tokens: Sequence[Token],
    *,
    k_max: int = 8,
    impression: str = "",
) -> ProofObject:
    toks = list(tokens or [])
    token_by_ref = {str(getattr(t, "ref", t.cell_id)): t for t in toks}

    frames, cites_ref = parse_findings_text(findings_text)

    citations: Dict[int, List[int]] = {}
    for k, refs in (cites_ref or {}).items():
        ids: List[int] = []
        for r in refs:
            # Prefer ref->token mapping; fall back to numeric token_id.
            t = token_by_ref.get(str(r))
            if t is not None:
                ids.append(int(getattr(t, "token_id", -1)))
                continue
            ri = _safe_int(r)
            if ri is not None:
                ids.append(int(ri))
        citations[int(k)] = ids

    gen = Generation(
        frames=list(frames),
        citations=citations,
        q={},
        refusal={},
        citations_ref=cites_ref,
        text="",
        impression=str(impression),
    )
    return build_proof_object_from_generation(gen, toks, k_max=int(k_max), impression=str(impression))

