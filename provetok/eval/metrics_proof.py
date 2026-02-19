from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..types import Frame, Generation, Token
from ..verifier.pp_v1_1 import PPVerifierV11, create_pp_verifier


@dataclass(frozen=True)
class ProofWeights:
    """pp.md default weights for WeightedIssue."""

    w1_r1: float = 3.0
    w2_r2: float = 2.0
    w3_r3: float = 2.0
    w4_r4: float = 2.0


def _is_positive_frame(gen: Generation, frame_idx: int, frame: Frame) -> bool:
    if bool((gen.refusal or {}).get(int(frame_idx), False)):
        return False
    if bool(getattr(frame, "uncertain", False)):
        # pp.md rewrite path can de-specify into uncertainty to avoid hard-rule triggers.
        return False
    pol = str(getattr(frame, "polarity", "")).lower()
    if pol not in ("present", "positive"):
        return False
    finding = str(getattr(frame, "finding", "")).strip().lower()
    if finding in ("", "normal"):
        return False
    return True


def attach_posthoc_citations(
    gen: Generation,
    tokens: Sequence[Token],
    *,
    k_max: int = 8,
    positive_only: bool = True,
    overwrite_empty_only: bool = True,
) -> Generation:
    """Attach deterministic citations for frames that do not have any.

    This implements pp.md §6.5 "post-hoc citation wrapper" in a minimal form:
    select top-K tokens by score (tie-break by token_id) and attach them.
    """
    if gen is None:
        raise ValueError("gen is required")
    toks = list(tokens or [])
    if not toks:
        return gen

    ranked = sorted(toks, key=lambda t: (-float(getattr(t, "score", 0.0)), int(getattr(t, "token_id", 0))))
    top = [int(t.token_id) for t in ranked[: max(0, min(int(k_max), len(ranked)))]]
    top_refs = [str(getattr(t, "ref", t.cell_id)) for t in ranked[: len(top)]]

    citations: Dict[int, List[int]] = {int(k): list(v or []) for k, v in (gen.citations or {}).items()}
    citations_ref: Dict[int, List[str]] = {int(k): [str(x) for x in (v or [])] for k, v in (gen.citations_ref or {}).items()}

    for idx, frame in enumerate(gen.frames or []):
        idx_i = int(idx)
        if positive_only and (not _is_positive_frame(gen, idx_i, frame)):
            continue
        existing = citations.get(idx_i, [])
        if overwrite_empty_only and existing:
            continue
        citations[idx_i] = list(top)
        citations_ref[idx_i] = list(top_refs)

    return Generation(
        frames=list(gen.frames),
        citations=citations,
        q=dict(gen.q or {}),
        refusal=dict(gen.refusal or {}),
        citations_ref=citations_ref,
        text=str(getattr(gen, "text", "") or ""),
        impression=str(getattr(gen, "impression", "") or ""),
        report_text=str(getattr(gen, "report_text", "") or ""),
    )


def compute_proof_metrics(
    gen: Generation,
    tokens: Sequence[Token],
    *,
    verifier: Optional[PPVerifierV11] = None,
    l_min: int = 2,
    weights: ProofWeights = ProofWeights(),
) -> Dict[str, float]:
    """Compute pp.md proof metrics (R1–R4 + WeightedIssue) for a single sample."""
    if verifier is None:
        verifier = create_pp_verifier(l_min=int(l_min))

    toks = list(tokens or [])
    issues = verifier.verify(gen, toks)

    # Denominators (rule applicability).
    n_pos = 0
    n_lat = 0
    n_bilat = 0
    for idx, frame in enumerate(gen.frames or []):
        if not _is_positive_frame(gen, idx, frame):
            continue
        n_pos += 1
        lat = str(getattr(frame, "laterality", "")).lower()
        if lat in ("left", "right"):
            n_lat += 1
        elif lat == "bilateral":
            n_bilat += 1

    violated_by_frame: Dict[str, set] = {"R1": set(), "R2": set(), "R3": set(), "R4": set()}
    for iss in issues:
        rid = str(getattr(iss, "rule_id", ""))
        if rid in violated_by_frame:
            violated_by_frame[rid].add(int(getattr(iss, "frame_idx", -1)))

    c1 = len(violated_by_frame["R1"])
    c2 = len(violated_by_frame["R2"])
    c3 = len(violated_by_frame["R3"])
    c4 = len(violated_by_frame["R4"])

    r1 = float(c1 / n_pos) if n_pos > 0 else 0.0
    r2 = float(c2 / n_pos) if n_pos > 0 else 0.0
    r3 = float(c3 / n_lat) if n_lat > 0 else 0.0
    r4 = float(c4 / n_bilat) if n_bilat > 0 else 0.0

    weighted = (
        float(weights.w1_r1) * r1
        + float(weights.w2_r2) * r2
        + float(weights.w3_r3) * r3
        + float(weights.w4_r4) * r4
    )

    return {
        "r1_no_citation": float(r1),
        "r2_coarse_only": float(r2),
        "r3_laterality_mismatch": float(r3),
        "r4_bilateral_separation": float(r4),
        "weighted_issue": float(weighted),
        # Diagnostics (counts/denoms): keep as floats for consistency with other metrics.
        "n_pos_frames": float(n_pos),
        "n_lat_frames": float(n_lat),
        "n_bilat_frames": float(n_bilat),
        "n_r1": float(c1),
        "n_r2": float(c2),
        "n_r3": float(c3),
        "n_r4": float(c4),
    }

