from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

from ..bet.allocator import pick_cell_to_split
from ..bet.tokenize import TokenEncoder
from ..grid.cells import Cell, cell_stable_id, root_cell, split
from ..pcg.text_contract import enforce_pp_contract, render_findings, render_impression
from ..eval.metrics_proof import attach_posthoc_citations
from ..types import Frame, Generation, Issue, Token
from ..verifier.pp_v1_1 import PPVerifierV11, create_pp_verifier


@dataclass(frozen=True)
class AgentConfig:
    """ProveTok-Agent (pp.md §4.5) configuration."""

    budget_tokens: int = 256
    emb_dim: int = 32
    init_level: int = 1
    max_depth: int = 3
    max_steps_per_finding: int = 8
    k_max_citations: int = 8
    l_min: int = 2

    # When NoCitation triggers, attach deterministic citations before attempting splits.
    use_posthoc_citations_for_r1: bool = True

    # When issues remain but we cannot split (budget/max_depth/no blamed token), rewrite by de-specifying.
    despecify_on_fail: bool = True
    despecify_clear_citations: bool = False


@dataclass(frozen=True)
class AgentStepTrace:
    finding_idx: int
    iter_idx: int
    num_tokens: int
    num_cells: int
    action: str  # "accept" | "split" | "despecify"
    issues: List[Dict]
    split_cell_id: str = ""
    stopped_reason: str = ""


@dataclass
class AgentResult:
    tokens: List[Token]
    generation: Generation
    issues: List[Issue]
    findings_lines: List[str]
    impression: str
    trace: List[AgentStepTrace] = field(default_factory=list)
    final_cells: List[Cell] = field(default_factory=list)


def _slice_generation(gen: Generation, *, frame_idx: int) -> Generation:
    """Extract a single-frame Generation for verifier-local reasoning."""
    if gen is None or not (gen.frames or []):
        return Generation(frames=[], citations={}, q={}, refusal={}, text="")
    idx = int(frame_idx)
    if idx < 0 or idx >= len(gen.frames):
        return Generation(frames=[], citations={}, q={}, refusal={}, text="")
    fr = gen.frames[idx]
    cites = {0: list((gen.citations or {}).get(idx, []) or [])}
    q = {0: float((gen.q or {}).get(idx, getattr(fr, "confidence", 0.5)))}
    refusal = {0: bool((gen.refusal or {}).get(idx, False))}
    cites_ref = {0: list((gen.citations_ref or {}).get(idx, []) or [])} if getattr(gen, "citations_ref", None) else {}
    return Generation(frames=[fr], citations=cites, q=q, refusal=refusal, citations_ref=cites_ref, text="")


def _issue_dict(iss: Issue) -> Dict:
    return {
        "frame_idx": int(getattr(iss, "frame_idx", -1)),
        "issue_type": str(getattr(iss, "issue_type", "")),
        "severity": int(getattr(iss, "severity", 0)),
        "rule_id": str(getattr(iss, "rule_id", "")),
        "message": str(getattr(iss, "message", "")),
        "evidence_trace": getattr(iss, "evidence_trace", {}) or {},
    }


def _collect_blame_refs(issues: Sequence[Issue]) -> List[str]:
    refs: List[str] = []
    for iss in issues:
        trace = getattr(iss, "evidence_trace", {}) or {}
        for r in (trace.get("blame_refs", []) or []):
            if isinstance(r, str) and r:
                refs.append(str(r))
    # Deterministic unique order
    seen: Set[str] = set()
    out: List[str] = []
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _expand_ids_to_current_leaves(
    ids: Sequence[int],
    *,
    leaf_ids: Set[int],
    split_children: Dict[int, List[int]],
    k_max: int,
) -> List[int]:
    """Expand citations to leaf token ids after splits (keeps proof-object valid)."""
    k_max = max(0, int(k_max))

    def expand_one(tid: int) -> List[int]:
        t = int(tid)
        if t in leaf_ids:
            return [t]
        ch = split_children.get(t)
        if not ch:
            return []
        out: List[int] = []
        for c in ch:
            out.extend(expand_one(int(c)))
        return out

    expanded: List[int] = []
    for tid in ids:
        expanded.extend(expand_one(int(tid)))

    # Deduplicate in-order and truncate.
    seen: Set[int] = set()
    out2: List[int] = []
    for tid in expanded:
        if tid in seen:
            continue
        seen.add(tid)
        out2.append(int(tid))
        if k_max and len(out2) >= k_max:
            break
    return out2


def _despecify_frame(frame: Frame) -> Frame:
    """Deterministic conservative rewrite used when budget is exhausted."""
    return Frame(
        finding=str(getattr(frame, "finding", "normal")),
        polarity=str(getattr(frame, "polarity", "present")),
        laterality="unspecified",
        confidence=min(0.6, float(getattr(frame, "confidence", 0.5))),
        location=str(getattr(frame, "location", "unspecified")),
        size_bin=str(getattr(frame, "size_bin", "unspecified")),
        severity=str(getattr(frame, "severity", "unspecified")),
        uncertain=True,
    )


def run_provetok_agent(
    volume: torch.Tensor,
    *,
    generator_fn: Callable[[List[Token]], Generation],
    verifier: Optional[PPVerifierV11] = None,
    cfg: AgentConfig = AgentConfig(),
    seed: int = 0,
    encoder=None,
    affine_zyx: Optional[Any] = None,
    split_cell_fn: Optional[Callable[[List[Cell], List[Token], List[Issue]], Optional[Cell]]] = None,
) -> AgentResult:
    """Run pp.md-style ProveTok-Agent write→verify→split→rewrite loop (v1.1 scope).

    Notes:
    - Works with any generator that outputs a fixed ordered list of frames, where
      each frame index is treated as one "finding line" in the agent loop.
    - Uses the pp.md hard verifier by default (R1–R4).
    """
    budget_tokens = int(cfg.budget_tokens)
    init_level = int(cfg.init_level)
    max_depth = int(cfg.max_depth)
    max_steps_per_finding = max(1, int(cfg.max_steps_per_finding))

    if init_level < 0:
        raise ValueError("--init_level must be >= 0")
    if max_depth < 0:
        raise ValueError("--max_depth must be >= 0")
    if init_level > max_depth:
        init_level = max_depth

    if verifier is None:
        verifier = create_pp_verifier(l_min=int(cfg.l_min))

    # Initial leaf cells: full coverage at init_level (pp.md default is 1 → 8 tokens).
    cells: List[Cell] = [root_cell()]
    if init_level > 0:
        # If budget is below the init coverage size, reduce init_level to fit.
        while init_level > 0:
            n = 2 ** int(init_level)
            if int(n * n * n) <= int(budget_tokens):
                break
            init_level -= 1
        n = 2 ** int(init_level)
        cells = [Cell(level=init_level, ix=ix, iy=iy, iz=iz) for ix in range(n) for iy in range(n) for iz in range(n)]

    token_encoder = TokenEncoder(volume=volume, emb_dim=int(cfg.emb_dim), seed=int(seed), encoder=encoder, affine_zyx=affine_zyx)
    trace: List[AgentStepTrace] = []

    # Split history for keeping earlier citations valid after later splits.
    split_children: Dict[int, List[int]] = {}

    # Determine number of finding slots from the generator output.
    toks0 = token_encoder.encode(cells)
    gen0 = generator_fn(toks0)
    K = len(gen0.frames or [])

    accepted_frames: List[Frame] = []
    accepted_citations: Dict[int, List[int]] = {}
    accepted_q: Dict[int, float] = {}
    accepted_refusal: Dict[int, bool] = {}

    for finding_idx in range(K):
        stopped_reason = "max_steps"
        done_finding = False
        for it in range(max_steps_per_finding):
            tokens = token_encoder.encode(cells)
            leaf_ids = {int(t.token_id) for t in tokens}

            gen_full = generator_fn(tokens)
            gen_one = _slice_generation(gen_full, frame_idx=int(finding_idx))

            # Enforce the pp.md contract (citations validity + k_max + citations_ref).
            gen_one = enforce_pp_contract(gen_one, tokens, k_max=int(cfg.k_max_citations))

            # R1 fallback: attach deterministic citations if missing.
            issues = verifier.verify(gen_one, tokens)
            if bool(cfg.use_posthoc_citations_for_r1):
                has_r1 = any(str(getattr(iss, "rule_id", "")) == "R1" for iss in issues)
                if has_r1:
                    gen_one = attach_posthoc_citations(
                        gen_one,
                        tokens,
                        k_max=int(cfg.k_max_citations),
                        positive_only=True,
                        overwrite_empty_only=True,
                    )
                    gen_one = enforce_pp_contract(gen_one, tokens, k_max=int(cfg.k_max_citations))
                    issues = verifier.verify(gen_one, tokens)

            # Accept if clean.
            if not issues:
                fr = gen_one.frames[0] if gen_one.frames else Frame(finding="normal", polarity="absent", laterality="unspecified", confidence=0.5)
                accepted_frames.append(fr)
                accepted_citations[finding_idx] = list((gen_one.citations or {}).get(0, []) or [])
                accepted_q[finding_idx] = float((gen_one.q or {}).get(0, getattr(fr, "confidence", 0.5)))
                accepted_refusal[finding_idx] = bool((gen_one.refusal or {}).get(0, False))
                trace.append(
                    AgentStepTrace(
                        finding_idx=int(finding_idx),
                        iter_idx=int(it),
                        num_tokens=int(len(tokens)),
                        num_cells=int(len(cells)),
                        action="accept",
                        issues=[],
                        stopped_reason="verified_clean",
                    )
                )
                stopped_reason = "verified_clean"
                done_finding = True
                break

            # Determine whether we can split a blamed token.
            blame_refs = _collect_blame_refs(issues)
            splittable_cells = [c for c in cells if int(c.level) < int(max_depth)]
            can_afford_split = (len(cells) + 7) <= int(budget_tokens)

            c_star: Optional[Cell] = None
            if blame_refs and splittable_cells and can_afford_split:
                if split_cell_fn is not None:
                    budget_frac = float(len(cells)) / float(max(1, budget_tokens))
                    step_frac = float(it) / float(max(1, max_steps_per_finding))
                    try:
                        c_star = split_cell_fn(
                            list(splittable_cells),
                            list(tokens),
                            list(issues),
                            budget_frac=float(budget_frac),
                            step_frac=float(step_frac),
                            finding_idx=int(finding_idx),
                            iter_idx=int(it),
                        )
                    except TypeError:
                        c_star = split_cell_fn(list(splittable_cells), list(tokens), list(issues))
                else:
                    c_star = pick_cell_to_split(splittable_cells, tokens, list(issues))

            if c_star is None:
                # No blamed splittable action or cannot afford split → despecify.
                fr0 = gen_one.frames[0] if gen_one.frames else Frame(finding="normal", polarity="absent", laterality="unspecified", confidence=0.5)
                fr1 = _despecify_frame(fr0) if bool(cfg.despecify_on_fail) else fr0
                cites = [] if bool(cfg.despecify_clear_citations) else list((gen_one.citations or {}).get(0, []) or [])

                # Expand citations to current leaves in case the candidate cited stale ids.
                cites = _expand_ids_to_current_leaves(
                    cites,
                    leaf_ids=leaf_ids,
                    split_children=split_children,
                    k_max=int(cfg.k_max_citations),
                )

                accepted_frames.append(fr1)
                accepted_citations[finding_idx] = list(cites)
                accepted_q[finding_idx] = float((gen_one.q or {}).get(0, getattr(fr1, "confidence", 0.5)))
                accepted_refusal[finding_idx] = bool((gen_one.refusal or {}).get(0, False))

                trace.append(
                    AgentStepTrace(
                        finding_idx=int(finding_idx),
                        iter_idx=int(it),
                        num_tokens=int(len(tokens)),
                        num_cells=int(len(cells)),
                        action="despecify",
                        issues=[_issue_dict(x) for x in issues],
                        stopped_reason=(
                            "budget_exhausted" if not can_afford_split else ("no_blame_refs" if not blame_refs else "no_splittable_blame")
                        ),
                    )
                )
                stopped_reason = "despecified"
                done_finding = True
                break

            # Split action.
            parent_id = int(cell_stable_id(c_star))
            children = split(c_star)
            child_ids = [int(cell_stable_id(ch)) for ch in children]
            split_children[parent_id] = list(child_ids)

            # Update previously accepted citations to stay valid under the new leaf set.
            cells = [c for c in cells if c.id() != c_star.id()] + children
            tokens_after = token_encoder.encode(cells)
            leaf_ids_after = {int(t.token_id) for t in tokens_after}
            for k in list(accepted_citations.keys()):
                accepted_citations[k] = _expand_ids_to_current_leaves(
                    accepted_citations[k],
                    leaf_ids=leaf_ids_after,
                    split_children=split_children,
                    k_max=int(cfg.k_max_citations),
                )

            trace.append(
                AgentStepTrace(
                    finding_idx=int(finding_idx),
                    iter_idx=int(it),
                    num_tokens=int(len(tokens)),
                    num_cells=int(len(cells)),
                    action="split",
                    issues=[_issue_dict(x) for x in issues],
                    split_cell_id=str(c_star.id()),
                    stopped_reason="split_and_retry",
                )
            )
            stopped_reason = "split"

        # If we exhausted the step budget without accepting/despecifying (e.g., kept splitting),
        # fall back to a conservative de-specification to ensure we always emit exactly one frame
        # per slot and keep frame<->citation indices aligned.
        if not done_finding:
            tokens = token_encoder.encode(cells)
            leaf_ids = {int(t.token_id) for t in tokens}
            gen_full = generator_fn(tokens)
            gen_one = _slice_generation(gen_full, frame_idx=int(finding_idx))
            gen_one = enforce_pp_contract(gen_one, tokens, k_max=int(cfg.k_max_citations))

            issues = verifier.verify(gen_one, tokens)
            if bool(cfg.use_posthoc_citations_for_r1):
                has_r1 = any(str(getattr(iss, "rule_id", "")) == "R1" for iss in issues)
                if has_r1:
                    gen_one = attach_posthoc_citations(
                        gen_one,
                        tokens,
                        k_max=int(cfg.k_max_citations),
                        positive_only=True,
                        overwrite_empty_only=True,
                    )
                    gen_one = enforce_pp_contract(gen_one, tokens, k_max=int(cfg.k_max_citations))
                    issues = verifier.verify(gen_one, tokens)

            fr0 = gen_one.frames[0] if gen_one.frames else Frame(finding="normal", polarity="absent", laterality="unspecified", confidence=0.5)
            fr1 = _despecify_frame(fr0) if bool(cfg.despecify_on_fail) else fr0
            cites = [] if bool(cfg.despecify_clear_citations) else list((gen_one.citations or {}).get(0, []) or [])
            cites = _expand_ids_to_current_leaves(
                cites,
                leaf_ids=leaf_ids,
                split_children=split_children,
                k_max=int(cfg.k_max_citations),
            )

            accepted_frames.append(fr1)
            accepted_citations[finding_idx] = list(cites)
            accepted_q[finding_idx] = float((gen_one.q or {}).get(0, getattr(fr1, "confidence", 0.5)))
            accepted_refusal[finding_idx] = bool((gen_one.refusal or {}).get(0, False))
            trace.append(
                AgentStepTrace(
                    finding_idx=int(finding_idx),
                    iter_idx=int(max_steps_per_finding),
                    num_tokens=int(len(tokens)),
                    num_cells=int(len(cells)),
                    action="despecify",
                    issues=[_issue_dict(x) for x in issues],
                    stopped_reason="max_steps_despecify",
                )
            )

    # Build final generation and ensure contract refs are populated.
    final_tokens = token_encoder.encode(cells)
    final_gen = Generation(
        frames=list(accepted_frames),
        citations={int(k): list(v) for k, v in accepted_citations.items()},
        q=dict(accepted_q),
        refusal=dict(accepted_refusal),
        text="",
    )
    final_gen = enforce_pp_contract(final_gen, final_tokens, k_max=int(cfg.k_max_citations))

    final_issues = verifier.verify(final_gen, final_tokens)
    findings_lines = render_findings(final_gen, k_max=int(cfg.k_max_citations))
    impression = render_impression(findings_lines)

    return AgentResult(
        tokens=final_tokens,
        generation=final_gen,
        issues=list(final_issues),
        findings_lines=findings_lines,
        impression=impression,
        trace=trace,
        final_cells=list(cells),
    )
