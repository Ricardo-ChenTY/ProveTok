from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..types import Frame, Generation, Issue, Token
from ..verifier.pp_v1_1 import PPVerifierV11, create_pp_verifier
from .text_contract import enforce_pp_contract, render_findings


@dataclass(frozen=True)
class FindingLine:
    finding_idx: int
    frame: Frame
    citations: List[int]
    citations_ref: List[str]
    q: float
    refusal: bool
    text: str

    def to_single_generation(self) -> Generation:
        return Generation(
            frames=[self.frame],
            citations={0: list(self.citations)},
            q={0: float(self.q)},
            refusal={0: bool(self.refusal)},
            citations_ref={0: [str(x) for x in (self.citations_ref or [])]},
            text="",
        )


class FindingGenerator:
    """Finding-by-finding generator interface for ProveTok-Agent (pp.md §4.5)."""

    def generate_one(
        self,
        tokens: Sequence[Token],
        *,
        finding_idx: int,
        context_findings: Sequence[str] = (),
    ) -> FindingLine:
        raise NotImplementedError

    def rewrite_one(
        self,
        tokens: Sequence[Token],
        *,
        finding_idx: int,
        context_findings: Sequence[str] = (),
        issues: Sequence[Issue] = (),
    ) -> FindingLine:
        raise NotImplementedError


def _slice_generation(gen: Generation, *, frame_idx: int) -> Generation:
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


class SlicedGenerationFindingGenerator(FindingGenerator):
    """Adapter: uses an existing `generator_fn(tokens)->Generation` and slices one frame.

    This is useful for incremental migration: it provides a finding-by-finding
    interface without forcing all PCGs to implement it natively.
    """

    def __init__(
        self,
        generator_fn: Callable[[List[Token]], Generation],
        *,
        k_max: int = 8,
        l_min: int = 2,
        verifier: Optional[PPVerifierV11] = None,
    ):
        self.generator_fn = generator_fn
        self.k_max = int(k_max)
        self.l_min = int(l_min)
        self.verifier = verifier or create_pp_verifier(l_min=int(l_min))

    def _make_line(self, gen_one: Generation, tokens: Sequence[Token], *, finding_idx: int) -> FindingLine:
        gen_one = enforce_pp_contract(gen_one, tokens, k_max=int(self.k_max))
        fr = gen_one.frames[0] if gen_one.frames else Frame(finding="normal", polarity="absent", laterality="unspecified", confidence=0.5)
        cites = list((gen_one.citations or {}).get(0, []) or [])
        cites_ref = [str(x) for x in (gen_one.citations_ref or {}).get(0, [])] if getattr(gen_one, "citations_ref", None) else [str(int(x)) for x in cites]
        q = float((gen_one.q or {}).get(0, getattr(fr, "confidence", 0.5)))
        refusal = bool((gen_one.refusal or {}).get(0, False))
        text = render_findings(gen_one, k_max=int(self.k_max))[0] if gen_one.frames else ""
        return FindingLine(
            finding_idx=int(finding_idx),
            frame=fr,
            citations=[int(x) for x in cites],
            citations_ref=[str(x) for x in cites_ref],
            q=float(q),
            refusal=bool(refusal),
            text=str(text),
        )

    def generate_one(
        self,
        tokens: Sequence[Token],
        *,
        finding_idx: int,
        context_findings: Sequence[str] = (),
    ) -> FindingLine:
        _ = context_findings  # reserved for future LLM prompts
        gen_full = self.generator_fn(list(tokens or []))
        gen_one = _slice_generation(gen_full, frame_idx=int(finding_idx))
        return self._make_line(gen_one, tokens, finding_idx=int(finding_idx))

    def rewrite_one(
        self,
        tokens: Sequence[Token],
        *,
        finding_idx: int,
        context_findings: Sequence[str] = (),
        issues: Sequence[Issue] = (),
    ) -> FindingLine:
        # Default rewrite: regenerate then apply a deterministic pp.md-aligned citation repair.
        base = self.generate_one(tokens, finding_idx=int(finding_idx), context_findings=context_findings)
        repaired = repair_finding_line_pp(
            base,
            tokens,
            issues=list(issues or []),
            k_max=int(self.k_max),
            l_min=int(self.l_min),
        )
        return repaired


def _infer_volume_width_from_tokens(tokens: Sequence[Token]) -> int:
    xs: List[int] = []
    for t in tokens or []:
        b = getattr(t, "bounds_voxel", None)
        if not isinstance(b, tuple) or len(b) != 6:
            continue
        try:
            x1 = int(b[5])
        except Exception:
            continue
        if x1 > 0:
            xs.append(x1)
    return int(max(xs)) if xs else 0


def _token_side(tok: Token, *, x_mid: float) -> str:
    b = getattr(tok, "bounds_voxel", None)
    if not isinstance(b, tuple) or len(b) != 6:
        return "unknown"
    try:
        x0, x1 = float(b[4]), float(b[5])
    except Exception:
        return "unknown"
    if x1 <= x0:
        return "unknown"
    if x1 <= float(x_mid):
        return "left"
    if x0 >= float(x_mid):
        return "right"
    return "cross"


def _is_positive_frame(frame: Frame) -> bool:
    if bool(getattr(frame, "uncertain", False)):
        return False
    pol = str(getattr(frame, "polarity", "")).lower()
    if pol not in ("present", "positive"):
        return False
    finding = str(getattr(frame, "finding", "")).strip().lower()
    if finding in ("", "normal"):
        return False
    return True


def repair_finding_line_pp(
    line: FindingLine,
    tokens: Sequence[Token],
    *,
    issues: Sequence[Issue],
    k_max: int,
    l_min: int,
) -> FindingLine:
    """Deterministic pp.md-aligned rewrite that repairs citations where possible.

    This does not attempt to rewrite the medical content; it repairs the proof-carrying
    citation object to satisfy R1–R4 when enough evidence is available.
    """
    toks = list(tokens or [])
    token_by_id: Dict[int, Token] = {int(getattr(t, "token_id", -1)): t for t in toks}
    ranked = sorted(toks, key=lambda t: (-float(getattr(t, "score", 0.0)), int(getattr(t, "token_id", 0))))

    W = _infer_volume_width_from_tokens(toks)
    x_mid = 0.5 * float(W) if W > 0 else 0.0

    frame = line.frame
    positive = _is_positive_frame(frame)
    want_lat = str(getattr(frame, "laterality", "")).lower()

    rule_ids = {str(getattr(i, "rule_id", "")) for i in (issues or [])}
    need_r1 = positive and ("R1" in rule_ids or not (line.citations or []))
    need_r2 = positive and ("R2" in rule_ids)
    need_r3 = positive and ("R3" in rule_ids) and want_lat in ("left", "right")
    need_r4 = positive and ("R4" in rule_ids) and want_lat == "bilateral"

    # Candidate pool subject to laterality constraints.
    def ok_lat(tok: Token) -> bool:
        if want_lat not in ("left", "right", "bilateral"):
            return True
        side = _token_side(tok, x_mid=float(x_mid))
        if want_lat in ("left", "right"):
            return side == want_lat
        # bilateral: allow any; we'll enforce both-sides below.
        return side in ("left", "right", "cross", "unknown")

    pool = [t for t in ranked if ok_lat(t)]

    # Start from existing citations if they still exist.
    cites: List[int] = []
    for tid in (line.citations or []):
        if int(tid) in token_by_id:
            cites.append(int(tid))
    # Dedup
    seen = set()
    cites = [x for x in cites if not (x in seen or seen.add(x))]

    # R1: ensure non-empty citations for positive claims.
    if need_r1 and not cites:
        cites = [int(t.token_id) for t in pool[: max(0, int(k_max))]]

    # R3: for left/right, all cited tokens must be on the claimed side.
    if need_r3 and want_lat in ("left", "right"):
        good = []
        for tid in cites:
            tok = token_by_id.get(int(tid))
            if tok is None:
                continue
            if _token_side(tok, x_mid=float(x_mid)) == want_lat:
                good.append(int(tid))
        if not good:
            good = [int(t.token_id) for t in pool[: max(0, int(k_max))]]
        cites = good

    # R4: ensure bilateral has >=1 left + >=1 right citation if possible.
    if need_r4 and want_lat == "bilateral":
        left = [t for t in pool if _token_side(t, x_mid=float(x_mid)) == "left"]
        right = [t for t in pool if _token_side(t, x_mid=float(x_mid)) == "right"]
        # Preserve existing if it already satisfies.
        has_left = any(_token_side(token_by_id[tid], x_mid=float(x_mid)) == "left" for tid in cites if tid in token_by_id)
        has_right = any(_token_side(token_by_id[tid], x_mid=float(x_mid)) == "right" for tid in cites if tid in token_by_id)
        if not (has_left and has_right):
            cites = []
            if left:
                cites.append(int(left[0].token_id))
            if right:
                cites.append(int(right[0].token_id))
            # Fill remaining with pool.
            for t in pool:
                if len(cites) >= int(k_max):
                    break
                tid = int(t.token_id)
                if tid not in cites:
                    cites.append(tid)

    # R2: ensure at least one cited token has level >= l_min.
    if need_r2 and cites:
        levels = [int(getattr(token_by_id.get(tid), "level", 0)) for tid in cites if tid in token_by_id]
        if not levels or int(max(levels)) < int(l_min):
            fine = [t for t in pool if int(getattr(t, "level", 0)) >= int(l_min)]
            if fine:
                # Replace the last citation (or append) with the best fine token.
                fine_id = int(fine[0].token_id)
                if fine_id not in cites:
                    if len(cites) >= 1:
                        cites[-1] = fine_id
                    else:
                        cites.append(fine_id)

    cites = cites[: max(0, int(k_max))]
    cites_ref = [str(getattr(token_by_id.get(int(tid)), "ref", tid)) for tid in cites if int(tid) in token_by_id]

    gen_one = Generation(
        frames=[frame],
        citations={0: list(cites)},
        q={0: float(line.q)},
        refusal={0: bool(line.refusal)},
        citations_ref={0: list(cites_ref)},
        text="",
    )
    gen_one = enforce_pp_contract(gen_one, toks, k_max=int(k_max))
    text = render_findings(gen_one, k_max=int(k_max))[0] if gen_one.frames else ""

    fr0 = gen_one.frames[0] if gen_one.frames else frame
    cites0 = list((gen_one.citations or {}).get(0, []) or [])
    cites_ref0 = [str(x) for x in (gen_one.citations_ref or {}).get(0, [])] if getattr(gen_one, "citations_ref", None) else [str(int(x)) for x in cites0]

    return FindingLine(
        finding_idx=int(line.finding_idx),
        frame=fr0,
        citations=[int(x) for x in cites0],
        citations_ref=[str(x) for x in cites_ref0],
        q=float((gen_one.q or {}).get(0, float(line.q))),
        refusal=bool((gen_one.refusal or {}).get(0, bool(line.refusal))),
        text=str(text),
    )
