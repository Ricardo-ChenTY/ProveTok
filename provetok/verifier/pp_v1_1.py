from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from ..types import Frame, Generation, Issue, Token, TokenRef


PP_RULE_SET_VERSION = "pp_v1.1_ruleset_v1"

TokenSide = Literal["left", "right", "cross", "unknown"]


def _infer_volume_width(tokens: List[Token]) -> int:
    xs: List[int] = []
    for t in tokens:
        b = getattr(t, "bounds_voxel", (0, 0, 0, 0, 0, 0))
        if not isinstance(b, tuple) or len(b) != 6:
            continue
        x1 = int(b[5])
        if x1 > 0:
            xs.append(x1)
    return int(max(xs)) if xs else 0


def _token_side(tok: Token, *, x_mid: float) -> TokenSide:
    b = getattr(tok, "bounds_voxel", (0, 0, 0, 0, 0, 0))
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


def _frame_is_positive(frame: Frame) -> bool:
    if bool(getattr(frame, "uncertain", False)):
        return False
    return str(frame.polarity) in ("present", "positive") and str(frame.finding).lower() not in ("normal", "")


def _frame_citations_ref(gen: Generation, *, frame_idx: int, token_by_id: Dict[int, Token]) -> List[TokenRef]:
    if gen.citations_ref and frame_idx in (gen.citations_ref or {}):
        return [str(x) for x in (gen.citations_ref or {}).get(int(frame_idx), [])]
    # Fallback: convert token_id citations to stable refs when possible.
    refs: List[str] = []
    for tid in (gen.citations or {}).get(int(frame_idx), []):
        try:
            tid_i = int(tid)
        except Exception:
            continue
        t = token_by_id.get(tid_i)
        if t is None:
            continue
        refs.append(str(getattr(t, "ref", t.cell_id)))
    return refs


def _issue(
    *,
    frame_idx: int,
    issue_type: str,
    severity: int,
    rule_id: str,
    message: str,
    cited_refs: List[str],
    blame_refs: List[str],
    rule_inputs: Optional[Dict] = None,
    rule_outputs: Optional[Dict] = None,
) -> Issue:
    return Issue(
        frame_idx=int(frame_idx),
        issue_type=issue_type,  # type: ignore[arg-type]
        severity=int(severity),
        rule_id=str(rule_id),
        message=str(message),
        evidence_trace={
            "cited_refs": list(cited_refs),
            "blame_refs": list(blame_refs),
            "rule_inputs": dict(rule_inputs or {}),
            "rule_outputs": dict(rule_outputs or {}),
        },
    )


@dataclass(frozen=True)
class PPVerifierConfig:
    l_min: int = 2


class PPVerifierV11:
    """pp.md v1.1 hard verifier (R1–R4).

    This verifier is intentionally deterministic and auditable:
    - depends only on citations + token geometry + frame slots
    - does not use learned judges or token scores by default
    """

    def __init__(self, *, cfg: PPVerifierConfig = PPVerifierConfig()):
        self.cfg = cfg

    def verify(self, gen: Generation, tokens: List[Token]) -> List[Issue]:
        token_by_id = {int(t.token_id): t for t in (tokens or [])}
        token_by_ref = {str(getattr(t, "ref", t.cell_id)): t for t in (tokens or [])}

        W = _infer_volume_width(tokens)
        x_mid = 0.5 * float(W) if W > 0 else 0.0

        issues: List[Issue] = []
        for frame_idx, frame in enumerate(gen.frames):
            if bool((gen.refusal or {}).get(int(frame_idx), False)):
                continue

            positive = _frame_is_positive(frame)
            cited_refs = _frame_citations_ref(gen, frame_idx=int(frame_idx), token_by_id=token_by_id)
            cited_tokens = [token_by_ref.get(str(r)) for r in cited_refs]
            cited_tokens = [t for t in cited_tokens if t is not None]

            # R1. NoCitation (fatal for positive claims).
            if positive and not cited_refs:
                issues.append(
                    _issue(
                        frame_idx=frame_idx,
                        issue_type="U1_unsupported",
                        severity=3,
                        rule_id="R1",
                        message="Positive claim without any citations.",
                        cited_refs=[],
                        blame_refs=[],
                    )
                )
                # If there are no citations, other rules are not meaningful.
                continue

            if not positive:
                # pp.md hard rules apply to positive abnormalities.
                continue

            # R2. CoarseOnly: at least one cited token must have level >= l_min.
            l_min = int(self.cfg.l_min)
            levels = [int(t.level) for t in cited_tokens]
            if not levels:
                issues.append(
                    _issue(
                        frame_idx=frame_idx,
                        issue_type="O1_overclaim",
                        severity=2,
                        rule_id="R2",
                        message="Citations refer to unknown tokens; cannot audit granularity.",
                        cited_refs=list(cited_refs),
                        blame_refs=list(cited_refs),
                        rule_inputs={"l_min": l_min},
                        rule_outputs={"levels": []},
                    )
                )
            elif int(max(levels)) < l_min:
                blame = [str(getattr(t, "ref", t.cell_id)) for t in cited_tokens if int(t.level) < l_min]
                if not blame:
                    blame = list(cited_refs)
                issues.append(
                    _issue(
                        frame_idx=frame_idx,
                        issue_type="O1_overclaim",
                        severity=2,
                        rule_id="R2",
                        message=f"Specific positive abnormality supported only by coarse evidence (need level>={l_min}).",
                        cited_refs=list(cited_refs),
                        blame_refs=blame,
                        rule_inputs={"l_min": l_min},
                        rule_outputs={"levels": [int(x) for x in levels], "max_level": int(max(levels))},
                    )
                )

            lat = str(frame.laterality)
            if lat in ("left", "right"):
                sides: List[TokenSide] = [_token_side(t, x_mid=x_mid) for t in cited_tokens]
                want = lat
                violated: List[str] = []
                for r, sd in zip(cited_refs, sides):
                    if sd == "unknown" or sd == "cross" or sd != want:
                        violated.append(str(r))
                if violated:
                    issues.append(
                        _issue(
                            frame_idx=frame_idx,
                            issue_type="I1_inconsistency",
                            severity=2,
                            rule_id="R3",
                            message=f"Laterality mismatch: claim={want}, cited token(s) not exclusively on that side.",
                            cited_refs=list(cited_refs),
                            blame_refs=violated,
                            rule_inputs={"x_mid": float(x_mid)},
                            rule_outputs={"sides": [str(x) for x in sides], "W_inferred": int(W)},
                        )
                    )

            if lat == "bilateral":
                sides = [_token_side(t, x_mid=x_mid) for t in cited_tokens]
                has_left = any(sd == "left" for sd in sides)
                has_right = any(sd == "right" for sd in sides)
                if not (has_left and has_right):
                    issues.append(
                        _issue(
                            frame_idx=frame_idx,
                            issue_type="I1_inconsistency",
                            severity=2,
                            rule_id="R4",
                            message="Bilateral claim requires >=1 left-side and >=1 right-side citation.",
                            cited_refs=list(cited_refs),
                            blame_refs=list(cited_refs),
                            rule_inputs={"x_mid": float(x_mid)},
                            rule_outputs={
                                "sides": [str(x) for x in sides],
                                "has_left": bool(has_left),
                                "has_right": bool(has_right),
                                "W_inferred": int(W),
                            },
                        )
                    )

        return issues


def create_pp_verifier(*, l_min: int = 2) -> PPVerifierV11:
    return PPVerifierV11(cfg=PPVerifierConfig(l_min=int(l_min)))
