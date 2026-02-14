from __future__ import annotations
from typing import List, Dict, Optional, Literal, Sequence
from ..types import Issue, Token
from ..grid.cells import Cell

PickPrefer = Literal["uncertainty", "score"]


def _infer_volume_width_from_tokens(tokens: Sequence[Token]) -> int:
    xs: List[int] = []
    for t in tokens:
        b = getattr(t, "bounds_voxel", (0, 0, 0, 0, 0, 0))
        if not isinstance(b, tuple) or len(b) != 6:
            continue
        try:
            x1 = int(b[5])
        except Exception:
            continue
        if x1 > 0:
            xs.append(x1)
    return int(max(xs)) if xs else 0


def _crosses_midline(tok: Token, *, x_mid: float) -> bool:
    b = getattr(tok, "bounds_voxel", (0, 0, 0, 0, 0, 0))
    if not isinstance(b, tuple) or len(b) != 6:
        return False
    try:
        x0, x1 = float(b[4]), float(b[5])
    except Exception:
        return False
    return bool(x0 < float(x_mid) < x1)


def _token_volume_vox(tok: Token) -> int:
    b = getattr(tok, "bounds_voxel", (0, 0, 0, 0, 0, 0))
    if not isinstance(b, tuple) or len(b) != 6:
        return 0
    try:
        z0, z1, y0, y1, x0, x1 = (int(v) for v in b)
    except Exception:
        return 0
    dz = max(0, int(z1 - z0))
    dy = max(0, int(y1 - y0))
    dx = max(0, int(x1 - x0))
    return int(dz * dy * dx)


def pick_cell_to_split(
    cells: List[Cell],
    tokens: List[Token],
    issues: List[Issue],
    *,
    prefer: PickPrefer = "uncertainty",
) -> Optional[Cell]:
    """Deterministic greedy allocator.

    Priority:
    1) If issues exist: split cells mentioned in evidence traces (highest uncertainty + severity)
    2) Else: split cell with highest uncertainty

    Tie-break by lexicographic cell_id.
    """
    if not cells:
        return None

    cell_by_id = {c.id(): c for c in cells}
    token_by_cell = {t.cell_id: t for t in tokens}
    token_by_ref = {str(getattr(t, "ref", t.cell_id)): t for t in tokens}

    W = _infer_volume_width_from_tokens(tokens)
    x_mid = 0.5 * float(W) if W > 0 else 0.0

    if prefer == "score":
        scored = []
        for c in cells:
            t = token_by_cell.get(c.id())
            s = float(t.score) if t else 0.0
            scored.append((s, c.id(), c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2] if scored else None

    blamed: Dict[str, float] = {}
    for iss in issues:
        trace = iss.evidence_trace or {}
        cids: List[str] = []
        for cid in trace.get("token_cell_ids", []) or []:
            if isinstance(cid, str) and cid:
                cids.append(cid)

        # pp.md verifier traces carry stable refs instead of cell ids.
        for ref in trace.get("blame_refs", []) or []:
            if not isinstance(ref, str) or not ref:
                continue
            t_ref = token_by_ref.get(str(ref))
            if t_ref is not None:
                cids.append(str(t_ref.cell_id))

        # Fallback: some traces carry token ids only.
        for tid in trace.get("token_ids", []) or []:
            try:
                tid_i = int(tid)
            except Exception:
                continue
            t_tok = next((t for t in tokens if int(t.token_id) == tid_i), None)
            if t_tok is not None:
                cids.append(str(t_tok.cell_id))

        for cid in cids:
            t = token_by_cell.get(cid)
            if t is None:
                continue

            score = float(t.uncertainty) + 0.25 * float(getattr(iss, "severity", 0))

            # Heuristic alignment with pp.md split priorities when rule ids are present.
            rid = str(getattr(iss, "rule_id", ""))
            if rid == "R2":
                score += 1.0 / float(1 + max(0, int(getattr(t, "level", 0))))
            elif rid == "R3" and x_mid > 0.0:
                score += 0.5 if _crosses_midline(t, x_mid=float(x_mid)) else 0.0

            score += 1e-9 * float(_token_volume_vox(t))
            blamed[cid] = max(blamed.get(cid, 0.0), float(score))

    if blamed:
        best = sorted(blamed.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return cell_by_id.get(best)

    scored = []
    for c in cells:
        t = token_by_cell.get(c.id())
        u = t.uncertainty if t else 0.0
        scored.append((u, c.id(), c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2] if scored else None
