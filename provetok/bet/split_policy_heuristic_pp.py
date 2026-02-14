from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..grid.cells import Cell, cell_stable_id
from ..types import Issue, Token


def _infer_volume_width(tokens: Sequence[Token]) -> int:
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


def _crosses_midline(tok: Token, *, x_mid: float) -> bool:
    b = getattr(tok, "bounds_voxel", None)
    if not isinstance(b, tuple) or len(b) != 6:
        return False
    try:
        x0, x1 = float(b[4]), float(b[5])
    except Exception:
        return False
    return bool(x0 < float(x_mid) < x1)


def _token_volume_vox(tok: Token) -> int:
    b = getattr(tok, "bounds_voxel", None)
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


def _blame_rules_by_ref(tokens: Sequence[Token], issues: Sequence[Issue]) -> Dict[str, Set[str]]:
    token_refs = {str(getattr(t, "ref", t.cell_id)) for t in tokens or []}
    out: Dict[str, Set[str]] = {r: set() for r in token_refs}
    for iss in issues or []:
        rid = str(getattr(iss, "rule_id", ""))
        trace = getattr(iss, "evidence_trace", {}) or {}
        for ref in (trace.get("blame_refs", []) or []):
            if not isinstance(ref, str) or not ref:
                continue
            if ref in out:
                out[ref].add(rid)
    return out


def pick_cell_to_split_heuristic_pp(
    cells: Sequence[Cell],
    tokens: Sequence[Token],
    issues: Sequence[Issue],
) -> Optional[Cell]:
    """pp.md v1.1 heuristic split policy (strong deterministic baseline).

    Priority (pp.md §5.1):
    1) If CoarseOnly (R2): prefer *coarser* blamed tokens (smaller level).
    2) If LateralityMismatch (R3): prefer blamed tokens that cross the midline.
    3) Tie-break by token volume (split the largest blamed cell).
    4) Final tie-break by stable cell id for determinism.
    """
    cand_cells = list(cells or [])
    if not cand_cells:
        return None

    cell_by_id = {c.id(): c for c in cand_cells}
    token_by_ref = {str(getattr(t, "ref", t.cell_id)): t for t in (tokens or [])}

    blame_rules = _blame_rules_by_ref(tokens, issues)
    blamed: List[Tuple[Token, Cell, Set[str]]] = []
    for ref, rules in blame_rules.items():
        if not rules:
            continue
        tok = token_by_ref.get(str(ref))
        if tok is None:
            continue
        c = cell_by_id.get(str(tok.cell_id))
        if c is None:
            continue
        blamed.append((tok, c, set(rules)))

    if not blamed:
        return None

    has_r2 = any(str(getattr(i, "rule_id", "")) == "R2" for i in (issues or []))
    has_r3 = any(str(getattr(i, "rule_id", "")) == "R3" for i in (issues or []))

    W = _infer_volume_width(tokens)
    x_mid = 0.5 * float(W) if W > 0 else 0.0

    def stable_id(c: Cell) -> int:
        return int(cell_stable_id(c))

    # 1) R2: coarser first (smaller level), then largest volume, then stable id.
    if has_r2:
        pool = [(t, c) for (t, c, rules) in blamed if "R2" in rules]
        if pool:
            pool.sort(
                key=lambda tc: (
                    int(getattr(tc[0], "level", tc[1].level)),
                    -int(_token_volume_vox(tc[0])),
                    stable_id(tc[1]),
                )
            )
            return pool[0][1]

    # 2) R3: boundary-crossing first, then largest volume, then stable id.
    if has_r3:
        pool = [(t, c) for (t, c, rules) in blamed if "R3" in rules]
        if pool:
            pool.sort(
                key=lambda tc: (
                    -int(_crosses_midline(tc[0], x_mid=float(x_mid))),
                    -int(_token_volume_vox(tc[0])),
                    stable_id(tc[1]),
                )
            )
            return pool[0][1]

    # 3) Fallback: largest volume among blamed, then stable id.
    pool2 = [(t, c) for (t, c, _rules) in blamed]
    pool2.sort(key=lambda tc: (-int(_token_volume_vox(tc[0])), stable_id(tc[1])))
    return pool2[0][1] if pool2 else None

