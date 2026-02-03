from __future__ import annotations
from typing import List, Dict, Optional
from ..types import Issue, Token
from ..grid.cells import Cell

def pick_cell_to_split(cells: List[Cell], tokens: List[Token], issues: List[Issue]) -> Optional[Cell]:
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

    blamed: Dict[str, float] = {}
    for iss in issues:
        trace = iss.evidence_trace or {}
        for cid in trace.get("token_cell_ids", []):
            t = token_by_cell.get(cid)
            if t is None:
                continue
            blamed[cid] = max(blamed.get(cid, 0.0), float(t.uncertainty + 0.25 * iss.severity))

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
