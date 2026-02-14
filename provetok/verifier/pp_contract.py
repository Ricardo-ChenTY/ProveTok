from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set

from ..types import Issue
from ..pcg.text_contract import parse_findings_lines


_CITE_RE = re.compile(r"\[\s*CITE\s*:\s*(?P<ids>[^\]]*)\]", re.IGNORECASE)


def extract_citations_from_text(text: str) -> List[str]:
    """Extract citation ids from all `[CITE: ...]` occurrences in text."""
    out: List[str] = []
    for m in _CITE_RE.finditer(str(text or "")):
        raw = str(m.group("ids") or "")
        for x in raw.split(","):
            x = x.strip()
            if x:
                out.append(x)
    return out


def check_impression_no_new_cite(
    *,
    findings_lines: Sequence[str],
    impression: str,
) -> Optional[Issue]:
    """R0 (contract): Impression must not introduce new citation ids."""
    parsed = parse_findings_lines(findings_lines)
    findings_refs: Set[str] = set()
    for row in parsed:
        for r in row.citations:
            findings_refs.add(str(r))

    imp_refs = set(extract_citations_from_text(impression))
    new_refs = sorted(list(imp_refs.difference(findings_refs)))
    if not new_refs:
        return None

    return Issue(
        frame_idx=-1,
        issue_type="I1_inconsistency",  # type: ignore[arg-type]
        severity=1,
        rule_id="R0",
        message="Impression introduces citation ids not present in Findings.",
        evidence_trace={
            "findings_refs": sorted(findings_refs),
            "impression_refs": sorted(imp_refs),
            "new_refs": new_refs,
        },
    )

