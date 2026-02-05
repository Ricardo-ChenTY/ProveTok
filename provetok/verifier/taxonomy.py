from __future__ import annotations

from typing import Dict, Set


TAXONOMY_VERSION = "taxonomy_v1.0"

# Verifier ruleset version (audited, must be stable for a paper artifact).
RULE_SET_VERSION = "taxonomy_v1.0_ruleset_v1"

# Severity weights used by BET's verifier-driven Δ(c) term.
# Aligns with the proposal: critical > non-critical.
ISSUE_SEVERITY_WEIGHTS: Dict[int, float] = {
    3: 1.0,  # critical
    2: 0.5,  # non-critical major
    1: 0.2,  # non-critical minor
}

# Critical finding set (safety / anti-"封嘴" accounting).
#
# Canonical finding types should be aligned with `provetok.pcg.schema.FINDINGS`.
CRITICAL_FINDINGS_CANON: Set[str] = {
    "pneumothorax",
    "effusion",
    "consolidation",
    "nodule",
}

# Common aliases / refinements observed in clinical text or evidence graphs.
CRITICAL_FINDING_ALIASES: Dict[str, str] = {
    "pleural_effusion": "effusion",
    "large_consolidation": "consolidation",
    "suspicious_nodule": "nodule",
}


def normalize_finding_name(name: str) -> str:
    n = (name or "").strip().lower()
    n = n.replace("-", "_").replace(" ", "_")
    return n


def canonicalize_finding(name: str) -> str:
    n = normalize_finding_name(name)
    return CRITICAL_FINDING_ALIASES.get(n, n)


def is_critical_finding(name: str) -> bool:
    n = canonicalize_finding(name)
    if n in CRITICAL_FINDINGS_CANON:
        return True
    # Also accept substring matches for common refinements (e.g., "suspicious_nodule_...")
    return any(alias in n for alias in CRITICAL_FINDING_ALIASES.keys())

