from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Literal

IssueType = Literal["U1_unsupported", "O1_overclaim", "I1_inconsistency", "M1_missing_slot"]

@dataclass(frozen=True)
class Token:
    token_id: int
    cell_id: str
    level: int
    embedding: Any         # torch.Tensor (D,) but keep Any to avoid hard import
    score: float
    uncertainty: float

@dataclass(frozen=True)
class Frame:
    finding: str
    polarity: str
    laterality: str
    confidence: float

@dataclass(frozen=True)
class Generation:
    frames: List[Frame]
    citations: Dict[int, List[int]]   # frame_idx -> token_ids
    q: Dict[int, float]               # frame_idx -> accept prob
    refusal: Dict[int, bool]          # frame_idx -> refusal

@dataclass(frozen=True)
class Issue:
    frame_idx: int
    issue_type: IssueType
    severity: int
    rule_id: str
    message: str
    evidence_trace: Dict[str, Any]
