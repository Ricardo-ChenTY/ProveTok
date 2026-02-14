from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

IssueType = Literal["U1_unsupported", "O1_overclaim", "I1_inconsistency", "M1_missing_slot"]

# Stable token reference id used by pp.md-style proof objects.
TokenRef = str

@dataclass(frozen=True)
class Token:
    token_id: int
    cell_id: str
    level: int
    embedding: Any         # torch.Tensor (D,) but keep Any to avoid hard import
    score: float
    uncertainty: float
    # Stable ref id (defaults to `cell_id` when unset).
    ref: TokenRef = ""
    # Deterministic geometry in voxel coordinates (z0,z1,y0,y1,x0,x1).
    bounds_voxel: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    center_voxel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Optional mm-space geometry (only available when an affine is provided).
    bounds_mm: Optional[Tuple[float, float, float, float, float, float]] = None
    center_mm: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        # Keep `ref` stable and non-empty by default.
        if not self.ref:
            object.__setattr__(self, "ref", str(self.cell_id))

        # Best-effort center derivation when bounds are available.
        if self.center_voxel == (0.0, 0.0, 0.0) and self.bounds_voxel != (0, 0, 0, 0, 0, 0):
            z0, z1, y0, y1, x0, x1 = (int(x) for x in self.bounds_voxel)
            cz = 0.5 * float(z0 + z1)
            cy = 0.5 * float(y0 + y1)
            cx = 0.5 * float(x0 + x1)
            object.__setattr__(self, "center_voxel", (cz, cy, cx))

@dataclass(frozen=True)
class Frame:
    finding: str
    polarity: str
    laterality: str
    confidence: float
    # Extended slots (paper-aligned). Defaults keep backward compatibility.
    location: str = "unspecified"
    size_bin: str = "unspecified"
    severity: str = "unspecified"
    uncertain: bool = False

@dataclass(frozen=True)
class Generation:
    frames: List[Frame]
    citations: Dict[int, List[int]]   # frame_idx -> token_ids
    q: Dict[int, float]               # frame_idx -> accept prob
    refusal: Dict[int, bool]          # frame_idx -> refusal
    # Stable-id citations for pp.md proof objects (frame_idx -> token refs).
    citations_ref: Optional[Dict[int, List[TokenRef]]] = None
    text: str = ""                    # deterministic narrative (dual-channel protocol)
    impression: str = ""              # optional free-form summary (pp.md contract layer)

    def __post_init__(self) -> None:
        # Keep backward compatibility: default empty citations_ref.
        if self.citations_ref is None:
            object.__setattr__(self, "citations_ref", {})

@dataclass(frozen=True)
class Issue:
    frame_idx: int
    issue_type: IssueType
    severity: int
    rule_id: str
    message: str
    evidence_trace: Dict[str, Any]
