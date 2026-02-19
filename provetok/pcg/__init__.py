"""PCG (proof-carrying generation) package.

Important: keep this module lightweight.

`provetok.data` may import `provetok.pcg.schema` for bounded vocabularies. If
`provetok.pcg.__init__` eagerly imports LLM backends, it can create circular
imports such as:

`provetok.data.frame_extractor -> provetok.pcg.schema -> provetok.pcg.__init__ -> provetok.pcg.llama2_pcg -> provetok.data.frame_extractor`

So we only import the lightweight components here and lazily import the LLM
backend when requested.
"""

from .generator import ToyPCG
from .evidence_graph import (
    EvidenceGraph,
    EvidenceGraphBuilder,
    TokenEvidence,
    SlotValue,
    compute_support_score,
    get_constrained_vocab,
)
from .refusal import (
    RefusalCalibrator,
    RefusalDecision,
    CalibrationMetrics,
    apply_refusal_to_generation,
    format_calibration_report,
)

__all__ = [
    "ToyPCG",
    "EvidenceGraph",
    "EvidenceGraphBuilder",
    "TokenEvidence",
    "SlotValue",
    "compute_support_score",
    "get_constrained_vocab",
    "RefusalCalibrator",
    "RefusalDecision",
    "CalibrationMetrics",
    "apply_refusal_to_generation",
    "format_calibration_report",
    "Llama2PCG",
    "Llama2PCGConfig",
]


def __getattr__(name: str):
    if name in ("Llama2PCG", "Llama2PCGConfig"):
        from .llama2_pcg import Llama2PCG, Llama2PCGConfig

        return Llama2PCG if name == "Llama2PCG" else Llama2PCGConfig
    raise AttributeError(name)
