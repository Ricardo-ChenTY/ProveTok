from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


class MissingClinicalMetricDependency(RuntimeError):
    """Raised when optional clinical metric dependencies/weights are unavailable."""


@dataclass(frozen=True)
class ClinicalMetricConfig:
    """Clinical-metric computation config (Table 1 scaffold in pp.md).

    Notes:
    - These metrics typically require external model weights and/or toolkits that
      are not shipped in this open-source scaffold.
    - When a metric is enabled but missing dependencies, callers should treat
      the returned value as NaN and proceed (paper-grade runners use nanmean).
    """

    compute_chexbert_f1: bool = False
    compute_radgraph_f1: bool = False
    compute_radcliq: bool = False
    compute_green: bool = False
    compute_ratescore: bool = False

    device: str = "cpu"
    chexbert_weights: str = ""


def compute_clinical_metrics(
    pred: str,
    ref: str,
    *,
    cfg: ClinicalMetricConfig = ClinicalMetricConfig(),
) -> Dict[str, float]:
    """Compute Table 1-style clinical metrics (stub-by-default).

    Returns:
        Dict with keys among:
        - chexbert_f1
        - radgraph_f1
        - radcliq
        - green
        - ratescore
    """
    _ = pred
    _ = ref
    out: Dict[str, float] = {}

    def missing(name: str) -> float:
        # Use NaN so nanmean/nanstd aggregation works out of the box.
        _ = name
        return float("nan")

    if bool(cfg.compute_chexbert_f1):
        out["chexbert_f1"] = missing("chexbert_f1")
    if bool(cfg.compute_radgraph_f1):
        out["radgraph_f1"] = missing("radgraph_f1")
    if bool(cfg.compute_radcliq):
        out["radcliq"] = missing("radcliq")
    if bool(cfg.compute_green):
        out["green"] = missing("green")
    if bool(cfg.compute_ratescore):
        out["ratescore"] = missing("ratescore")

    return out


__all__ = [
    "ClinicalMetricConfig",
    "MissingClinicalMetricDependency",
    "compute_clinical_metrics",
]

