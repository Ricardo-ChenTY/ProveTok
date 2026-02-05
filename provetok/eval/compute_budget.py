from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ComputeUnitCosts:
    """Estimated compute unit costs (toy scaffold).

    For paper-grade reporting, replace these constants with profiler-calibrated values.
    """

    flops_per_enc_token: float = 1.0
    flops_per_dec_token: float = 2.0
    flops_per_verify_call: float = 10.0

    @classmethod
    def from_json(cls, path: str) -> "ComputeUnitCosts":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            flops_per_enc_token=float(d.get("flops_per_enc_token", 1.0)),
            flops_per_dec_token=float(d.get("flops_per_dec_token", 2.0)),
            flops_per_verify_call=float(d.get("flops_per_verify_call", 10.0)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "flops_per_enc_token": float(self.flops_per_enc_token),
            "flops_per_dec_token": float(self.flops_per_dec_token),
            "flops_per_verify_call": float(self.flops_per_verify_call),
        }


def estimate_total_flops(
    *,
    b_enc: int,
    b_gen: int,
    n_verify: int,
    costs: ComputeUnitCosts = ComputeUnitCosts(),
    flops_extra: float = 0.0,
) -> float:
    return (
        float(flops_extra)
        + float(b_enc) * costs.flops_per_enc_token
        + float(b_gen) * costs.flops_per_dec_token
        + float(n_verify) * costs.flops_per_verify_call
    )


def match_b_enc_for_total_flops(
    *,
    flops_total: float,
    b_gen: int,
    n_verify: int,
    costs: ComputeUnitCosts = ComputeUnitCosts(),
    flops_extra: float = 0.0,
    min_b_enc: int = 1,
    max_b_enc: int = 4096,
) -> int:
    """Solve for b_enc to match a target total FLOPs (toy analytical matching).

    This is a utility for FLOPs-matched experiments when using unit-cost models.
    """
    remaining = float(flops_total) - float(flops_extra) - (
        float(b_gen) * costs.flops_per_dec_token + float(n_verify) * costs.flops_per_verify_call
    )
    if remaining <= 0:
        return int(min_b_enc)
    b_enc = int(round(remaining / max(costs.flops_per_enc_token, 1e-9)))
    return int(max(min_b_enc, min(max_b_enc, b_enc)))


def format_budget_report(
    *,
    b_enc: int,
    b_gen: int,
    n_verify: int,
    costs: ComputeUnitCosts = ComputeUnitCosts(),
    flops_extra: float = 0.0,
) -> Dict[str, float]:
    total = estimate_total_flops(b_enc=b_enc, b_gen=b_gen, n_verify=n_verify, costs=costs, flops_extra=flops_extra)
    return {
        "b_enc": float(b_enc),
        "b_gen": float(b_gen),
        "n_verify": float(n_verify),
        "flops_enc": float(b_enc) * costs.flops_per_enc_token,
        "flops_dec": float(b_gen) * costs.flops_per_dec_token,
        "flops_verify": float(n_verify) * costs.flops_per_verify_call,
        "flops_extra": float(flops_extra),
        "flops_total": total,
    }
