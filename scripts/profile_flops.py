from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.profiler import ProfilerActivity, profile

# Ensure repo root is on sys.path when running as `python scripts/*.py ...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.models.pcg_head import PCGHead
from provetok.bet.evidence_head import EvidenceHead
from provetok.models.encoder3d import ToyEncoder3D


def _sum_flops(prof) -> int:
    return int(sum(getattr(e, "flops", 0) for e in prof.key_averages()))


def profile_pcg_head(*, emb_dim: int, num_tokens: int, num_findings: int, device: torch.device) -> Dict[str, Any]:
    model = PCGHead(emb_dim=emb_dim, num_findings=num_findings).to(device)
    model.eval()
    x = torch.randn((num_tokens, emb_dim), device=device)

    activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        with profile(activities=activities, with_flops=True) as prof:
            _ = model(x)

    flops = _sum_flops(prof)
    return {
        "pcg_head_forward_flops_total": flops,
        "pcg_head_forward_flops_per_token": float(flops) / max(num_tokens, 1),
    }


def profile_evidence_head(*, emb_dim: int, num_tokens: int, device: torch.device) -> Dict[str, Any]:
    model = EvidenceHead(emb_dim=emb_dim).to(device)
    model.eval()
    x = torch.randn((num_tokens, emb_dim), device=device)

    activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        with profile(activities=activities, with_flops=True) as prof:
            _ = model(x)

    flops = _sum_flops(prof)
    return {
        "evidence_head_forward_flops_total": flops,
        "evidence_head_forward_flops_per_token": float(flops) / max(num_tokens, 1),
    }


def profile_toy_encoder(*, emb_dim: int, vol_shape: tuple[int, int, int], device: torch.device) -> Dict[str, Any]:
    model = ToyEncoder3D(in_channels=1, emb_dim=emb_dim).to(device)
    model.eval()
    x = torch.randn((1, 1, *vol_shape), device=device)

    activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        with profile(activities=activities, with_flops=True) as prof:
            _ = model(x)

    flops = _sum_flops(prof)
    return {
        "toy_encoder_forward_flops_total": flops,
        "toy_encoder_forward_flops_per_voxel": float(flops) / max(int(vol_shape[0] * vol_shape[1] * vol_shape[2]), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile approximate FLOPs using torch.profiler (scaffold).")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--num-tokens", type=int, default=64)
    ap.add_argument("--num-findings", type=int, default=8)
    ap.add_argument("--vol-shape", type=int, nargs=3, default=[32, 64, 64])
    ap.add_argument("--out", type=str, default="./outputs/flops_profile.json")
    ap.add_argument("--out-costs", type=str, default="", help="Optional path to write ComputeUnitCosts JSON.")
    ap.add_argument("--flops-per-verify-call", type=float, default=10.0, help="Verifier FLOPs per call (not profiled).")
    args = ap.parse_args()

    device = torch.device(args.device)
    out: Dict[str, Any] = {
        "config": {
            "device": str(device),
            "emb_dim": int(args.emb_dim),
            "num_tokens": int(args.num_tokens),
            "num_findings": int(args.num_findings),
            "vol_shape": list(args.vol_shape),
        }
    }
    out.update(profile_pcg_head(emb_dim=args.emb_dim, num_tokens=args.num_tokens, num_findings=args.num_findings, device=device))
    out.update(profile_evidence_head(emb_dim=args.emb_dim, num_tokens=args.num_tokens, device=device))
    out.update(profile_toy_encoder(emb_dim=args.emb_dim, vol_shape=tuple(args.vol_shape), device=device))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    out_costs = None
    if args.out_costs:
        out_costs = {
            # Unit-cost model used by provetok.eval.compute_budget.ComputeUnitCosts.
            "flops_per_enc_token": float(out.get("evidence_head_forward_flops_per_token", 1.0)),
            "flops_per_dec_token": float(out.get("pcg_head_forward_flops_per_token", 2.0)),
            "flops_per_verify_call": float(args.flops_per_verify_call),
            "source_profile_json": str(out_path),
        }
        costs_path = Path(args.out_costs)
        costs_path.parent.mkdir(parents=True, exist_ok=True)
        costs_path.write_text(json.dumps(out_costs, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"out": str(out_path), "out_costs": str(args.out_costs) if args.out_costs else ""}, indent=2))


if __name__ == "__main__":
    main()
