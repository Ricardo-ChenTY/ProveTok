from __future__ import annotations

import argparse
import json
import time
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path when running as `python scripts/bench_latency.py ...`
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.models.system import ProveTokSystem
from provetok.data.io import load_volume


def _p95(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(np.quantile(np.asarray(xs, dtype=np.float64), 0.95))


def _time_inference(system: ProveTokSystem, volume: torch.Tensor, *, seed: int) -> float:
    t0 = time.perf_counter()
    _ = system.inference(volume, use_refinement=True, seed=seed)
    t1 = time.perf_counter()
    return float(t1 - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Latency benchmark (mean + P95, cold vs warm)")
    ap.add_argument("--smoke", action="store_true", help="Quick run (small N)")
    ap.add_argument("--n", type=int, default=200, help="Number of samples for warm P95")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--emb-dim", type=int, default=32)
    args = ap.parse_args()

    n = 20 if args.smoke else int(args.n)

    device = torch.device(args.device)
    system = ProveTokSystem(emb_dim=args.emb_dim, budget_tokens=args.budget, bet_steps=args.steps).to(device)
    system.eval()

    # Cold: new system instance each time (includes module init + first inference)
    cold_times = []
    for i in range(max(3, min(10, n))):
        sys_i = ProveTokSystem(emb_dim=args.emb_dim, budget_tokens=args.budget, bet_steps=args.steps).to(device)
        vol = load_volume(None, seed=args.seed + i, shape=(32, 64, 64)).to(device)
        cold_times.append(_time_inference(sys_i, vol, seed=args.seed + i))

    # Warm: reuse system
    warm_times = []
    for i in range(n):
        vol = load_volume(None, seed=args.seed + 10_000 + i, shape=(32, 64, 64)).to(device)
        warm_times.append(_time_inference(system, vol, seed=args.seed + 10_000 + i))

    report = {
        "device": str(device),
        "n_warm": n,
        "n_cold": len(cold_times),
        "cold_mean_s": float(mean(cold_times)) if cold_times else 0.0,
        "cold_p95_s": _p95(cold_times),
        "warm_mean_s": float(mean(warm_times)) if warm_times else 0.0,
        "warm_p95_s": _p95(warm_times),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
