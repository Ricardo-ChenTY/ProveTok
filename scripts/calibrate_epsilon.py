from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

# Ensure repo root is on sys.path when running as `python scripts/*.py ...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from provetok.bet.evidence_head import EvidenceHead, rank_cells_by_delta
from provetok.bet.tokenize import encode_tokens
from provetok.grid.cells import root_cell, split, Cell
from provetok.pcg.generator import ToyPCG
from provetok.verifier import verify
from provetok.experiments.utils import create_synthetic_volume, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Calibrate BET stop threshold epsilon (ε) using a fixed quantile rule on a dev set.\n\n"
            "Implements a minimal version of the proposal: collect candidate Δ(c) values and\n"
            "choose ε as a fixed quantile (default 5%). The selected ε should be shared across\n"
            "all budgets B (do NOT tune per-budget)."
        )
    )
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--volume-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--n-lesions", type=int, default=3)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5, help="How many refinement steps to sample deltas from")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--quantile", type=float, default=0.05, help="Quantile for epsilon selection (e.g., 0.05)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="./outputs/calibrate_epsilon.json")
    args = ap.parse_args()

    set_seed(args.seed)

    evidence_head = EvidenceHead(emb_dim=args.emb_dim)
    deltas: List[float] = []

    for i in range(args.n_samples):
        sample_seed = args.seed + i
        volume, _ = create_synthetic_volume(
            shape=tuple(args.volume_shape),
            n_lesions=args.n_lesions,
            seed=sample_seed,
        )
        pcg = ToyPCG(emb_dim=args.emb_dim, topk=3, seed=sample_seed)

        cells: List[Cell] = [root_cell()]
        issues = []

        for step in range(args.steps):
            tokens = encode_tokens(volume, cells, emb_dim=args.emb_dim, seed=sample_seed)
            gen = pcg(tokens)
            issues = verify(gen, tokens)

            cell_embeddings = {t.cell_id: t.embedding for t in tokens}
            ranked = rank_cells_by_delta(
                cells=cells,
                cell_embeddings=cell_embeddings,
                evidence_head=evidence_head,
                current_issues=issues,
                max_depth=args.max_depth,
                epsilon=0.0,  # collect full distribution
            )
            deltas.extend([float(s.delta) for s in ranked])

            if not ranked:
                break
            # Advance one deterministic split (to explore more deltas)
            c_star = ranked[0].cell
            cells = [c for c in cells if c.id() != c_star.id()] + split(c_star)

    if not deltas:
        raise SystemExit("No deltas collected; increase n-samples/steps or check the evidence head.")

    eps = float(np.quantile(np.asarray(deltas, dtype=np.float64), float(args.quantile)))
    stats = {
        "count": len(deltas),
        "min": float(np.min(deltas)),
        "p05": float(np.quantile(deltas, 0.05)),
        "median": float(np.median(deltas)),
        "p95": float(np.quantile(deltas, 0.95)),
        "max": float(np.max(deltas)),
    }

    out = {
        "config": {
            "n_samples": args.n_samples,
            "volume_shape": list(args.volume_shape),
            "n_lesions": args.n_lesions,
            "emb_dim": args.emb_dim,
            "steps": args.steps,
            "max_depth": args.max_depth,
            "quantile": args.quantile,
            "seed": args.seed,
        },
        "epsilon": eps,
        "delta_stats": stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"epsilon": eps, "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()

