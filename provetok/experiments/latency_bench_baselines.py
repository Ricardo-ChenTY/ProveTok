from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np
import torch

from ..baselines.tokenizers import (
    FixedGridTokenizer,
    NoRefineTokenizer,
    ROICropTokenizer,
    ROIVarianceTokenizer,
    SliceTokenizer2D,
    SliceTokenizer2p5D,
)
from ..data.io import load_volume
from ..pcg.generator import ToyPCG
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta
from ..verifier.rules import RULE_SET_VERSION, create_verifier
from ..verifier.taxonomy import TAXONOMY_VERSION
from .utils import make_output_dir, save_results_json, set_seed


def _p95(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(np.quantile(np.asarray(xs, dtype=np.float64), 0.95))


@dataclass(frozen=True)
class BenchConfig:
    seed: int = 0
    device: str = "cpu"
    volume_shape: tuple[int, int, int] = (32, 64, 64)
    budget_tokens: int = 128
    emb_dim: int = 32
    topk_citations: int = 3
    warm_n: int = 200
    cold_n: int = 10
    output_dir: str = "./outputs/latency_bench_baselines"


def _bench_one(
    *,
    name: str,
    tokenizer,
    cfg: BenchConfig,
    volumes_warm: List[torch.Tensor],
    volumes_cold: List[torch.Tensor],
) -> Dict[str, Any]:
    verifier = create_verifier()

    # Warm: reuse tokenizer + pcg.
    warm_times: List[float] = []
    pcg = ToyPCG(
        emb_dim=int(cfg.emb_dim),
        topk=int(cfg.topk_citations),
        seed=int(cfg.seed),
        refusal_threshold=0.0,
        citation_strategy="attention",
        q_strategy="confidence",
    )

    for i, vol in enumerate(volumes_warm):
        t0 = time.perf_counter()
        toks = tokenizer.tokenize(vol, budget_tokens=int(cfg.budget_tokens), emb_dim=int(cfg.emb_dim), seed=int(cfg.seed) + i)
        gen = pcg(toks)
        _ = verifier.verify(gen, toks)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        warm_times.append(float(t1 - t0))

    # Cold: re-init tokenizer + pcg each time.
    cold_times: List[float] = []
    for i, vol in enumerate(volumes_cold):
        tok_i = tokenizer.__class__(**getattr(tokenizer, "__dict__", {}))
        pcg_i = ToyPCG(
            emb_dim=int(cfg.emb_dim),
            topk=int(cfg.topk_citations),
            seed=int(cfg.seed) + 10_000 + i,
            refusal_threshold=0.0,
            citation_strategy="attention",
            q_strategy="confidence",
        )
        t0 = time.perf_counter()
        toks = tok_i.tokenize(vol, budget_tokens=int(cfg.budget_tokens), emb_dim=int(cfg.emb_dim), seed=int(cfg.seed) + 10_000 + i)
        gen = pcg_i(toks)
        _ = verifier.verify(gen, toks)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        cold_times.append(float(t1 - t0))

    return {
        "name": str(name),
        "warm_n": int(len(warm_times)),
        "cold_n": int(len(cold_times)),
        "warm_mean_s": float(mean(warm_times)) if warm_times else 0.0,
        "warm_p95_s": _p95(warm_times),
        "cold_mean_s": float(mean(cold_times)) if cold_times else 0.0,
        "cold_p95_s": _p95(cold_times),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Latency benchmark across baseline tokenizers (mean + P95, cold vs warm).")
    ap.add_argument("--smoke", action="store_true", help="Quick run (small N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    ap.add_argument("--budget-tokens", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--topk-citations", type=int, default=3)
    ap.add_argument("--warm-n", type=int, default=200)
    ap.add_argument("--cold-n", type=int, default=10)
    ap.add_argument("--output-dir", type=str, default="./outputs/latency_bench_baselines")
    args = ap.parse_args()

    cfg = BenchConfig(
        seed=int(args.seed),
        device=str(args.device),
        volume_shape=(32, 64, 64),
        budget_tokens=int(args.budget_tokens),
        emb_dim=int(args.emb_dim),
        topk_citations=int(args.topk_citations),
        warm_n=(20 if bool(args.smoke) else int(args.warm_n)),
        cold_n=(3 if bool(args.smoke) else int(args.cold_n)),
        output_dir=str(args.output_dir),
    )

    set_seed(int(cfg.seed))

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available.")

    # Pre-build volumes for stable timing.
    volumes_warm = [load_volume(None, seed=cfg.seed + i, shape=cfg.volume_shape).to(device) for i in range(int(cfg.warm_n))]
    volumes_cold = [load_volume(None, seed=cfg.seed + 20_000 + i, shape=cfg.volume_shape).to(device) for i in range(int(cfg.cold_n))]

    tokenizers = {
        "fixed_grid": FixedGridTokenizer(max_depth=6),
        "provetok_no_refine": NoRefineTokenizer(level=3),
        "slice_2d": SliceTokenizer2D(level=3),
        "slice_2p5d": SliceTokenizer2p5D(level=3, band=3),
        "roi_variance": ROIVarianceTokenizer(candidate_level=3),
        "roi_crop": ROICropTokenizer(candidate_level=3, roi_max_depth=6),
        # Placeholder "strong RRG baseline" without citations (CT2Rep-like contract).
        "ct2rep_like": FixedGridTokenizer(max_depth=6),
    }

    repo_root = Path(__file__).resolve().parents[2]
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(cfg.seed),
        config=asdict(cfg),
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision="synthetic",
        split_manifest_path="",
    )

    rows: Dict[str, Any] = {}
    for name, tok in tokenizers.items():
        rows[str(name)] = _bench_one(
            name=str(name),
            tokenizer=tok,
            cfg=cfg,
            volumes_warm=volumes_warm,
            volumes_cold=volumes_cold,
        )

    out_dir = make_output_dir(cfg.output_dir, "latency_bench_baselines")
    report = {"meta": meta.to_dict(), "config": asdict(cfg), "per_method": rows}

    out_path = Path(out_dir) / "latency_bench_baselines.json"
    save_results_json(report, str(out_path))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

