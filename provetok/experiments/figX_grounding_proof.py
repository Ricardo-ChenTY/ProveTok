"""Paper-grade grounding proof runner (C0004 helper).

Runs ProveTok (lesionness-driven tokenization + score-based citations) against
baseline tokenizers on ReXGroundingCT manifest data, then aggregates metrics with
multi-seed hierarchical bootstrap and paired bootstrap significance tests.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..baselines import FixedGridTokenizer, ROIVarianceTokenizer, SliceTokenizer2p5D
from ..eval.compute_budget import ComputeUnitCosts, format_budget_report, match_b_enc_for_total_flops
from ..eval.metrics_grounding import compute_generation_grounding
from ..eval.stats import bootstrap_mean_ci, bootstrap_quantile_ci, holm_bonferroni, paired_bootstrap_mean_diff
from ..bet.allocator import PickPrefer
from ..bet.refine_loop import run_refine_loop
from ..models.lesionness_head import load_lesionness_head
from ..models.saliency_cnn3d import load_saliency_cnn3d
from ..pcg.generator import ToyPCG
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from ..verifier import verify
from ..grid.cells import parse_cell_id, cell_bounds
from ..data import make_dataloader
from ..types import Token
from .utils import save_results_json, set_seed


@dataclass(frozen=True)
class GroundingProofConfig:
    dataset_type: str = "manifest"
    manifest_path: str = ""
    split: str = "test"
    resize_shape: Tuple[int, int, int] = (64, 64, 64)
    n_samples: int = 200
    use_sample_cache: bool = True
    sample_cache_dir: str = "./outputs/_cache_samples"
    budget_mode: str = "flops"  # "flops" or "tokens"
    budgets: Tuple[float, ...] = (2_000_000.0, 3_000_000.0, 4_000_000.0, 5_000_000.0)
    costs_json: str = ""
    b_gen: int = 128
    n_verify: int = 1
    emb_dim: int = 32
    topk_citations: int = 3  # legacy: applies to both unless overridden
    provetok_topk_citations: int = 3
    baseline_topk_citations: int = 3
    provetok_citation_strategy: str = "score_interleave"  # attention|score|score_interleave|attn_score
    baseline_citation_strategy: str = "attention"  # attention|score|score_interleave|attn_score
    provetok_polarity_strategy: str = "confidence"  # confidence|support
    baseline_polarity_strategy: str = "confidence"  # confidence|support
    provetok_score_bias: float = 0.0
    baseline_score_bias: float = 0.0
    # ProveTok refine controls
    provetok_tokenizer: str = "fixed_grid"  # fixed_grid|refine_loop
    provetok_allocator_prefer: PickPrefer = "score"
    provetok_score_fuse: str = "override"  # override|max|blend
    provetok_score_blend_alpha: float = 1.0
    provetok_score_max_beta: float = 1.0
    provetok_token_score_scale: float = 1.0
    max_steps: int = 40
    max_depth: int = 6
    require_full_budget: bool = True
    score_level_power: float = 0.0
    # Lesionness model
    lesionness_weights: str = ""  # optional; when empty, fall back to TokenEncoder.score (variance heuristic)
    lesionness_device: str = "cpu"
    # Optional stronger scoring model: a 3D saliency CNN that predicts union lesion masks.
    # When provided, token.score will be overridden by mean predicted prob within each cell.
    saliency_weights: str = ""
    saliency_device: str = "cpu"
    # Baseline tokenizers to compare against (keep small but strong)
    baselines: Tuple[str, ...] = ("fixed_grid", "roi_variance", "slice_2p5d")
    # Stats
    seeds: Tuple[int, ...] = (0, 1, 2)
    n_bootstrap: int = 10_000
    ci: float = 0.95
    output_dir: str = "./outputs/figX_grounding_proof"


def _mean_ci_from_seed_sample_matrix(x: np.ndarray, *, n_boot: int, seed: int, ci: float) -> Dict[str, float]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D (S,N), got {x.shape}")
    if x.shape[1] == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    per_sample = x.mean(axis=0)
    res = bootstrap_mean_ci(per_sample.tolist(), n_boot=int(n_boot), seed=int(seed), ci=float(ci))
    return {"mean": float(res.mean), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}


def _p95_ci_from_seed_sample_matrix(x: np.ndarray, *, n_boot: int, seed: int, ci: float) -> Dict[str, float]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D (S,N), got {x.shape}")
    if x.shape[1] == 0:
        return {"p95_s": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    per_sample = x.mean(axis=0)
    res = bootstrap_quantile_ci(per_sample.tolist(), q=0.95, n_boot=int(n_boot), seed=int(seed), ci=float(ci))
    return {"p95_s": float(res.value), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}


def _tokenizers_for(cfg: GroundingProofConfig) -> Dict[str, Any]:
    toks: Dict[str, Any] = {}
    if "fixed_grid" in cfg.baselines:
        toks["fixed_grid"] = FixedGridTokenizer(max_depth=6)
    if "roi_variance" in cfg.baselines:
        toks["roi_variance"] = ROIVarianceTokenizer(candidate_level=4)
    if "slice_2p5d" in cfg.baselines:
        toks["slice_2p5d"] = SliceTokenizer2p5D(level=3, band=3)
    return toks


def _load_samples(cfg: GroundingProofConfig) -> List[Dict[str, Any]]:
    if cfg.dataset_type != "manifest":
        raise ValueError("GroundingProof only supports dataset_type=manifest for now.")
    if not cfg.manifest_path:
        raise ValueError("--manifest is required")

    cache_path: Path | None = None
    if bool(cfg.use_sample_cache):
        try:
            st = os.stat(cfg.manifest_path)
            key = f"{Path(cfg.manifest_path).resolve()}|{cfg.split}|{cfg.resize_shape}|{cfg.n_samples}|{st.st_size}|{int(st.st_mtime)}"
            h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
            cache_dir = Path(cfg.sample_cache_dir)
            cache_path = cache_dir / f"rex_samples_{h}.pt"
        except Exception:
            cache_path = None

    if cache_path is not None and cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu")
            if isinstance(cached, list) and cached:
                return cached
        except Exception:
            pass

    dl = make_dataloader(
        {
            "dataset_type": "manifest",
            "manifest_path": cfg.manifest_path,
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": int(cfg.n_samples),
            "resize_shape": tuple(int(x) for x in cfg.resize_shape),
        },
        split=str(cfg.split),
    )
    out: List[Dict[str, Any]] = []
    for batch in dl:
        out.append(
            {
                "sample_id": str(batch.get("sample_id", [""])[0]),
                "volume": batch["volume"][0],
                "lesion_masks": batch.get("lesion_masks", [{}])[0] or {},
            }
        )
    if not out:
        raise RuntimeError("No samples loaded (empty dataloader).")
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out, cache_path)
        except Exception:
            pass
    return out


def _budget_tokens_for_method(
    *,
    cfg: GroundingProofConfig,
    costs: ComputeUnitCosts,
    flops_total: float,
    method: str,
) -> int:
    if cfg.budget_mode == "tokens":
        return int(flops_total)
    extra = float(_extra_flops_for_method(method, costs))
    return match_b_enc_for_total_flops(
        flops_total=float(flops_total),
        b_gen=int(cfg.b_gen),
        n_verify=int(cfg.n_verify),
        costs=costs,
        flops_extra=float(extra),
        min_b_enc=1,
        max_b_enc=4096,
    )


def _extra_flops_for_method(method: str, costs: ComputeUnitCosts) -> float:
    # Mirror run_baselines: ROI-like baselines include selector cost.
    if method in ("roi_variance",):
        n = 2 ** 4
        num_candidates = int(n * n * n)
        selector_ratio = 0.1
        return float(num_candidates) * float(costs.flops_per_enc_token) * float(selector_ratio)
    return 0.0


def _score_fn_from_lesionness(model: torch.nn.Module, device: torch.device):
    def fn(emb: torch.Tensor) -> torch.Tensor:
        x = emb.to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))
        return prob.detach().cpu()

    return fn


def _fuse_token_scores(
    tokens: List[Token],
    *,
    lesion_scores: List[float],
    score_level_power: float = 0.0,
    score_fuse: str = "override",
    score_blend_alpha: float = 1.0,
    score_max_beta: float = 1.0,
    token_score_scale: float = 1.0,
) -> List[Token]:
    if not tokens:
        return tokens
    fuse = str(score_fuse)
    alpha = float(score_blend_alpha)
    beta = float(score_max_beta)
    scale = float(token_score_scale)
    if fuse not in ("override", "max", "blend"):
        raise ValueError(f"score_fuse must be one of override|max|blend (got {fuse!r})")
    if fuse == "blend" and not (0.0 <= alpha <= 1.0):
        raise ValueError(f"score_blend_alpha must be in [0,1] when score_fuse=blend (got {alpha})")
    if fuse == "max" and beta < 0.0:
        raise ValueError(f"score_max_beta must be >=0 when score_fuse=max (got {beta})")
    if scale < 0.0:
        raise ValueError(f"token_score_scale must be >=0 (got {scale})")
    if len(lesion_scores) != len(tokens):
        raise ValueError(f"lesion_scores must match tokens length (got {len(lesion_scores)} for N={len(tokens)})")

    out: List[Token] = []
    for t, s in zip(tokens, lesion_scores):
        ss = float(s)
        ss = max(0.0, min(1.0, scale * ss))
        base = float(t.score)
        if fuse == "max":
            ss = max(ss, beta * base)
        elif fuse == "blend":
            ss = alpha * ss + (1.0 - alpha) * base
        if float(score_level_power) != 0.0:
            ss = ss * float((1 + int(t.level)) ** float(score_level_power))
        out.append(
            Token(
                token_id=int(t.token_id),
                cell_id=str(t.cell_id),
                level=int(t.level),
                embedding=t.embedding,
                score=float(ss),
                uncertainty=float(t.uncertainty),
            )
        )
    return out


def _apply_token_scores(
    tokens: List[Token],
    *,
    token_score_fn: Any,
    score_level_power: float = 0.0,
    score_fuse: str = "override",
    score_blend_alpha: float = 1.0,
    score_max_beta: float = 1.0,
    token_score_scale: float = 1.0,
) -> List[Token]:
    if token_score_fn is None or not tokens:
        return tokens
    emb = torch.stack([t.embedding for t in tokens], dim=0)
    scores_t = token_score_fn(emb)
    if not isinstance(scores_t, torch.Tensor):
        scores_t = torch.tensor(scores_t)  # type: ignore[arg-type]
    scores = scores_t.detach().cpu().flatten().tolist()
    return _fuse_token_scores(
        tokens,
        lesion_scores=[float(x) for x in scores],
        score_level_power=float(score_level_power),
        score_fuse=str(score_fuse),
        score_blend_alpha=float(score_blend_alpha),
        score_max_beta=float(score_max_beta),
        token_score_scale=float(token_score_scale),
    )


def _scores_from_saliency_prob(prob: torch.Tensor, tokens: List[Token], vol_shape: Tuple[int, int, int]) -> List[float]:
    scores: List[float] = []
    for t in tokens:
        cell = parse_cell_id(str(t.cell_id))
        if cell is None:
            scores.append(0.0)
            continue
        sl = cell_bounds(cell, shape=vol_shape)
        scores.append(float(prob[sl[0], sl[1], sl[2]].mean().item()))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Grounding proof runner for C0004 (paper-grade scaffold).")
    ap.add_argument("--dataset-type", type=str, default="manifest", choices=["manifest"])
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--smoke", action="store_true", help="Quick sanity run (single budget/seed, small N, low bootstrap).")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--no-sample-cache", action="store_true", help="Disable caching of loaded volumes/masks (slower but avoids cache stale issues).")
    ap.add_argument("--sample-cache-dir", type=str, default="./outputs/_cache_samples")
    ap.add_argument("--budget-mode", type=str, default="flops", choices=["flops", "tokens"])
    ap.add_argument("--budgets", type=float, nargs="+", required=True, help="FLOPs total budgets (or b_enc when budget-mode=tokens).")
    ap.add_argument("--costs-json", type=str, default="")
    ap.add_argument("--b-gen", type=int, default=128)
    ap.add_argument("--n-verify", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--topk-citations", type=int, default=3)
    ap.add_argument("--provetok-topk-citations", type=int, default=None, help="Override topk citations for ProveTok only.")
    ap.add_argument("--baseline-topk-citations", type=int, default=None, help="Override topk citations for baselines only.")
    ap.add_argument(
        "--provetok-citation-strategy",
        type=str,
        default="score_interleave",
        choices=["attention", "score", "score_interleave", "attn_score"],
        help="Citation selection strategy for ProveTok (affects union citations and IoU/Dice).",
    )
    ap.add_argument(
        "--baseline-citation-strategy",
        type=str,
        default="attention",
        choices=["attention", "score", "score_interleave", "attn_score"],
        help="Citation selection strategy for baseline tokenizers.",
    )
    ap.add_argument("--provetok-tokenizer", type=str, default="fixed_grid", choices=["fixed_grid", "refine_loop"], help="ProveTok tokenization method.")
    ap.add_argument(
        "--provetok-allocator-prefer",
        type=str,
        default="score",
        choices=["uncertainty", "score"],
        help="When provetok-tokenizer=refine_loop and use_evidence_head=False, pick which token attribute to refine on.",
    )
    ap.add_argument(
        "--provetok-polarity-strategy",
        type=str,
        default="confidence",
        choices=["confidence", "support"],
        help="ToyPCG polarity decision strategy for ProveTok (affects positive_only grounding).",
    )
    ap.add_argument(
        "--baseline-polarity-strategy",
        type=str,
        default="confidence",
        choices=["confidence", "support"],
        help="ToyPCG polarity decision strategy for baselines.",
    )
    ap.add_argument("--provetok-score-bias", type=float, default=0.0, help="Add score_bias * token.score to ToyPCG attention logits for ProveTok.")
    ap.add_argument("--baseline-score-bias", type=float, default=0.0, help="Add score_bias * token.score to ToyPCG attention logits for baselines.")
    ap.add_argument("--provetok-score-fuse", type=str, default="override", choices=["override", "max", "blend"], help="How to fuse lesionness score with the tokenizer's base score.")
    ap.add_argument("--provetok-score-blend-alpha", type=float, default=1.0, help="When provetok-score-fuse=blend: alpha*lesion + (1-alpha)*base.")
    ap.add_argument("--provetok-score-max-beta", type=float, default=1.0, help="When provetok-score-fuse=max: use max(lesion, beta*base).")
    ap.add_argument("--provetok-token-score-scale", type=float, default=1.0, help="Scale the lesionness head output before fusion (clamped to [0,1]).")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--require-full-budget", action="store_true")
    ap.add_argument("--score-level-power", type=float, default=0.0, help="Multiply token scores by (level+1)^p when using score-based refinement/citations.")
    ap.add_argument("--lesionness-weights", type=str, default="", help="Optional path to lesionness_head.pt (when empty: use variance score).")
    ap.add_argument("--lesionness-device", type=str, default="cpu")
    ap.add_argument("--saliency-weights", type=str, default="", help="Optional saliency_cnn3d.pt for token scoring (overrides token.score).")
    ap.add_argument("--saliency-device", type=str, default="cpu", help="Device for saliency model inference (cpu/cuda).")
    ap.add_argument("--baselines", type=str, nargs="+", default=["fixed_grid", "roi_variance", "slice_2p5d"])
    ap.add_argument("--output-dir", type=str, default="./outputs/figX_grounding_proof")
    args = ap.parse_args()

    if bool(args.smoke):
        pt_topk = int(args.topk_citations) if args.provetok_topk_citations is None else int(args.provetok_topk_citations)
        bl_topk = int(args.topk_citations) if args.baseline_topk_citations is None else int(args.baseline_topk_citations)
        cfg = GroundingProofConfig(
            dataset_type=str(args.dataset_type),
            manifest_path=str(args.manifest),
            split=str(args.split),
            resize_shape=(32, 32, 32),
            n_samples=min(10, int(args.n_samples)),
            use_sample_cache=not bool(args.no_sample_cache),
            sample_cache_dir=str(args.sample_cache_dir),
            budget_mode=str(args.budget_mode),
            budgets=(float(args.budgets[0]),),
            costs_json=str(args.costs_json),
            b_gen=int(args.b_gen),
            n_verify=int(args.n_verify),
            emb_dim=int(args.emb_dim),
            topk_citations=int(args.topk_citations),
            provetok_topk_citations=int(pt_topk),
            baseline_topk_citations=int(bl_topk),
            provetok_citation_strategy=str(args.provetok_citation_strategy),
            baseline_citation_strategy=str(args.baseline_citation_strategy),
            provetok_polarity_strategy=str(args.provetok_polarity_strategy),
            baseline_polarity_strategy=str(args.baseline_polarity_strategy),
            provetok_score_bias=float(args.provetok_score_bias),
            baseline_score_bias=float(args.baseline_score_bias),
            provetok_tokenizer=str(args.provetok_tokenizer),
            provetok_allocator_prefer=str(args.provetok_allocator_prefer),  # type: ignore[arg-type]
            provetok_score_fuse=str(args.provetok_score_fuse),
            provetok_score_blend_alpha=float(args.provetok_score_blend_alpha),
            provetok_score_max_beta=float(args.provetok_score_max_beta),
            provetok_token_score_scale=float(args.provetok_token_score_scale),
            max_steps=min(20, int(args.max_steps)),
            max_depth=min(5, int(args.max_depth)),
            require_full_budget=True,
            score_level_power=float(args.score_level_power) if args.score_level_power else 1.0,
            lesionness_weights=str(args.lesionness_weights),
            lesionness_device=str(args.lesionness_device),
            saliency_weights=str(args.saliency_weights),
            saliency_device=str(args.saliency_device),
            baselines=tuple(str(x) for x in args.baselines),
            seeds=(int(args.seeds[0]),),
            n_bootstrap=min(1_000, int(args.n_bootstrap)),
            ci=float(args.ci),
            output_dir=str(args.output_dir),
        )
    else:
        pt_topk = int(args.topk_citations) if args.provetok_topk_citations is None else int(args.provetok_topk_citations)
        bl_topk = int(args.topk_citations) if args.baseline_topk_citations is None else int(args.baseline_topk_citations)
        cfg = GroundingProofConfig(
            dataset_type=str(args.dataset_type),
            manifest_path=str(args.manifest),
            split=str(args.split),
            resize_shape=tuple(int(x) for x in args.resize_shape),
            n_samples=int(args.n_samples),
            use_sample_cache=not bool(args.no_sample_cache),
            sample_cache_dir=str(args.sample_cache_dir),
            budget_mode=str(args.budget_mode),
            budgets=tuple(float(x) for x in args.budgets),
            costs_json=str(args.costs_json),
            b_gen=int(args.b_gen),
            n_verify=int(args.n_verify),
            emb_dim=int(args.emb_dim),
            topk_citations=int(args.topk_citations),
            provetok_topk_citations=int(pt_topk),
            baseline_topk_citations=int(bl_topk),
            provetok_citation_strategy=str(args.provetok_citation_strategy),
            baseline_citation_strategy=str(args.baseline_citation_strategy),
            provetok_polarity_strategy=str(args.provetok_polarity_strategy),
            baseline_polarity_strategy=str(args.baseline_polarity_strategy),
            provetok_score_bias=float(args.provetok_score_bias),
            baseline_score_bias=float(args.baseline_score_bias),
            provetok_tokenizer=str(args.provetok_tokenizer),
            provetok_allocator_prefer=str(args.provetok_allocator_prefer),  # type: ignore[arg-type]
            provetok_score_fuse=str(args.provetok_score_fuse),
            provetok_score_blend_alpha=float(args.provetok_score_blend_alpha),
            provetok_score_max_beta=float(args.provetok_score_max_beta),
            provetok_token_score_scale=float(args.provetok_token_score_scale),
            max_steps=int(args.max_steps),
            max_depth=int(args.max_depth),
            require_full_budget=bool(args.require_full_budget),
            score_level_power=float(args.score_level_power),
            lesionness_weights=str(args.lesionness_weights),
            lesionness_device=str(args.lesionness_device),
            saliency_weights=str(args.saliency_weights),
            saliency_device=str(args.saliency_device),
            baselines=tuple(str(x) for x in args.baselines),
            seeds=tuple(int(s) for s in args.seeds),
            n_bootstrap=int(args.n_bootstrap),
            ci=float(args.ci),
            output_dir=str(args.output_dir),
        )

    set_seed(int(cfg.seeds[0]) if cfg.seeds else 0)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Meta
    repo_root = Path(__file__).resolve().parents[2]
    data_revision, split_manifest_path = try_manifest_revision(cfg.manifest_path)
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(cfg.seeds[0]) if cfg.seeds else 0,
        config=asdict(cfg),
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    costs = ComputeUnitCosts.from_json(cfg.costs_json) if cfg.costs_json else ComputeUnitCosts()
    samples = _load_samples(cfg)

    token_score_fn = None
    if cfg.lesionness_weights:
        # Load lesionness model once.
        lesion_model = load_lesionness_head(cfg.lesionness_weights, map_location="cpu")
        lesion_device = torch.device(cfg.lesionness_device)
        lesion_model = lesion_model.to(lesion_device).eval()
        token_score_fn = _score_fn_from_lesionness(lesion_model, lesion_device)

    saliency_model = None
    saliency_device = torch.device(str(cfg.saliency_device or "cpu"))
    if cfg.saliency_weights:
        saliency_model = load_saliency_cnn3d(cfg.saliency_weights, map_location="cpu")
        saliency_model = saliency_model.to(saliency_device).eval()

    def _preprocess_volume_for_saliency(vol: torch.Tensor) -> torch.Tensor:
        # Keep consistent with train_saliency_cnn3d.py defaults.
        v = vol.float()
        v = v.clamp(min=-1000.0, max=1000.0)
        v = v / 1000.0
        return v

    if saliency_model is not None:
        for idx, s in enumerate(samples):
            if "saliency_prob" in s:
                continue
            volume = s["volume"]
            with torch.no_grad():
                x = _preprocess_volume_for_saliency(volume)
                if x.dim() == 3:
                    x = x.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
                prob = saliency_model.predict_proba(x.to(device=saliency_device)).detach().cpu()[0, 0]  # (D,H,W)
            s["saliency_prob"] = prob
            if idx == 0 or ((idx + 1) % 25) == 0:
                print(f"[figX_grounding_proof] cached saliency {idx+1}/{len(samples)}", flush=True)

    tokenizers = _tokenizers_for(cfg)
    methods = ["provetok_lesionness"] + list(tokenizers.keys())

    metric_keys = [
        "iou_union",
        "dice_union",
        "hit",
        "hit_any_intersection",
        "hit_lesion_coverage",
        "overlap_ratio_token",
        "overlap_ratio_lesion",
    ]

    per_budget_seed: Dict[str, Dict[str, Dict[str, List[float]]]] = {}  # b -> seed -> method.metric -> [N]
    per_budget_seed_meta: Dict[str, Dict[str, Any]] = {}
    per_budget_seed_dirs: Dict[str, Dict[str, str]] = {}

    for bidx, budget in enumerate(cfg.budgets):
        b_key = f"{budget:g}"
        per_budget_seed[b_key] = {}
        per_budget_seed_dirs[b_key] = {}
        per_budget_seed_meta[b_key] = {"budget": float(budget)}

        for seed in cfg.seeds:
            seed_key = str(int(seed))
            seed_dir = Path(cfg.output_dir) / f"budget_{b_key.replace('.', '_')}" / f"seed_{seed_key}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            per_budget_seed_dirs[b_key][seed_key] = str(seed_dir)

            set_seed(int(seed))
            pcg_provetok = ToyPCG(
                emb_dim=int(cfg.emb_dim),
                topk=int(cfg.provetok_topk_citations),
                seed=int(seed),
                score_bias=float(cfg.provetok_score_bias),
                refusal_threshold=0.0,
                citation_strategy=str(cfg.provetok_citation_strategy),
                polarity_strategy=str(cfg.provetok_polarity_strategy),
            )
            pcg_base = ToyPCG(
                emb_dim=int(cfg.emb_dim),
                topk=int(cfg.baseline_topk_citations),
                seed=int(seed),
                score_bias=float(cfg.baseline_score_bias),
                refusal_threshold=0.0,
                citation_strategy=str(cfg.baseline_citation_strategy),
                polarity_strategy=str(cfg.baseline_polarity_strategy),
            )

            # Per-method metric arrays
            out_seed: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in metric_keys} for m in methods}
            out_seed_aux: Dict[str, Any] = {"sample_ids": [], "b_enc": {}, "latency_warm_s": {}}

            # Precompute b_enc targets per method for this budget.
            for m in methods:
                out_seed_aux["b_enc"][m] = int(_budget_tokens_for_method(cfg=cfg, costs=costs, flops_total=float(budget), method=m))
                out_seed_aux["latency_warm_s"][m] = []

            for i, s in enumerate(samples):
                sample_seed = int(seed) + int(i)
                volume = s["volume"]
                lesion_masks = s.get("lesion_masks", {}) or {}
                vol_shape = tuple(int(x) for x in volume.shape)
                out_seed_aux["sample_ids"].append(str(s.get("sample_id", "")))

                saliency_prob = s.get("saliency_prob") if saliency_model is not None else None

                # ProveTok: fixed-grid tokenization + lesionness score + score-interleaved citations.
                #
                # Motivation: using score-only refinement can collapse coverage and miss lesions.
                # For pixel-level grounding, keep a full-covering token family and use learned
                # lesionness to steer citations deterministically.
                b_enc_pt = int(out_seed_aux["b_enc"]["provetok_lesionness"])
                t0 = time.perf_counter()
                if str(cfg.provetok_tokenizer) == "refine_loop":
                    def _noop_verifier(_gen, _tokens):
                        return []

                    # Start from the coarsest full-covering grid that fits in the
                    # token budget, then refine within it. This avoids the
                    # "picked the wrong octant early → miss lesion entirely"
                    # failure mode when starting from root_cell().
                    init_level = 0
                    for lvl in range(1, int(cfg.max_depth) + 1):
                        if (2**lvl) ** 3 <= int(b_enc_pt):
                            init_level = int(lvl)
                        else:
                            break

                    res = run_refine_loop(
                        volume=volume,
                        budget_tokens=int(b_enc_pt),
                        steps=int(cfg.max_steps),
                        generator_fn=pcg_provetok,
                        verifier_fn=_noop_verifier,
                        emb_dim=int(cfg.emb_dim),
                        seed=int(sample_seed),
                        require_full_budget=bool(cfg.require_full_budget),
                        use_evidence_head=False,
                        epsilon=0.0,
                        init_level=int(init_level),
                        max_depth=int(cfg.max_depth),
                        allocator_prefer=cfg.provetok_allocator_prefer,
                        token_score_fn=token_score_fn,
                        token_score_scale=float(cfg.provetok_token_score_scale),
                        token_score_fuse=str(cfg.provetok_score_fuse),
                        token_score_blend_alpha=float(cfg.provetok_score_blend_alpha),
                        token_score_max_beta=float(cfg.provetok_score_max_beta),
                        score_level_power=float(cfg.score_level_power),
                    )
                    toks_pt = res.tokens
                else:
                    toks_pt = FixedGridTokenizer(max_depth=int(cfg.max_depth)).tokenize(
                        volume,
                        budget_tokens=b_enc_pt,
                        emb_dim=int(cfg.emb_dim),
                        seed=int(sample_seed),
                    )

                # Apply token scoring for citations:
                # - Prefer saliency_cnn3d if provided (volume-based, more robust to budget growth)
                # - Else use embedding-based lesionness head if provided
                if saliency_prob is not None and toks_pt:
                    sal_scores = _scores_from_saliency_prob(saliency_prob, toks_pt, vol_shape)
                    toks_pt = _fuse_token_scores(
                        toks_pt,
                        lesion_scores=sal_scores,
                        score_level_power=float(cfg.score_level_power),
                        score_fuse=str(cfg.provetok_score_fuse),
                        score_blend_alpha=float(cfg.provetok_score_blend_alpha),
                        score_max_beta=float(cfg.provetok_score_max_beta),
                        token_score_scale=float(cfg.provetok_token_score_scale),
                    )
                else:
                    toks_pt = _apply_token_scores(
                        toks_pt,
                        token_score_fn=token_score_fn,
                        score_level_power=float(cfg.score_level_power),
                        score_fuse=str(cfg.provetok_score_fuse),
                        score_blend_alpha=float(cfg.provetok_score_blend_alpha),
                        score_max_beta=float(cfg.provetok_score_max_beta),
                        token_score_scale=float(cfg.provetok_token_score_scale),
                    )

                # Always re-run PCG after scoring so citations reflect the final token scores.
                gen_pt = pcg_provetok(toks_pt)
                t1 = time.perf_counter()
                out_seed_aux["latency_warm_s"]["provetok_lesionness"].append(float(t1 - t0))

                g_pt = compute_generation_grounding(gen_pt, toks_pt, lesion_masks, vol_shape)
                out_seed["provetok_lesionness"]["iou_union"].append(float(g_pt.get("iou_union", 0.0)))
                out_seed["provetok_lesionness"]["dice_union"].append(float(g_pt.get("dice_union", 0.0)))
                out_seed["provetok_lesionness"]["hit"].append(float(g_pt.get("hit", 0.0)))
                out_seed["provetok_lesionness"]["hit_any_intersection"].append(float(g_pt.get("hit_any_intersection", 0.0)))
                out_seed["provetok_lesionness"]["hit_lesion_coverage"].append(float(g_pt.get("hit_lesion_coverage", 0.0)))
                out_seed["provetok_lesionness"]["overlap_ratio_token"].append(float(g_pt.get("overlap_ratio_token", g_pt.get("overlap_ratio", 0.0))))
                out_seed["provetok_lesionness"]["overlap_ratio_lesion"].append(float(g_pt.get("overlap_ratio_lesion", 0.0)))

                # Baselines: tokenizers + attention citations
                for name, tok in tokenizers.items():
                    b_enc = int(out_seed_aux["b_enc"][name])
                    t0 = time.perf_counter()
                    toks = tok.tokenize(volume, budget_tokens=b_enc, emb_dim=int(cfg.emb_dim), seed=int(sample_seed))
                    t1 = time.perf_counter()
                    out_seed_aux["latency_warm_s"][name].append(float(t1 - t0))
                    gen = pcg_base(toks)
                    g = compute_generation_grounding(gen, toks, lesion_masks, vol_shape)
                    out_seed[name]["iou_union"].append(float(g.get("iou_union", 0.0)))
                    out_seed[name]["dice_union"].append(float(g.get("dice_union", 0.0)))
                    out_seed[name]["hit"].append(float(g.get("hit", 0.0)))
                    out_seed[name]["hit_any_intersection"].append(float(g.get("hit_any_intersection", 0.0)))
                    out_seed[name]["hit_lesion_coverage"].append(float(g.get("hit_lesion_coverage", 0.0)))
                    out_seed[name]["overlap_ratio_token"].append(float(g.get("overlap_ratio_token", g.get("overlap_ratio", 0.0))))
                    out_seed[name]["overlap_ratio_lesion"].append(float(g.get("overlap_ratio_lesion", 0.0)))

            # Save per-seed raw
            raw_path = seed_dir / "grounding_raw.json"
            save_results_json({"metrics": out_seed, "aux": out_seed_aux}, str(raw_path))
            per_budget_seed[b_key][seed_key] = out_seed  # type: ignore[assignment]

        # Save a budget accounting snapshot (deterministic; by method).
        bud_report: Dict[str, Any] = {}
        for m in methods:
            b_enc = int(_budget_tokens_for_method(cfg=cfg, costs=costs, flops_total=float(budget), method=m))
            bud_report[m] = format_budget_report(
                b_enc=b_enc,
                b_gen=int(cfg.b_gen),
                n_verify=int(cfg.n_verify),
                costs=costs,
                flops_extra=float(_extra_flops_for_method(m, costs)),
            )
        per_budget_seed_meta[b_key]["budgets"] = bud_report

    # Aggregate multiseed curve and paired bootstrap comparisons (per budget, per metric, per baseline).
    metrics_agg: Dict[str, Dict[str, List[Dict[str, float]]]] = {k: {m: [] for m in methods} for k in metric_keys}
    latency_agg: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        "latency_warm_s": {m: [] for m in methods},
        "latency_warm_p95_s": {m: [] for m in methods},
    }
    paired: Dict[str, Any] = {}

    for bidx, budget in enumerate(cfg.budgets):
        b_key = f"{budget:g}"

        # Build seed-sample matrices per method/metric.
        for m in methods:
            for k in metric_keys:
                mats = []
                for seed in cfg.seeds:
                    seed_key = str(int(seed))
                    mats.append(per_budget_seed[b_key][seed_key][m][k])
                arr = np.asarray(mats, dtype=np.float64)  # (S,N)
                stable = hashlib.sha1(f"{m}:{k}".encode("utf-8")).digest()
                stable_seed = int.from_bytes(stable[:4], "little", signed=False)
                ci_rec = _mean_ci_from_seed_sample_matrix(
                    arr,
                    n_boot=int(cfg.n_bootstrap),
                    seed=int(cfg.seeds[0]) + 1000 * bidx + (stable_seed % 997),
                    ci=float(cfg.ci),
                )
                metrics_agg[k][m].append(ci_rec)

        # Latency aggregation from per-seed raw files (optional; best-effort).
        for m in methods:
            mats = []
            for seed in cfg.seeds:
                seed_dir = Path(per_budget_seed_dirs[b_key][str(int(seed))])
                raw = Path(seed_dir / "grounding_raw.json")
                d = {} if not raw.exists() else json_load(raw)
                mats.append((d.get("aux") or {}).get("latency_warm_s", {}).get(m, []))
            if mats and mats[0]:
                min_n = min(len(x) for x in mats)
                arr = np.asarray([x[:min_n] for x in mats], dtype=np.float64)
                ci_rec = _mean_ci_from_seed_sample_matrix(
                    arr, n_boot=int(cfg.n_bootstrap), seed=int(cfg.seeds[0]) + 2000 * bidx, ci=float(cfg.ci)
                )
                p95_rec = _p95_ci_from_seed_sample_matrix(
                    arr, n_boot=int(cfg.n_bootstrap), seed=int(cfg.seeds[0]) + 3000 * bidx, ci=float(cfg.ci)
                )
            else:
                ci_rec = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
                p95_rec = {"p95_s": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            latency_agg["latency_warm_s"][m].append(ci_rec)
            latency_agg["latency_warm_p95_s"][m].append(p95_rec)

        # Paired bootstrap diffs: provetok_lesionness vs each baseline for key metrics.
        paired[b_key] = {}
        for base in tokenizers.keys():
            paired[b_key][base] = {}
            pvals = []
            recs = []
            keys_for_test = ["iou_union", "dice_union", "hit_any_intersection", "hit_lesion_coverage"]
            for k in keys_for_test:
                # Pairing uses per-sample mean over seeds.
                mats_pt = np.asarray([per_budget_seed[b_key][str(int(s))]["provetok_lesionness"][k] for s in cfg.seeds], dtype=np.float64)
                mats_bl = np.asarray([per_budget_seed[b_key][str(int(s))][base][k] for s in cfg.seeds], dtype=np.float64)
                pt = mats_pt.mean(axis=0).tolist()
                bl = mats_bl.mean(axis=0).tolist()
                stable = hashlib.sha1(f"{b_key}:{base}:{k}".encode("utf-8")).digest()
                stable_seed = int.from_bytes(stable[:4], "little", signed=False)
                r = paired_bootstrap_mean_diff(pt, bl, n_boot=int(cfg.n_bootstrap), seed=int(cfg.seeds[0]) + (stable_seed % 10007), ci=float(cfg.ci))
                rec = {"mean_diff": float(r.mean_diff), "ci_low": float(r.ci_low), "ci_high": float(r.ci_high), "p_value": float(r.p_value)}
                paired[b_key][base][k] = rec
                pvals.append(float(r.p_value))
                recs.append(rec)
            p_holm = holm_bonferroni(pvals)
            for k, ph in zip(keys_for_test, p_holm):
                paired[b_key][base][k]["p_value_holm"] = float(ph)

    report: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "budget_mode": cfg.budget_mode,
        "budgets": [float(x) for x in cfg.budgets],
        "seeds": [int(x) for x in cfg.seeds],
        "n_samples": int(cfg.n_samples),
        "n_bootstrap": int(cfg.n_bootstrap),
        "ci": float(cfg.ci),
        "methods": methods,
        "metrics": metrics_agg,
        "latency": latency_agg,
        "paired_bootstrap": paired,
        "budgets_by_method": per_budget_seed_meta,
        "per_budget_seed_dirs": per_budget_seed_dirs,
    }

    out_path = Path(cfg.output_dir) / "figX_grounding_proof.json"
    save_results_json(report, str(out_path))
    print(f"Saved -> {out_path}")


def json_load(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
