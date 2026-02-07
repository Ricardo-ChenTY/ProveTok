"""Baseline runner scaffold (proposal §7.2 / §7.3).

Runs a small set of tokenization/protocol baselines on synthetic data and reports:
- Frame F1 (vs a fixed synthetic report-derived GT)
- Grounding IoU (union) vs synthetic lesion masks
- Verifier issue rates (unsupported/overclaim/etc.)
- Estimated FLOPs (toy) via ComputeUnitCosts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..types import Generation, Token
from ..baselines import (
    FixedGridTokenizer,
    FixedGridTokenizerScored,
    NoRefineTokenizer,
    ROIVarianceTokenizer,
    ROICropTokenizer,
    SliceTokenizer2D,
    SliceTokenizer2p5D,
    apply_no_citation,
)
from ..eval.compute_budget import ComputeUnitCosts, format_budget_report
from ..eval.compute_budget import match_b_enc_for_total_flops
from ..eval.metrics_frames import compute_frame_f1
from ..eval.metrics_grounding import compute_generation_grounding
from ..eval.metrics_text import MissingTextMetricDependency, compute_text_metrics
from ..eval.stats import bootstrap_mean_ci
from ..pcg.generator import ToyPCG
from ..pcg.llama2_pcg import Llama2PCG, Llama2PCGConfig
from ..pcg.refusal import RefusalPolicy
from ..pcg.schema_version import SCHEMA_VERSION
from ..models.lesionness_head import load_lesionness_head
from ..models.ct2rep_strong import load_ct2rep_strong
from ..pcg.schema import FINDINGS
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier import verify
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from ..verifier.taxonomy import is_critical_finding
from ..data import make_dataloader
from ..data.frame_extractor import FrameExtractor, frames_to_report
from .utils import create_synthetic_volume, make_output_dir, save_results_json, set_seed


@dataclass
class BaselineRunConfig:
    dataset_type: str = "synthetic"  # "synthetic" or "manifest"
    manifest_path: str = ""
    split: str = "test"
    n_samples: int = 30
    volume_shape: Tuple[int, int, int] = (64, 64, 64)
    n_lesions: int = 3
    budget_tokens: int = 64
    emb_dim: int = 32
    topk_citations: int = 3
    seed: int = 42
    output_dir: str = "./outputs/baselines"
    refusal_policy_path: str = ""
    flops_total: float = 0.0
    b_gen: int = 128
    n_verify: int = 1
    costs_json: str = ""
    selector_ratio: float = 0.1
    resize_shape: Tuple[int, int, int] = (64, 64, 64)
    pcg_backend: str = "toy"  # "toy" or "llama2"
    llama2_path: str = "/data/models/Llama-2-7b-chat-hf"
    llama2_quant: str = "fp16"  # "fp16" or "8bit"
    nlg_weight: float = 0.5
    grounding_weight: float = 0.5
    lesionness_weights: str = ""
    lesionness_device: str = "cpu"
    lesionness_score_level_power: float = 0.0
    ct2rep_strong_weights: str = ""
    ct2rep_strong_device: str = "cpu"
    compute_text_metrics: bool = True


def _grounding_union(gen: Generation, tokens, lesion_masks, volume_shape) -> Dict[str, float]:
    return {k: float(v) for k, v in compute_generation_grounding(gen, tokens, lesion_masks, volume_shape).items()}


def _score_fn_from_lesionness(model: torch.nn.Module, device: torch.device):
    def fn(emb: torch.Tensor) -> torch.Tensor:
        x = emb.to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))
        return prob.detach().cpu()

    return fn


def _apply_token_scores(tokens: List[Token], *, token_score_fn: Any) -> List[Token]:
    if token_score_fn is None or not tokens:
        return tokens
    emb = torch.stack([t.embedding for t in tokens], dim=0)
    scores_t = token_score_fn(emb)
    if not isinstance(scores_t, torch.Tensor):
        scores_t = torch.tensor(scores_t)  # type: ignore[arg-type]
    scores = scores_t.detach().cpu().flatten().tolist()
    if len(scores) != len(tokens):
        raise ValueError(f"token_score_fn must return N scores, got {len(scores)} for N={len(tokens)}")
    out: List[Token] = []
    for t, s in zip(tokens, scores):
        out.append(
            Token(
                token_id=int(t.token_id),
                cell_id=str(t.cell_id),
                level=int(t.level),
                embedding=t.embedding,
                score=float(s),
                uncertainty=float(t.uncertainty),
            )
        )
    return out


def _generate_ct2rep_strong(tokens: List[Token], *, model: torch.nn.Module, topk: int) -> Generation:
    if not tokens:
        return Generation(frames=[], citations={}, q={}, refusal={}, text="")

    device = next(model.parameters()).device
    T = torch.stack([t.embedding for t in tokens], dim=0).to(device=device)
    logits, att = model(T)  # (K,C), (K,N)
    probs = torch.softmax(logits, dim=-1)

    frames = []
    citations: Dict[int, List[int]] = {}
    q: Dict[int, float] = {}
    refusal: Dict[int, bool] = {}
    token_ids = [int(t.token_id) for t in tokens]

    # Emit at most one positive frame to avoid the degenerate all-none argmax case
    # on imbalanced manifest labels (many findings absent by default).
    p_present = probs[:, 1]  # (K,)
    k = int(torch.argmax(p_present).item())
    conf = float(p_present[k].item())
    min_conf = 0.15
    if conf >= float(min_conf):
        from ..types import Frame

        frames.append(Frame(finding=str(FINDINGS[k]), polarity="present", laterality="unspecified", confidence=conf))
        frame_idx = 0
        q[frame_idx] = conf
        refusal[frame_idx] = False

        if topk > 0 and att.numel() > 0:
            w = att[k]
            kk = min(int(topk), int(w.shape[0]))
            top_idx = torch.topk(w, k=kk).indices.tolist()
            citations[frame_idx] = [token_ids[i] for i in top_idx if 0 <= int(i) < len(token_ids)]
        else:
            citations[frame_idx] = []

    return Generation(frames=frames, citations=citations, q=q, refusal=refusal, text="")


def run_baselines(cfg: BaselineRunConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)

    repo_root = Path(__file__).resolve().parents[2]
    data_revision = "synthetic"
    split_manifest_path = ""
    if cfg.dataset_type == "manifest":
        data_revision, split_manifest_path = try_manifest_revision(cfg.manifest_path)
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(cfg.seed),
        config=asdict(cfg),
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    extractor = FrameExtractor()
    gt_report = (
        "There is a small left pleural effusion. "
        "A 5mm nodule is seen in the right upper lobe. "
        "No pneumothorax."
    )
    gt_frames_synth = extractor.extract_frames(gt_report)

    refusal_policy = None
    if cfg.refusal_policy_path:
        d = json.loads(Path(cfg.refusal_policy_path).read_text(encoding="utf-8"))
        refusal_policy = RefusalPolicy.from_dict(d)

    costs = ComputeUnitCosts.from_json(cfg.costs_json) if cfg.costs_json else ComputeUnitCosts()

    text_metrics_enabled = bool(cfg.compute_text_metrics)
    if text_metrics_enabled:
        try:
            _ = compute_text_metrics("a", "a")
        except MissingTextMetricDependency:
            text_metrics_enabled = False

    pcg_llm = None
    if cfg.pcg_backend == "llama2":
        pcg_llm = Llama2PCG(
            Llama2PCGConfig(
                model_path=cfg.llama2_path,
                device="cuda",
                quantization=cfg.llama2_quant,
                # Keep output cap aligned with compute accounting (b_gen).
                max_new_tokens=max(128, int(cfg.b_gen)),
                temperature=0.0,
                topk_citations=cfg.topk_citations,
            )
        )

    token_score_fn = None
    if cfg.lesionness_weights:
        lesion_model = load_lesionness_head(cfg.lesionness_weights, map_location="cpu")
        lesion_device = torch.device(cfg.lesionness_device)
        lesion_model.to(lesion_device)
        lesion_model.eval()
        token_score_fn = _score_fn_from_lesionness(lesion_model, lesion_device)

    tokenizers = {
        "provetok_no_refine": NoRefineTokenizer(),
        "provetok_lesionness": (
            FixedGridTokenizerScored(
                max_depth=6,
                token_score_fn=token_score_fn,
                token_score_level_power=float(cfg.lesionness_score_level_power),
            )
            if token_score_fn is not None
            else FixedGridTokenizer(max_depth=6)
        ),
        "fixed_grid": FixedGridTokenizer(max_depth=6),
        "slice_2d": SliceTokenizer2D(level=3),
        "slice_2p5d": SliceTokenizer2p5D(level=3, band=3),
        "roi_variance": ROIVarianceTokenizer(candidate_level=3),
        "roi_crop": ROICropTokenizer(candidate_level=3, roi_max_depth=6),
        # Note: ct2rep_noproof is added below only when real trained weights exist.
    }
    if cfg.ct2rep_strong_weights:
        if not Path(cfg.ct2rep_strong_weights).exists():
            raise FileNotFoundError(f"ct2rep_strong_weights not found: {cfg.ct2rep_strong_weights!r}")
        tokenizers["ct2rep_strong"] = FixedGridTokenizer(max_depth=6)
        # Real baseline ablation: same learned CT2RepStrong model, but no citations/refusal.
        # This keeps the generator "real" while removing proof-carrying behaviors.
        tokenizers["ct2rep_noproof"] = FixedGridTokenizer(max_depth=6)

    results: Dict[str, Dict[str, List[float]]] = {}
    for name in tokenizers.keys():
        results[name] = {
            "frame_f1": [],
            "critical_present_f1": [],
            "critical_present_recall": [],
            "iou": [],
            "dice": [],
            "hit_rate": [],
            "hit_any_intersection": [],
            "hit_lesion_coverage": [],
            "overlap_ratio_token": [],
            "overlap_ratio_lesion": [],
            "combined": [],
            "unsupported": [],
            "overclaim": [],
            "warm_time_s": [],
        }
        if text_metrics_enabled:
            results[name].update(
                {
                    "bleu": [],
                    "rouge1": [],
                    "rouge2": [],
                    "rougeL": [],
                }
            )

    def extra_flops_for(tok) -> float:
        # ROI-like baselines may require a selector/detector to pick candidate regions.
        # Count it as extra FLOPs to respect the matched-compute protocol.
        cand_level = getattr(tok, "candidate_level", None)
        if cand_level is None:
            return 0.0
        n = 2 ** int(cand_level)
        num_candidates = int(n * n * n)
        # Heuristic: selector is cheaper than full token encoding; scale by selector_ratio.
        return float(num_candidates) * float(costs.flops_per_enc_token) * float(cfg.selector_ratio)

    budgets: Dict[str, Any] = {}

    ct2rep_strong = None
    if cfg.ct2rep_strong_weights:
        ct2rep_strong = load_ct2rep_strong(cfg.ct2rep_strong_weights, map_location="cpu")
        ct2rep_strong = ct2rep_strong.to(torch.device(cfg.ct2rep_strong_device)).eval()

    if cfg.dataset_type == "manifest":
        if not cfg.manifest_path:
            raise ValueError("dataset_type=manifest requires `manifest_path`")
        dl = make_dataloader(
            {
                "dataset_type": "manifest",
                "manifest_path": cfg.manifest_path,
                "batch_size": 1,
                "num_workers": 0,
                "max_samples": cfg.n_samples,
                "resize_shape": cfg.resize_shape,
            },
            split=cfg.split,
        )
        samples_iter = enumerate(dl)
    else:
        samples_iter = ((i, None) for i in range(cfg.n_samples))

    for i, batch in samples_iter:
        sample_seed = cfg.seed + i
        if cfg.dataset_type == "manifest":
            volume = batch["volume"][0]
            lesion_masks = batch.get("lesion_masks", [{}])[0] or {}
            gt_frames = batch.get("frames", [[]])[0] or []
            gt_report_raw = str((batch.get("report_text") or [""])[0])
            volume_shape = tuple(volume.shape)
        else:
            volume, lesion_masks = create_synthetic_volume(shape=cfg.volume_shape, n_lesions=cfg.n_lesions, seed=sample_seed)
            gt_frames = gt_frames_synth
            gt_report_raw = str(gt_report)
            volume_shape = cfg.volume_shape

        gt_critical_present = [
            f for f in gt_frames
            if is_critical_finding(getattr(f, "finding", "")) and str(getattr(f, "polarity", "")) in ("present", "positive")
        ]

        if pcg_llm is not None:
            pcg_base = pcg_llm
            pcg_provetok = pcg_llm
        else:
            pcg_base = ToyPCG(
                emb_dim=cfg.emb_dim,
                topk=cfg.topk_citations,
                seed=sample_seed,
                # For baseline comparisons, disable implicit refusal inside ToyPCG.
                # Refusal is evaluated separately via `refusal_policy`.
                refusal_threshold=0.0,
                citation_strategy="attention",
                q_strategy="confidence",
            )
            pcg_provetok = ToyPCG(
                emb_dim=cfg.emb_dim,
                topk=cfg.topk_citations,
                seed=sample_seed,
                refusal_threshold=0.0,
                citation_strategy="score_interleave",
                q_strategy="support",
            )

        for name, tok in tokenizers.items():
            t0 = time.perf_counter()
            extra_flops = extra_flops_for(tok)
            budget_tokens = int(cfg.budget_tokens)
            if cfg.flops_total and cfg.flops_total > 0:
                budget_tokens = match_b_enc_for_total_flops(
                    flops_total=float(cfg.flops_total),
                    b_gen=int(cfg.b_gen),
                    n_verify=int(cfg.n_verify),
                    costs=costs,
                    flops_extra=float(extra_flops),
                    min_b_enc=1,
                    max_b_enc=4096,
                )

            tokens = tok.tokenize(volume, budget_tokens=budget_tokens, emb_dim=cfg.emb_dim, seed=sample_seed)
            if cfg.flops_total and cfg.flops_total > 0 and len(tokens) < budget_tokens:
                raise RuntimeError(
                    f"Baseline '{name}' cannot satisfy matched b_enc={budget_tokens} (got {len(tokens)}). "
                    f"Consider adjusting tokenizer params or lowering --flops-total."
                )
            tokens_eval = tokens
            if name == "provetok_lesionness":
                gen = pcg_provetok(tokens_eval)
            elif name in {"ct2rep_strong", "ct2rep_noproof"}:
                if ct2rep_strong is None:
                    raise RuntimeError("ct2rep_strong requested but model is not loaded")
                gen = _generate_ct2rep_strong(tokens, model=ct2rep_strong, topk=int(cfg.topk_citations))
            else:
                gen = pcg_base(tokens)

            # protocol ablation example: no-citation variant for fixed_grid
            gen_eval = gen
            if name == "ct2rep_noproof":
                gen_eval = apply_no_citation(gen_eval)
                # Disable refusal for non-proof baseline.
                gen_eval = Generation(frames=gen_eval.frames, citations=gen_eval.citations, q=gen_eval.q, refusal={k: False for k in range(len(gen_eval.frames))}, text=gen_eval.text)

            if refusal_policy is not None:
                gen_eval = refusal_policy.apply(gen_eval)

            issues = verify(gen_eval, tokens_eval)
            t1 = time.perf_counter()
            issue_counts = {}
            for iss in issues:
                issue_counts[iss.issue_type] = issue_counts.get(iss.issue_type, 0) + 1

            frame_metrics = compute_frame_f1(gen_eval.frames, gt_frames, threshold=0.3)
            frame_f1 = float(frame_metrics.f1)

            # Clinical correctness proxy for critical findings:
            # average over studies where GT has at least one critical present frame.
            if gt_critical_present:
                pred_critical_present = [
                    f for f in gen_eval.frames
                    if is_critical_finding(getattr(f, "finding", "")) and str(getattr(f, "polarity", "")) in ("present", "positive")
                ]
                crit = compute_frame_f1(pred_critical_present, gt_critical_present, threshold=0.3)
                critical_present_f1 = float(crit.f1)
                critical_present_recall = float(crit.recall)
            else:
                critical_present_f1 = float("nan")
                critical_present_recall = float("nan")

            g = _grounding_union(gen_eval, tokens_eval, lesion_masks, volume_shape)
            iou = float(g.get("iou_union", 0.0))
            dice = float(g.get("dice_union", 0.0))
            hit = float(g.get("hit", 0.0))
            hit_any = float(g.get("hit_any_intersection", 0.0))
            hit_lesion = float(g.get("hit_lesion_coverage", 0.0))
            overlap_token = float(g.get("overlap_ratio_token", g.get("overlap_ratio", 0.0)))
            overlap_lesion = float(g.get("overlap_ratio_lesion", 0.0))
            combined = float(cfg.nlg_weight) * float(frame_f1) + float(cfg.grounding_weight) * float(iou)

            results[name]["frame_f1"].append(float(frame_f1))
            results[name]["critical_present_f1"].append(float(critical_present_f1))
            results[name]["critical_present_recall"].append(float(critical_present_recall))
            results[name]["iou"].append(float(iou))
            results[name]["dice"].append(float(dice))
            results[name]["hit_rate"].append(float(hit))
            results[name]["hit_any_intersection"].append(float(hit_any))
            results[name]["hit_lesion_coverage"].append(float(hit_lesion))
            results[name]["overlap_ratio_token"].append(float(overlap_token))
            results[name]["overlap_ratio_lesion"].append(float(overlap_lesion))
            results[name]["combined"].append(float(combined))
            denom = max(len(gen_eval.frames), 1)
            results[name]["unsupported"].append(issue_counts.get("U1_unsupported", 0) / denom)
            results[name]["overclaim"].append(issue_counts.get("O1_overclaim", 0) / denom)
            results[name]["warm_time_s"].append(float(t1 - t0))

            if text_metrics_enabled:
                pred_text = frames_to_report(gen_eval.frames)
                try:
                    m = compute_text_metrics(pred_text, gt_report_raw)
                except MissingTextMetricDependency:
                    # Should not happen because we probe once above, but keep robust.
                    text_metrics_enabled = False
                    m = {}
                results[name]["bleu"].append(float(m.get("bleu", 0.0)))
                results[name]["rouge1"].append(float(m.get("rouge1", 0.0)))
                results[name]["rouge2"].append(float(m.get("rouge2", 0.0)))
                results[name]["rougeL"].append(float(m.get("rougeL", 0.0)))

            # Record per-baseline budget info once (deterministic given cfg + tokenizer type).
            if name not in budgets:
                budgets[name] = format_budget_report(
                    b_enc=int(budget_tokens),
                    b_gen=int(cfg.b_gen),
                    n_verify=int(cfg.n_verify),
                    costs=costs,
                    flops_extra=float(extra_flops),
                )

                if cfg.flops_total and cfg.flops_total > 0:
                    tol = max(1e-6, float(costs.flops_per_enc_token))
                    if abs(float(budgets[name]["flops_total"]) - float(cfg.flops_total)) > tol:
                        raise RuntimeError(
                            f"Baseline '{name}' failed FLOPs-matching: "
                            f"total={budgets[name]['flops_total']:.2f} vs target={cfg.flops_total:.2f} (tol={tol:.2f})."
                        )

    # Aggregate
    summary = {}
    for name, vals in results.items():
        # Use nanmean/nanstd to support conditional metrics (e.g., critical_present_*).
        summary[name] = {k: float(np.nanmean(v)) for k, v in vals.items()}
        summary[name].update({f"{k}_std": float(np.nanstd(v)) for k, v in vals.items()})

    budget_target = None
    if cfg.flops_total and cfg.flops_total > 0:
        budget_target = {"flops_total": float(cfg.flops_total), "b_gen": float(cfg.b_gen), "n_verify": float(cfg.n_verify)}
    else:
        budget_target = format_budget_report(b_enc=cfg.budget_tokens, b_gen=cfg.b_gen, n_verify=cfg.n_verify, costs=costs, flops_extra=0.0)

    return {
        "meta": meta.to_dict(),
        "config": asdict(cfg),
        "budget_target": budget_target,
        "budgets": budgets,
        "costs": costs.to_dict(),
        "summary": summary,
        "raw": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline tokenization/protocol comparisons (synthetic scaffold)")
    ap.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest"])
    ap.add_argument("--manifest", type=str, default="", help="Manifest path when dataset-type=manifest")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64], help="Resize (D,H,W) for manifest volumes")
    ap.add_argument("--pcg", type=str, default="toy", choices=["toy", "llama2"], help="PCG backend")
    ap.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])
    ap.add_argument("--smoke", action="store_true", help="Quick run")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional list of seeds for multi-seed runs.")
    ap.add_argument("--n-bootstrap", type=int, default=10_000, help="Bootstrap resamples for CI when --seeds is used.")
    ap.add_argument("--ci", type=float, default=0.95, help="CI level for bootstrap when --seeds is used.")
    ap.add_argument("--output-dir", type=str, default="./outputs/baselines")
    ap.add_argument("--refusal-policy", type=str, default="", help="Path to refusal_policy.json (optional)")
    ap.add_argument("--flops-total", type=float, default=0.0, help="Optional FLOPs-matched total budget (toy unit-cost model).")
    ap.add_argument("--b-gen", type=int, default=128, help="Decoder token budget for matched accounting (toy).")
    ap.add_argument("--n-verify", type=int, default=1, help="Verifier call count for matched accounting (toy).")
    ap.add_argument("--costs-json", type=str, default="", help="Optional JSON with ComputeUnitCosts.")
    ap.add_argument("--selector-ratio", type=float, default=0.1, help="ROI selector cost as a fraction of enc-token FLOPs.")
    ap.add_argument("--nlg-weight", type=float, default=0.5, help="Weight for frame_f1 in combined metric.")
    ap.add_argument("--grounding-weight", type=float, default=0.5, help="Weight for IoU in combined metric.")
    ap.add_argument("--lesionness-weights", type=str, default="", help="Optional path to lesionness_head.pt for provetok_lesionness.")
    ap.add_argument("--lesionness-device", type=str, default="cpu")
    ap.add_argument(
        "--lesionness-score-level-power",
        type=float,
        default=0.0,
        help="Multiply lesionness scores by (level+1)^p to favor finer cells when selecting citations.",
    )
    ap.add_argument("--ct2rep-strong-weights", type=str, default="", help="Optional path to ct2rep_strong.pt (paper-grade baseline).")
    ap.add_argument("--ct2rep-strong-device", type=str, default="cpu")
    ap.add_argument("--no-text-metrics", action="store_true", help="Disable BLEU/ROUGE computation.")
    args = ap.parse_args()

    seed_list = list(args.seeds) if args.seeds is not None else [int(args.seed)]

    def make_cfg(*, seed: int, out_dir: str) -> BaselineRunConfig:
        cfg = BaselineRunConfig(
            dataset_type=args.dataset_type,
            manifest_path=args.manifest,
            split=args.split,
            n_samples=args.n_samples,
            seed=seed,
            output_dir=out_dir,
            refusal_policy_path=args.refusal_policy,
            budget_tokens=64,
            flops_total=float(args.flops_total),
            b_gen=int(args.b_gen),
            n_verify=int(args.n_verify),
            costs_json=str(args.costs_json),
            selector_ratio=float(args.selector_ratio),
            resize_shape=tuple(args.resize_shape),
            pcg_backend=args.pcg,
            llama2_path=args.llama2_path,
            llama2_quant=args.llama2_quant,
            nlg_weight=float(args.nlg_weight),
            grounding_weight=float(args.grounding_weight),
            lesionness_weights=str(args.lesionness_weights),
            lesionness_device=str(args.lesionness_device),
            ct2rep_strong_weights=str(args.ct2rep_strong_weights),
            ct2rep_strong_device=str(args.ct2rep_strong_device),
            compute_text_metrics=(not bool(args.no_text_metrics)),
        )
        if args.smoke:
            cfg = BaselineRunConfig(
                dataset_type=args.dataset_type,
                manifest_path=args.manifest,
                split=args.split,
                n_samples=5,
                volume_shape=(32, 32, 32),
                budget_tokens=32,
                seed=seed,
                output_dir=out_dir,
                refusal_policy_path=args.refusal_policy,
                flops_total=float(args.flops_total),
                b_gen=int(args.b_gen),
                n_verify=int(args.n_verify),
                costs_json=str(args.costs_json),
                selector_ratio=float(args.selector_ratio),
                resize_shape=(32, 32, 32),
                pcg_backend=args.pcg,
                llama2_path=args.llama2_path,
                llama2_quant=args.llama2_quant,
                nlg_weight=float(args.nlg_weight),
                compute_text_metrics=(not bool(args.no_text_metrics)),
            grounding_weight=float(args.grounding_weight),
            lesionness_weights=str(args.lesionness_weights),
            lesionness_device=str(args.lesionness_device),
            lesionness_score_level_power=float(args.lesionness_score_level_power),
            ct2rep_strong_weights=str(args.ct2rep_strong_weights),
            ct2rep_strong_device=str(args.ct2rep_strong_device),
        )
        return cfg

    if len(seed_list) <= 1:
        cfg = make_cfg(seed=int(seed_list[0]), out_dir=args.output_dir)
        out_dir = make_output_dir(cfg.output_dir, "baselines")
        cfg = BaselineRunConfig(**{**asdict(cfg), "output_dir": out_dir})
        report = run_baselines(cfg)
        save_results_json(report, os.path.join(out_dir, "baselines.json"))
        print(f"Saved -> {out_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    per_seed: Dict[int, Dict[str, Any]] = {}
    for seed in seed_list:
        seed_out = os.path.join(args.output_dir, f"seed_{int(seed)}")
        os.makedirs(seed_out, exist_ok=True)
        cfg = make_cfg(seed=int(seed), out_dir=seed_out)
        rep = run_baselines(cfg)
        save_results_json(rep, os.path.join(seed_out, "baselines.json"))
        per_seed[int(seed)] = rep

    # Aggregate CI per method/metric (hierarchical bootstrap: avg over seeds per-sample, bootstrap over samples).
    any_seed = per_seed[int(seed_list[0])]
    methods = sorted(any_seed.get("raw", {}).keys())
    ci_out: Dict[str, Dict[str, Dict[str, float]]] = {}
    summary: Dict[str, Dict[str, float]] = {}

    for m in methods:
        ci_out[m] = {}
        summary[m] = {}
        metric_keys = sorted(any_seed["raw"][m].keys())
        for k in metric_keys:
            mats = []
            for s in seed_list:
                mats.append(per_seed[int(s)]["raw"][m][k])
            arr = np.asarray(mats, dtype=np.float64)  # (S,N)
            per_sample = arr.mean(axis=0)
            res = bootstrap_mean_ci(per_sample.tolist(), n_boot=int(args.n_bootstrap), seed=int(seed_list[0]), ci=float(args.ci))
            ci_out[m][k] = {"mean": float(res.mean), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}
            summary[m][k] = float(res.mean)
            summary[m][f"{k}_std"] = float(np.std(per_sample))

    repo_root = Path(__file__).resolve().parents[2]
    data_revision = "synthetic"
    split_manifest_path = ""
    if args.dataset_type == "manifest":
        data_revision, split_manifest_path = try_manifest_revision(str(args.manifest))
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(seed_list[0]),
        config={
            **asdict(make_cfg(seed=int(seed_list[0]), out_dir=args.output_dir)),
            "seeds": [int(x) for x in seed_list],
            "n_bootstrap": int(args.n_bootstrap),
            "ci": float(args.ci),
        },
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    agg = {
        "meta": meta.to_dict(),
        "seeds": [int(x) for x in seed_list],
        "n_bootstrap": int(args.n_bootstrap),
        "ci": float(args.ci),
        "budget_target": any_seed.get("budget_target"),
        "budgets": any_seed.get("budgets"),
        "costs": any_seed.get("costs"),
        "summary": summary,
        "ci_summary": ci_out,
        "per_seed_dirs": {str(int(s)): os.path.join(args.output_dir, f"seed_{int(s)}") for s in seed_list},
    }

    out_path = os.path.join(args.output_dir, "baselines_multiseed.json")
    save_results_json(agg, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
