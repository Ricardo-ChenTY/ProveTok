"""Fig X: Counterfactual Suite (proposal §7.4)

Implements a minimal, runnable counterfactual harness on synthetic data:
- Ω-permutation (permute cell_id, keep embedding)
- token-permutation (permute embedding, keep cell_id)
- citation-swap (swap C_k within report)
- evidence-drop (drop cited tokens, re-generate)
- no-citation control
- mask sanity (coarse vs refined token mask overlap)

Outputs a JSON report with paired bootstrap CIs and Holm-corrected p-values.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..data import make_dataloader
from ..bet.refine_loop import run_refine_loop
from ..bet.tokenize import encode_tokens
from ..baselines.tokenizers import FixedGridTokenizer
from ..grid.cells import cell_bounds, parse_cell_id, root_cell
from ..pcg.generator import ToyPCG
from ..models.lesionness_head import load_lesionness_head
from ..models.saliency_cnn3d import load_saliency_cnn3d
from ..verifier import verify
from ..eval.counterfactual import (
    drop_cited_tokens,
    permute_cell_ids,
    permute_embeddings,
    remove_all_citations,
    swap_citations,
    issue_rate,
)
from ..eval.metrics_frames import compute_frame_f1
from ..eval.metrics_grounding import compute_generation_grounding, compute_mask_sanity, union_lesion_masks
from ..eval.metrics_text import MissingTextMetricDependency, compute_text_metrics
from ..eval.stats import paired_bootstrap_mean_diff, holm_bonferroni
from ..eval.compute_budget import ComputeUnitCosts, format_budget_report, match_b_enc_for_total_flops
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.rules import RULE_SET_VERSION, RuleBasedVerifier, U1_CitationRelevance, U1_NoCitation
from ..verifier.taxonomy import TAXONOMY_VERSION
from ..data.frame_extractor import frames_to_report
from .utils import create_synthetic_volume, make_output_dir, save_results_json, set_seed


def _iou_union(
    *,
    gen,
    tokens,
    lesion_masks: Dict[int, np.ndarray],
    volume_shape: Tuple[int, int, int],
    positive_only: bool = True,
) -> float:
    g = compute_generation_grounding(gen, tokens, lesion_masks, volume_shape, positive_only=bool(positive_only))
    return float(g["iou_union"])

def _iou_max(
    *,
    gen,
    tokens,
    lesion_masks: Dict[int, np.ndarray],
    volume_shape: Tuple[int, int, int],
    positive_only: bool = True,
) -> float:
    g = compute_generation_grounding(gen, tokens, lesion_masks, volume_shape, positive_only=bool(positive_only))
    return float(g["iou_max"])


@dataclass
class CounterfactualConfig:
    dataset_type: str = "synthetic"  # "synthetic" or "manifest"
    manifest_path: str = ""          # required when dataset_type="manifest"
    split: str = "test"
    n_samples: int = 20
    volume_shape: Tuple[int, int, int] = (64, 64, 64)
    resize_shape: Tuple[int, int, int] = (64, 64, 64)
    n_lesions: int = 3
    budget_tokens: int = 64
    refine_steps: int = 5
    refine_max_depth: int = 4
    emb_dim: int = 32
    topk_citations: int = 3
    seed: int = 42
    n_bootstrap: int = 10_000
    output_dir: str = "./outputs/figX_counterfactual"
    flops_total: float = 0.0
    b_gen: int = 128
    n_verify: int = 1
    costs_json: str = ""
    fail_fast_matched: bool = True
    use_evidence_head: bool = True
    require_full_budget: bool = False
    lambda_uncertainty: float = 0.3
    pcg_score_bias: float = 0.0
    allocator_prefer: str = "uncertainty"
    pcg_citation_strategy: str = "attention"
    pcg_q_strategy: str = "confidence"
    lesionness_weights: str = ""
    lesionness_device: str = "cpu"
    saliency_weights: str = ""
    saliency_device: str = "cpu"
    score_to_uncertainty: bool = False
    tokenizer: str = "refine_loop"  # "refine_loop" | "fixed_grid"
    fixed_grid_max_depth: int = 6
    oracle_score: bool = False
    grounding_positive_only: bool = True
    score_level_power: float = 0.0
    compute_text_metrics: bool = True


def run_counterfactual(config: CounterfactualConfig) -> Dict[str, Any]:
    set_seed(config.seed)

    repo_root = Path(__file__).resolve().parents[2]
    data_revision = "synthetic"
    split_manifest_path = ""
    if config.dataset_type == "manifest":
        data_revision, split_manifest_path = try_manifest_revision(config.manifest_path)
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(config.seed),
        config=asdict(config),
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    scores: Dict[str, List[float]] = {k: [] for k in ("orig", "omega_perm", "token_perm", "cite_swap", "evidence_drop", "no_cite")}
    scores_iou_max: Dict[str, List[float]] = {k: [] for k in scores.keys()}
    # For counterfactual C0003 we want "unsupported" to reflect citation integrity, not
    # evidence-quality confounders (low-score / high-uncertainty / coverage). Use a
    # relevance-only verifier for the primary unsupported_rate metric, and keep the
    # full verifier rates for analysis/debug.
    unsupported_rates: Dict[str, List[float]] = {k: [] for k in scores.keys()}
    unsupported_rates_full: Dict[str, List[float]] = {k: [] for k in scores.keys()}
    overclaim_rates: Dict[str, List[float]] = {k: [] for k in scores.keys()}
    mask_sanity_iou_impr: List[float] = []
    flops_total_samples: List[float] = []
    warm_times_s: List[float] = []

    text_metrics_enabled = bool(config.compute_text_metrics)
    if text_metrics_enabled:
        try:
            _ = compute_text_metrics("a", "a")
        except MissingTextMetricDependency:
            text_metrics_enabled = False

    costs = ComputeUnitCosts.from_json(config.costs_json) if config.costs_json else ComputeUnitCosts()
    b_enc_target = int(config.budget_tokens)
    flops_total_target = float(config.flops_total) if config.flops_total and config.flops_total > 0 else 0.0
    if flops_total_target > 0:
        b_enc_target = match_b_enc_for_total_flops(
            flops_total=float(flops_total_target),
            b_gen=int(config.b_gen),
            n_verify=int(config.n_verify),
            costs=costs,
            min_b_enc=1,
            max_b_enc=4096,
        )

    if config.dataset_type == "manifest":
        if not config.manifest_path:
            raise ValueError("dataset_type=manifest requires `manifest_path`")
        dl = make_dataloader(
            {
                "dataset_type": "manifest",
                "manifest_path": config.manifest_path,
                "batch_size": 1,
                "num_workers": 0,
                "max_samples": config.n_samples,
                "resize_shape": config.resize_shape,
            },
            split=config.split,
        )
        samples_iter = enumerate(dl)
    else:
        samples_iter = ((i, None) for i in range(config.n_samples))

    frame_f1_scores: Dict[str, List[float]] = {k: [] for k in scores.keys()}
    bleu_scores: Dict[str, List[float]] = {k: [] for k in scores.keys()}
    rouge1_scores: Dict[str, List[float]] = {k: [] for k in scores.keys()}
    rouge2_scores: Dict[str, List[float]] = {k: [] for k in scores.keys()}
    rougeL_scores: Dict[str, List[float]] = {k: [] for k in scores.keys()}

    relevance_verifier = RuleBasedVerifier().add_rule(U1_NoCitation()).add_rule(
        U1_CitationRelevance(
            min_recall_at_k=1.0,
            min_attention_mass=0.1,
            score_bias=float(config.pcg_score_bias),
        )
    )

    token_score_fn = None
    lesion_device = str(config.lesionness_device or "cpu")
    if config.lesionness_weights:
        lesionness = load_lesionness_head(config.lesionness_weights, map_location=lesion_device)
        lesionness.eval()

        def token_score_fn(emb: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                probs = lesionness.predict_proba(emb.to(device=lesion_device))
                return probs.to(device="cpu")

    saliency_model = None
    saliency_device = str(config.saliency_device or "cpu")
    if config.saliency_weights:
        saliency_model = load_saliency_cnn3d(config.saliency_weights, map_location=saliency_device)
        saliency_model.to(device=saliency_device)
        saliency_model.eval()

    def _preprocess_volume_for_saliency(vol: torch.Tensor) -> torch.Tensor:
        # Keep consistent with train_saliency_cnn3d.py defaults.
        v = vol.float()
        v = v.clamp(min=-1000.0, max=1000.0)
        v = v / 1000.0
        return v

    def _record_text_and_frame_metrics(*, key: str, gen_x, gt_frames_x, ref_report_text: str) -> None:
        frame_f1_scores[key].append(float(compute_frame_f1(gen_x.frames, gt_frames_x, threshold=0.3).f1))
        if not text_metrics_enabled:
            return
        pred_text = frames_to_report(gen_x.frames)
        try:
            m = compute_text_metrics(pred_text, ref_report_text)
        except MissingTextMetricDependency:
            m = {}
        bleu_scores[key].append(float(m.get("bleu", 0.0)))
        rouge1_scores[key].append(float(m.get("rouge1", 0.0)))
        rouge2_scores[key].append(float(m.get("rouge2", 0.0)))
        rougeL_scores[key].append(float(m.get("rougeL", 0.0)))

    for i, batch in samples_iter:
        sample_seed = config.seed + i

        if config.dataset_type == "manifest":
            volume = batch["volume"][0]  # (D,H,W)
            lesion_masks = batch.get("lesion_masks", [{}])[0] or {}
            volume_shape = tuple(volume.shape)
            gt_frames = batch.get("frames", [[]])[0] or []
            ref_report_text = str((batch.get("report_text") or [""])[0])
        else:
            volume, lesion_masks = create_synthetic_volume(
                shape=config.volume_shape,
                n_lesions=config.n_lesions,
                seed=sample_seed,
            )
            volume_shape = config.volume_shape
            gt_frames = None
            ref_report_text = ""

        # Disable refusal in counterfactual suite: C0003 is about stress-testing
        # citations/Ω integrity (refusal calibration is covered by C0005).
        pcg = ToyPCG(
            emb_dim=config.emb_dim,
            topk=config.topk_citations,
            seed=sample_seed,
            score_bias=float(config.pcg_score_bias),
            refusal_threshold=0.0,
            citation_strategy=str(config.pcg_citation_strategy),
            q_strategy=str(config.pcg_q_strategy),
        )

        t0 = time.perf_counter()
        if str(config.tokenizer) == "fixed_grid":
            toks = FixedGridTokenizer(max_depth=int(config.fixed_grid_max_depth)).tokenize(
                volume,
                budget_tokens=int(b_enc_target),
                emb_dim=int(config.emb_dim),
                seed=int(sample_seed),
            )
            if saliency_model is not None and toks:
                with torch.no_grad():
                    x = _preprocess_volume_for_saliency(volume)
                    if x.dim() == 3:
                        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
                    prob = saliency_model.predict_proba(x.to(device=saliency_device)).detach().cpu()[0, 0]  # (D,H,W)

                score_list = []
                for t in toks:
                    cell = parse_cell_id(str(t.cell_id))
                    if cell is None:
                        score_list.append(0.0)
                        continue
                    sl = cell_bounds(cell, shape=volume_shape)
                    score_list.append(float(prob[sl[0], sl[1], sl[2]].mean().item()))

                toks = [
                    t.__class__(
                        token_id=int(t.token_id),
                        cell_id=str(t.cell_id),
                        level=int(t.level),
                        embedding=t.embedding,
                        score=float(s),
                        uncertainty=(1.0 - float(s)) if bool(config.score_to_uncertainty) else float(t.uncertainty),
                    )
                    for t, s in zip(toks, score_list)
                ]
            elif bool(config.oracle_score):
                lesion_union = union_lesion_masks(lesion_masks, volume_shape)
                scores_oracle = []
                if lesion_union.sum() <= 0:
                    scores_oracle = [0.0 for _ in toks]
                else:
                    for t in toks:
                        cell = parse_cell_id(str(t.cell_id))
                        if cell is None:
                            scores_oracle.append(0.0)
                            continue
                        sl = cell_bounds(cell, shape=volume_shape)
                        inter = float(lesion_union[sl[0], sl[1], sl[2]].sum())
                        token_vol = float(
                            (sl[0].stop - sl[0].start)
                            * (sl[1].stop - sl[1].start)
                            * (sl[2].stop - sl[2].start)
                        )
                        scores_oracle.append(float(inter / token_vol) if token_vol > 0 else 0.0)

                toks = [
                    t.__class__(
                        token_id=int(t.token_id),
                        cell_id=str(t.cell_id),
                        level=int(t.level),
                        embedding=t.embedding,
                        score=float(s),
                        uncertainty=(1.0 - float(s)) if bool(config.score_to_uncertainty) else float(t.uncertainty),
                    )
                    for t, s in zip(toks, scores_oracle)
                ]
            elif token_score_fn is not None and toks:
                emb = torch.stack([t.embedding for t in toks], dim=0)
                scores_t = token_score_fn(emb)
                score_list = scores_t.detach().cpu().flatten().tolist()
                toks = [
                    t.__class__(
                        token_id=int(t.token_id),
                        cell_id=str(t.cell_id),
                        level=int(t.level),
                        embedding=t.embedding,
                        score=float(s),
                        uncertainty=(1.0 - float(s)) if bool(config.score_to_uncertainty) else float(t.uncertainty),
                    )
                    for t, s in zip(toks, score_list)
                ]

            gen = pcg(toks)
            issues = verify(gen, toks)
            result_tokens = toks
            result_gen = gen
            result_issues = issues
        else:
            result = run_refine_loop(
                volume=volume,
                budget_tokens=int(b_enc_target),
                steps=config.refine_steps,
                generator_fn=lambda toks: pcg(toks),
                verifier_fn=lambda gen, toks: verify(gen, toks),
                emb_dim=config.emb_dim,
                seed=sample_seed,
                require_full_budget=bool(config.require_full_budget),
                use_evidence_head=bool(config.use_evidence_head),
                lambda_uncertainty=float(config.lambda_uncertainty),
                max_depth=int(config.refine_max_depth),
                allocator_prefer=str(config.allocator_prefer),
                token_score_fn=token_score_fn,
                score_to_uncertainty=bool(config.score_to_uncertainty),
                score_level_power=float(config.score_level_power),
            )
            result_tokens = result.tokens
            result_gen = result.gen
            result_issues = verify(result.gen, result.tokens)
        t1 = time.perf_counter()
        warm_times_s.append(float(t1 - t0))

        tokens = result_tokens
        gen = result_gen
        issues = result_issues
        issues_rel = relevance_verifier.verify(gen, tokens)

        # Coarse vs refined mask sanity
        coarse_tokens = encode_tokens(volume, [root_cell()], emb_dim=config.emb_dim, seed=sample_seed)
        per_lesion = []
        for _, mask in lesion_masks.items():
            per_lesion.append(
                compute_mask_sanity(coarse_tokens, tokens, mask, volume_shape)["iou_improvement"]
            )
        mask_sanity_iou_impr.append(float(np.mean(per_lesion)) if per_lesion else 0.0)

        # Baseline
        gt_frames_eval = gt_frames if gt_frames is not None else list(gen.frames)
        ref_report_eval = ref_report_text if ref_report_text else frames_to_report(gt_frames_eval)
        scores["orig"].append(
            _iou_union(
                gen=gen,
                tokens=tokens,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )
        scores_iou_max["orig"].append(
            _iou_max(
                gen=gen,
                tokens=tokens,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )

        r = issue_rate(issues_rel, num_frames=len(gen.frames))
        unsupported_rates["orig"].append(float(r.get("U1_unsupported", 0.0)))

        r_full = issue_rate(issues, num_frames=len(gen.frames))
        unsupported_rates_full["orig"].append(float(r_full.get("U1_unsupported", 0.0)))
        overclaim_rates["orig"].append(float(r_full.get("O1_overclaim", 0.0)))
        _record_text_and_frame_metrics(key="orig", gen_x=gen, gt_frames_x=gt_frames_eval, ref_report_text=ref_report_eval)

        # Ω-permutation (keep citations fixed)
        toks_omega = permute_cell_ids(tokens, seed=sample_seed + 1000)
        issues_omega = verify(gen, toks_omega)
        issues_omega_rel = relevance_verifier.verify(gen, toks_omega)
        scores["omega_perm"].append(
            _iou_union(
                gen=gen,
                tokens=toks_omega,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )
        scores_iou_max["omega_perm"].append(
            _iou_max(
                gen=gen,
                tokens=toks_omega,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )

        r = issue_rate(issues_omega_rel, num_frames=len(gen.frames))
        unsupported_rates["omega_perm"].append(float(r.get("U1_unsupported", 0.0)))

        r_full = issue_rate(issues_omega, num_frames=len(gen.frames))
        unsupported_rates_full["omega_perm"].append(float(r_full.get("U1_unsupported", 0.0)))
        overclaim_rates["omega_perm"].append(float(r_full.get("O1_overclaim", 0.0)))
        _record_text_and_frame_metrics(key="omega_perm", gen_x=gen, gt_frames_x=gt_frames_eval, ref_report_text=ref_report_eval)

        # token-permutation (re-generate)
        toks_tp = permute_embeddings(tokens, seed=sample_seed + 2000)
        gen_tp = pcg(toks_tp)
        issues_tp = verify(gen_tp, toks_tp)
        issues_tp_rel = relevance_verifier.verify(gen_tp, toks_tp)
        scores["token_perm"].append(
            _iou_union(
                gen=gen_tp,
                tokens=toks_tp,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )
        scores_iou_max["token_perm"].append(
            _iou_max(
                gen=gen_tp,
                tokens=toks_tp,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )

        r = issue_rate(issues_tp_rel, num_frames=len(gen_tp.frames))
        unsupported_rates["token_perm"].append(float(r.get("U1_unsupported", 0.0)))

        r_full = issue_rate(issues_tp, num_frames=len(gen_tp.frames))
        unsupported_rates_full["token_perm"].append(float(r_full.get("U1_unsupported", 0.0)))
        overclaim_rates["token_perm"].append(float(r_full.get("O1_overclaim", 0.0)))
        _record_text_and_frame_metrics(key="token_perm", gen_x=gen_tp, gt_frames_x=gt_frames_eval, ref_report_text=ref_report_eval)

        # citation-swap (keep tokens)
        gen_sw = swap_citations(gen, seed=sample_seed + 3000)
        issues_sw = verify(gen_sw, tokens)
        issues_sw_rel = relevance_verifier.verify(gen_sw, tokens)
        scores["cite_swap"].append(
            _iou_union(
                gen=gen_sw,
                tokens=tokens,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )
        scores_iou_max["cite_swap"].append(
            _iou_max(
                gen=gen_sw,
                tokens=tokens,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )

        r = issue_rate(issues_sw_rel, num_frames=len(gen_sw.frames))
        unsupported_rates["cite_swap"].append(float(r.get("U1_unsupported", 0.0)))

        r_full = issue_rate(issues_sw, num_frames=len(gen_sw.frames))
        unsupported_rates_full["cite_swap"].append(float(r_full.get("U1_unsupported", 0.0)))
        overclaim_rates["cite_swap"].append(float(r_full.get("O1_overclaim", 0.0)))
        _record_text_and_frame_metrics(key="cite_swap", gen_x=gen_sw, gt_frames_x=gt_frames_eval, ref_report_text=ref_report_eval)

        # evidence-drop (drop cited tokens, re-generate)
        toks_drop = drop_cited_tokens(tokens, gen)
        gen_drop = pcg(toks_drop)
        issues_drop = verify(gen_drop, toks_drop)
        issues_drop_rel = relevance_verifier.verify(gen_drop, toks_drop)
        scores["evidence_drop"].append(
            _iou_union(
                gen=gen_drop,
                tokens=toks_drop,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )
        scores_iou_max["evidence_drop"].append(
            _iou_max(
                gen=gen_drop,
                tokens=toks_drop,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )

        r = issue_rate(issues_drop_rel, num_frames=len(gen_drop.frames))
        unsupported_rates["evidence_drop"].append(float(r.get("U1_unsupported", 0.0)))

        r_full = issue_rate(issues_drop, num_frames=len(gen_drop.frames))
        unsupported_rates_full["evidence_drop"].append(float(r_full.get("U1_unsupported", 0.0)))
        overclaim_rates["evidence_drop"].append(float(r_full.get("O1_overclaim", 0.0)))
        _record_text_and_frame_metrics(key="evidence_drop", gen_x=gen_drop, gt_frames_x=gt_frames_eval, ref_report_text=ref_report_eval)

        # no-citation control (keep frames/tokens, drop citations)
        gen_nc = remove_all_citations(gen)
        issues_nc = verify(gen_nc, tokens)
        issues_nc_rel = relevance_verifier.verify(gen_nc, tokens)
        scores["no_cite"].append(
            _iou_union(
                gen=gen_nc,
                tokens=tokens,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )
        scores_iou_max["no_cite"].append(
            _iou_max(
                gen=gen_nc,
                tokens=tokens,
                lesion_masks=lesion_masks,
                volume_shape=volume_shape,
                positive_only=bool(config.grounding_positive_only),
            )
        )

        r = issue_rate(issues_nc_rel, num_frames=len(gen_nc.frames))
        unsupported_rates["no_cite"].append(float(r.get("U1_unsupported", 0.0)))

        r_full = issue_rate(issues_nc, num_frames=len(gen_nc.frames))
        unsupported_rates_full["no_cite"].append(float(r_full.get("U1_unsupported", 0.0)))
        overclaim_rates["no_cite"].append(float(r_full.get("O1_overclaim", 0.0)))
        _record_text_and_frame_metrics(key="no_cite", gen_x=gen_nc, gt_frames_x=gt_frames_eval, ref_report_text=ref_report_eval)

        bud_realized = format_budget_report(
            b_enc=int(len(tokens)),
            b_gen=int(config.b_gen),
            n_verify=int(config.n_verify),
            costs=costs,
            flops_extra=0.0,
        )
        flops_total_samples.append(float(bud_realized["flops_total"]))
        if flops_total_target > 0 and config.fail_fast_matched:
            # Matching is defined on the configured budget cap (b_enc_target). Realized token
            # count may be lower due to early stopping, which should be recorded but not fail.
            bud_target = format_budget_report(
                b_enc=int(b_enc_target),
                b_gen=int(config.b_gen),
                n_verify=int(config.n_verify),
                costs=costs,
                flops_extra=0.0,
            )
            tol = max(1e-6, 8.0 * float(costs.flops_per_enc_token))
            if abs(float(bud_target["flops_total"]) - float(flops_total_target)) > tol:
                raise RuntimeError(
                    "FLOPs-matching failed (budget cap): "
                    f"total={float(bud_target['flops_total']):.2f} vs target={float(flops_total_target):.2f} "
                    f"(tol={tol:.2f}). Consider adjusting --b-gen/--n-verify or budgets."
                )

    cold_times = warm_times_s[: min(3, len(warm_times_s))]
    warm_mean_s = float(np.mean(warm_times_s)) if warm_times_s else 0.0
    warm_p95_s = float(np.quantile(np.asarray(warm_times_s, dtype=np.float64), 0.95)) if warm_times_s else 0.0
    cold_mean_s = float(np.mean(cold_times)) if cold_times else 0.0
    cold_p95_s = float(np.quantile(np.asarray(cold_times, dtype=np.float64), 0.95)) if cold_times else 0.0

    # Paired bootstrap: orig - cf (degradation => positive mean_diff)
    cf_keys = ["omega_perm", "token_perm", "cite_swap", "evidence_drop", "no_cite"]
    grounding_boot = {}
    pvals = []
    for k in cf_keys:
        res = paired_bootstrap_mean_diff(scores["orig"], scores[k], n_boot=config.n_bootstrap, seed=config.seed)
        grounding_boot[k] = asdict(res)
        pvals.append(res.p_value)

    pvals_holm = holm_bonferroni(pvals)
    for k, adj in zip(cf_keys, pvals_holm):
        grounding_boot[k]["p_value_holm"] = float(adj)

    grounding_boot_max = {}
    pvals_max = []
    for k in cf_keys:
        res = paired_bootstrap_mean_diff(scores_iou_max["orig"], scores_iou_max[k], n_boot=config.n_bootstrap, seed=config.seed)
        grounding_boot_max[k] = asdict(res)
        pvals_max.append(res.p_value)

    pvals_max_holm = holm_bonferroni(pvals_max)
    for k, adj in zip(cf_keys, pvals_max_holm):
        grounding_boot_max[k]["p_value_holm"] = float(adj)

    unsupported_boot = {}
    pvals_u = []
    for k in cf_keys:
        # For issue rates we want increases: cf - orig
        res = paired_bootstrap_mean_diff(unsupported_rates[k], unsupported_rates["orig"], n_boot=config.n_bootstrap, seed=config.seed)
        unsupported_boot[k] = asdict(res)
        pvals_u.append(res.p_value)

    pvals_u_holm = holm_bonferroni(pvals_u)
    for k, adj in zip(cf_keys, pvals_u_holm):
        unsupported_boot[k]["p_value_holm"] = float(adj)

    overclaim_boot = {}
    pvals_o = []
    for k in cf_keys:
        res = paired_bootstrap_mean_diff(overclaim_rates[k], overclaim_rates["orig"], n_boot=config.n_bootstrap, seed=config.seed)
        overclaim_boot[k] = asdict(res)
        pvals_o.append(res.p_value)

    pvals_o_holm = holm_bonferroni(pvals_o)
    for k, adj in zip(cf_keys, pvals_o_holm):
        overclaim_boot[k]["p_value_holm"] = float(adj)

    # Optional extra paired bootstrap: correctness/text metrics.
    frame_boot = {}
    pvals_f = []
    for k in cf_keys:
        res = paired_bootstrap_mean_diff(frame_f1_scores["orig"], frame_f1_scores[k], n_boot=config.n_bootstrap, seed=config.seed)
        frame_boot[k] = asdict(res)
        pvals_f.append(res.p_value)
    pvals_f_holm = holm_bonferroni(pvals_f)
    for k, adj in zip(cf_keys, pvals_f_holm):
        frame_boot[k]["p_value_holm"] = float(adj)

    bleu_boot = {}
    rougeL_boot = {}
    if text_metrics_enabled:
        pvals_bleu = []
        pvals_rl = []
        for k in cf_keys:
            res_bleu = paired_bootstrap_mean_diff(bleu_scores["orig"], bleu_scores[k], n_boot=config.n_bootstrap, seed=config.seed)
            bleu_boot[k] = asdict(res_bleu)
            pvals_bleu.append(res_bleu.p_value)
            res_rl = paired_bootstrap_mean_diff(rougeL_scores["orig"], rougeL_scores[k], n_boot=config.n_bootstrap, seed=config.seed)
            rougeL_boot[k] = asdict(res_rl)
            pvals_rl.append(res_rl.p_value)
        for k, adj in zip(cf_keys, holm_bonferroni(pvals_bleu)):
            bleu_boot[k]["p_value_holm"] = float(adj)
        for k, adj in zip(cf_keys, holm_bonferroni(pvals_rl)):
            rougeL_boot[k]["p_value_holm"] = float(adj)

    return {
        "meta": meta.to_dict(),
        "config": asdict(config),
        "budget_target": (
            {"flops_total": float(flops_total_target), "b_gen": float(config.b_gen), "n_verify": float(config.n_verify)}
            if flops_total_target > 0
            else format_budget_report(b_enc=int(b_enc_target), b_gen=int(config.b_gen), n_verify=int(config.n_verify), costs=costs, flops_extra=0.0)
        ),
        "costs": costs.to_dict(),
        "latency": {
            "n": int(len(warm_times_s)),
            "warm_mean_s": warm_mean_s,
            "warm_p95_s": warm_p95_s,
            "cold_mean_s": cold_mean_s,
            "cold_p95_s": cold_p95_s,
        },
        "compute": {
            "flops_total_per_sample": flops_total_samples,
            "flops_total_mean": float(np.mean(flops_total_samples)) if flops_total_samples else 0.0,
            "flops_total_std": float(np.std(flops_total_samples)) if flops_total_samples else 0.0,
        },
        "scores": scores,
        "scores_iou_max": scores_iou_max,
        "frame_f1_scores": frame_f1_scores,
        "text_metrics": {
            "enabled": bool(text_metrics_enabled),
            "bleu": bleu_scores,
            "rouge1": rouge1_scores,
            "rouge2": rouge2_scores,
            "rougeL": rougeL_scores,
        },
        "unsupported_rates": unsupported_rates,
        "unsupported_rates_full": unsupported_rates_full,
        "overclaim_rates": overclaim_rates,
        "mask_sanity": {
            "iou_improvement_per_sample": mask_sanity_iou_impr,
            "mean_iou_improvement": float(np.mean(mask_sanity_iou_impr)) if mask_sanity_iou_impr else 0.0,
        },
        "paired_bootstrap": {
            "grounding_iou_union_orig_minus_cf": grounding_boot,
            "grounding_iou_max_orig_minus_cf": grounding_boot_max,
            "unsupported_rate_cf_minus_orig": unsupported_boot,
            "overclaim_rate_cf_minus_orig": overclaim_boot,
            "frame_f1_orig_minus_cf": frame_boot,
            "bleu_orig_minus_cf": bleu_boot,
            "rougeL_orig_minus_cf": rougeL_boot,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run counterfactual suite (synthetic scaffold)")
    ap.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest"])
    ap.add_argument("--manifest", type=str, default="", help="Manifest path when dataset-type=manifest")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64], help="Resize (D,H,W) for manifest volumes")
    ap.add_argument("--smoke", action="store_true", help="Small config for quick sanity")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--no-text-metrics", action="store_true", help="Disable BLEU/ROUGE computation.")
    ap.add_argument("--flops-total", type=float, default=0.0, help="Optional FLOPs-matched total budget (toy unit-cost model).")
    ap.add_argument("--b-gen", type=int, default=128, help="Decoder token budget for matched accounting (toy unit-cost model).")
    ap.add_argument("--n-verify", type=int, default=1, help="Verifier call count for matched accounting (toy unit-cost model).")
    ap.add_argument("--topk-citations", type=int, default=3, help="Citations per frame (top-k).")
    ap.add_argument("--costs-json", type=str, default="", help="Optional JSON with ComputeUnitCosts.")
    ap.add_argument("--no-evidence-head", action="store_true", help="Disable EvidenceHead and use the simple allocator.")
    ap.add_argument("--require-full-budget", action="store_true", help="Avoid early-stop on no-issues/epsilon; spend budget when possible.")
    ap.add_argument("--refine-max-depth", type=int, default=4, help="Max cell depth for refine_loop tokenizer.")
    ap.add_argument("--lambda-uncertainty", type=float, default=0.3, help="Uncertainty weight in EvidenceHead Δ(c).")
    ap.add_argument("--pcg-score-bias", type=float, default=0.0, help="ToyPCG attention bias on token.score (keep verifier U1.4 aligned).")
    ap.add_argument(
        "--pcg-citation-strategy",
        type=str,
        default="attention",
        choices=["attention", "score", "score_interleave", "attn_score"],
        help="ToyPCG citation strategy.",
    )
    ap.add_argument(
        "--pcg-q-strategy",
        type=str,
        default="confidence",
        choices=["confidence", "support"],
        help="ToyPCG q_k strategy.",
    )
    ap.add_argument("--lesionness-weights", type=str, default="", help="Optional lesionness head checkpoint (.pt).")
    ap.add_argument("--lesionness-device", type=str, default="cpu", help="Device for lesionness head inference (cpu/cuda).")
    ap.add_argument("--saliency-weights", type=str, default="", help="Optional saliency_cnn3d.pt for token score (fixed_grid only).")
    ap.add_argument("--saliency-device", type=str, default="cpu", help="Device for saliency model inference (cpu/cuda).")
    ap.add_argument(
        "--score-to-uncertainty",
        action="store_true",
        help="When lesionness scores are used, also set token.uncertainty = 1 - score (for verifier rules).",
    )
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="refine_loop",
        choices=["refine_loop", "fixed_grid"],
        help="Evidence tokenizer to use for the counterfactual suite.",
    )
    ap.add_argument("--fixed-grid-max-depth", type=int, default=6, help="Max depth for fixed_grid tokenizer.")
    ap.add_argument(
        "--oracle-score",
        action="store_true",
        help="Use oracle token scores from GT lesion masks (intersection/token_volume). Only valid when masks exist.",
    )
    ap.add_argument(
        "--grounding-all-frames",
        action="store_true",
        help="Compute grounding using union citations over all frames (not just positive frames).",
    )
    ap.add_argument(
        "--score-level-power",
        type=float,
        default=0.0,
        help="Optional multiplicative bias on token.score by token level: score *= (1 + level) ** p.",
    )
    ap.add_argument("--allocator-prefer", type=str, default="uncertainty", choices=["uncertainty", "score"], help="Allocator preference when --no-evidence-head is set.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="./outputs/figX_counterfactual")
    args = ap.parse_args()

    cfg = CounterfactualConfig(
        dataset_type=args.dataset_type,
        manifest_path=args.manifest,
        split=args.split,
        n_samples=args.n_samples,
        resize_shape=tuple(args.resize_shape),
        topk_citations=int(args.topk_citations),
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir,
        flops_total=float(args.flops_total),
        b_gen=int(args.b_gen),
        n_verify=int(args.n_verify),
        costs_json=str(args.costs_json),
        use_evidence_head=(not bool(args.no_evidence_head)),
        require_full_budget=bool(args.require_full_budget),
        lambda_uncertainty=float(args.lambda_uncertainty),
        pcg_score_bias=float(args.pcg_score_bias),
        allocator_prefer=str(args.allocator_prefer),
        pcg_citation_strategy=str(args.pcg_citation_strategy),
        pcg_q_strategy=str(args.pcg_q_strategy),
        lesionness_weights=str(args.lesionness_weights),
        lesionness_device=str(args.lesionness_device),
        saliency_weights=str(args.saliency_weights),
        saliency_device=str(args.saliency_device),
        score_to_uncertainty=bool(args.score_to_uncertainty),
        tokenizer=str(args.tokenizer),
        fixed_grid_max_depth=int(args.fixed_grid_max_depth),
        oracle_score=bool(args.oracle_score),
        grounding_positive_only=(not bool(args.grounding_all_frames)),
        score_level_power=float(args.score_level_power),
        refine_max_depth=int(args.refine_max_depth),
        compute_text_metrics=(not bool(args.no_text_metrics)),
    )
    if args.smoke:
        # Smoke should be fast by default, but still respect the user's requested
        # sample/bootstrap sizes (clamped to safe upper bounds).
        n_samples_smoke = min(int(args.n_samples), 20)
        n_boot_smoke = min(int(args.n_bootstrap), 2_000)
        cfg = CounterfactualConfig(
            dataset_type=args.dataset_type,
            manifest_path=args.manifest,
            split=args.split,
            n_samples=n_samples_smoke,
            volume_shape=(32, 32, 32),
            resize_shape=(32, 32, 32),
            n_lesions=3,
            budget_tokens=32,
            refine_steps=3,
            refine_max_depth=min(6, int(args.refine_max_depth)),
            emb_dim=32,
            topk_citations=max(1, min(20, int(args.topk_citations))),
            seed=args.seed,
            n_bootstrap=n_boot_smoke,
            output_dir=args.output_dir,
            flops_total=float(args.flops_total),
            b_gen=int(args.b_gen),
            n_verify=int(args.n_verify),
            costs_json=str(args.costs_json),
            use_evidence_head=(not bool(args.no_evidence_head)),
            require_full_budget=bool(args.require_full_budget),
            lambda_uncertainty=float(args.lambda_uncertainty),
            pcg_score_bias=float(args.pcg_score_bias),
            allocator_prefer=str(args.allocator_prefer),
            pcg_citation_strategy=str(args.pcg_citation_strategy),
            pcg_q_strategy=str(args.pcg_q_strategy),
            lesionness_weights=str(args.lesionness_weights),
            lesionness_device=str(args.lesionness_device),
            saliency_weights=str(args.saliency_weights),
            saliency_device=str(args.saliency_device),
            score_to_uncertainty=bool(args.score_to_uncertainty),
            tokenizer=str(args.tokenizer),
            fixed_grid_max_depth=int(args.fixed_grid_max_depth),
            oracle_score=bool(args.oracle_score),
            grounding_positive_only=(not bool(args.grounding_all_frames)),
            score_level_power=float(args.score_level_power),
            compute_text_metrics=(not bool(args.no_text_metrics)),
        )

    out_dir = make_output_dir(cfg.output_dir, "figX_counterfactual")
    cfg = CounterfactualConfig(**{**asdict(cfg), "output_dir": out_dir})

    report = run_counterfactual(cfg)
    save_results_json(report, os.path.join(out_dir, "figX_counterfactual.json"))
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
