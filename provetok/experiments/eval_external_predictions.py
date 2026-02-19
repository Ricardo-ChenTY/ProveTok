"""Evaluate external text-only baselines on ProveTok metrics (pp.md §6.3).

This runner consumes a manifest dataset (volumes + optional masks + reference text)
plus a JSONL of external predictions, and produces:
- Standard NLG text metrics (BLEU/ROUGE/METEOR)
- Proof-centric reliability metrics (R1–R4 + WeightedIssue) using a post-hoc
  citation wrapper for fairness (pp.md §6.5)
- Evidence grounding metrics when masks are available in the manifest

Expected predictions JSONL format (one per sample per method):
  {"sample_id": "<scan_hash>", "method": "diallama", "pred_text": "..."}

Notes:
- For proof/grounding metrics, we tokenize the volume deterministically and attach
  citations to the extracted finding frames via `attach_posthoc_citations`.
- We do not run the external baselines inside this repo; this is an adapter.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..baselines import BETAlgTokenizer, FixedGridTokenizer
from ..data import make_dataloader
from ..data.frame_extractor import FrameExtractor
from ..eval.metrics_grounding import compute_generation_grounding
from ..eval.metrics_proof import ProofWeights, attach_posthoc_citations, compute_proof_metrics
from ..eval.metrics_text import MissingTextMetricDependency, TextMetricConfig, compute_text_metrics
from ..types import Generation
from ..verifier import verify
from .utils import make_output_dir, save_results_json, set_seed


@dataclass
class EvalExternalConfig:
    manifest_path: str
    split: str = "test"
    resize_shape: Tuple[int, int, int] = (64, 64, 64)
    max_samples: int = 0

    # Tokenization for post-hoc citations
    tokenizer: str = "fixed_grid"  # fixed_grid|bet_alg
    max_depth: int = 6
    budget_tokens: int = 256
    emb_dim: int = 32

    # Proof metrics
    l_min: int = 2
    k_max: int = 8
    proof_w1: float = 3.0
    proof_w2: float = 2.0
    proof_w3: float = 2.0
    proof_w4: float = 2.0

    # Metrics
    compute_text_metrics: bool = True

    # IO
    pred_jsonl: str = ""
    methods: List[str] = field(default_factory=list)
    seed: int = 42
    output_dir: str = "./outputs/eval_external_predictions"


def _load_predictions(path: str) -> Dict[Tuple[str, str], str]:
    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(str(p))

    out: Dict[Tuple[str, str], str] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            sid = str(d.get("sample_id") or d.get("scan_hash") or "").strip()
            method = str(d.get("method") or "").strip()
            pred = str(d.get("pred_text") or d.get("pred") or "")
            if not sid or not method:
                continue
            key = (sid, method)
            # Keep first occurrence to avoid accidental duplicates.
            if key not in out:
                out[key] = pred
    if not out:
        raise ValueError(f"No usable predictions found in {p}")
    return out


def _mean(xs: List[float]) -> float:
    return float(np.nanmean(np.asarray(xs, dtype=np.float64))) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate external text-only baselines with ProveTok metrics (pp.md §6.3).")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64], help="Resize (D,H,W) for manifest volumes")
    ap.add_argument("--max-samples", type=int, default=0)

    ap.add_argument("--pred-jsonl", type=str, required=True, help="Predictions JSONL: {sample_id, method, pred_text}")
    ap.add_argument("--methods", type=str, nargs="+", default=[], help="Optional subset of methods to evaluate")

    ap.add_argument("--tokenizer", type=str, default="fixed_grid", choices=["fixed_grid", "bet_alg"])
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--budget-tokens", type=int, default=256)
    ap.add_argument("--emb-dim", type=int, default=32)

    ap.add_argument("--l-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)

    ap.add_argument("--proof-w1", type=float, default=3.0)
    ap.add_argument("--proof-w2", type=float, default=2.0)
    ap.add_argument("--proof-w3", type=float, default=2.0)
    ap.add_argument("--proof-w4", type=float, default=2.0)

    ap.add_argument("--no-text-metrics", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="./outputs/eval_external_predictions")

    args = ap.parse_args()

    cfg = EvalExternalConfig(
        manifest_path=str(args.manifest),
        split=str(args.split),
        resize_shape=tuple(int(x) for x in args.resize_shape),
        max_samples=int(args.max_samples),
        pred_jsonl=str(args.pred_jsonl),
        methods=[str(m) for m in (args.methods or [])],
        tokenizer=str(args.tokenizer),
        max_depth=int(args.max_depth),
        budget_tokens=int(args.budget_tokens),
        emb_dim=int(args.emb_dim),
        l_min=int(args.l_min),
        k_max=int(args.k_max),
        proof_w1=float(args.proof_w1),
        proof_w2=float(args.proof_w2),
        proof_w3=float(args.proof_w3),
        proof_w4=float(args.proof_w4),
        compute_text_metrics=not bool(args.no_text_metrics),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
    )

    set_seed(int(cfg.seed))
    out_dir = Path(make_output_dir(str(cfg.output_dir), "eval_external_predictions"))
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = _load_predictions(cfg.pred_jsonl)

    # Discover available methods if not provided.
    all_methods = sorted({m for (_, m) in preds.keys()})
    methods = cfg.methods or all_methods
    missing = [m for m in methods if m not in all_methods]
    if missing:
        raise SystemExit(f"Unknown --methods {missing}. Available: {all_methods}")

    # Build tokenizer
    if cfg.tokenizer == "fixed_grid":
        tok = FixedGridTokenizer(max_depth=int(cfg.max_depth))
    else:
        tok = BETAlgTokenizer(max_depth=int(cfg.max_depth), min_cover_tokens=min(64, int(cfg.budget_tokens)))

    extractor = FrameExtractor()

    weights = ProofWeights(w1_r1=float(cfg.proof_w1), w2_r2=float(cfg.proof_w2), w3_r3=float(cfg.proof_w3), w4_r4=float(cfg.proof_w4))

    text_metrics_enabled = bool(cfg.compute_text_metrics)
    if text_metrics_enabled:
        try:
            _ = compute_text_metrics("a", "a", cfg=TextMetricConfig(compute_meteor=True))
        except MissingTextMetricDependency:
            text_metrics_enabled = False

    dl = make_dataloader(
        {
            "dataset_type": "manifest",
            "manifest_path": str(cfg.manifest_path),
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": int(cfg.max_samples),
            "resize_shape": tuple(int(x) for x in cfg.resize_shape),
        },
        split=str(cfg.split),
    )

    # Storage: method -> metric -> list
    raw: Dict[str, Dict[str, List[float]]] = {}
    for m in methods:
        raw[m] = {
            "bleu": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "meteor": [],
            "r1_no_citation": [],
            "r2_coarse_only": [],
            "r3_laterality_mismatch": [],
            "r4_bilateral_separation": [],
            "weighted_issue": [],
            "iou_union": [],
            "dice_union": [],
            "hit": [],
            "hit_at_8": [],
            "coverage_at_8": [],
            "laterality_grounding_acc": [],
            "unsupported": [],
            "overclaim": [],
            "warm_time_s": [],
            "n_frames_pred": [],
            "n_citations": [],
        }

    per_sample: List[Dict[str, Any]] = []

    for i, batch in enumerate(dl):
        sid = str((batch.get("sample_id") or [""])[0])
        if not sid:
            continue
        volume = batch["volume"][0]
        lesion_masks = (batch.get("lesion_masks") or [{}])[0] or {}
        ref_text = str((batch.get("report_text") or [""])[0] or "")
        volume_shape = tuple(int(x) for x in volume.shape)

        # Tokenize once per sample for fairness.
        t0 = time.perf_counter()
        tokens = tok.tokenize(volume, budget_tokens=int(cfg.budget_tokens), emb_dim=int(cfg.emb_dim), seed=int(cfg.seed) + i)
        t1 = time.perf_counter()

        for method in methods:
            key = (sid, method)
            if key not in preds:
                continue
            pred_text = preds[key]

            frames = extractor.extract_frames(str(pred_text))
            gen = Generation(frames=frames, citations={}, q={}, refusal={}, text="", report_text=str(pred_text))
            gen = attach_posthoc_citations(gen, tokens, k_max=int(cfg.k_max), positive_only=True, overwrite_empty_only=True)

            # Proof metrics
            pm = compute_proof_metrics(gen, tokens, l_min=int(cfg.l_min), weights=weights)

            # Verifier taxonomy (unsupported/overclaim) for interpretability.
            issues = verify(gen, tokens)
            denom = max(1, len(gen.frames))
            issue_counts: Dict[str, int] = {}
            for iss in issues:
                it = str(getattr(iss, "issue_type", ""))
                issue_counts[it] = issue_counts.get(it, 0) + 1

            # Grounding metrics (only meaningful when masks exist)
            gm = compute_generation_grounding(gen, tokens, lesion_masks, volume_shape)

            row: Dict[str, Any] = {
                "sample_id": sid,
                "method": method,
            }

            # Text metrics
            if text_metrics_enabled:
                try:
                    tm = compute_text_metrics(str(pred_text), str(ref_text), cfg=TextMetricConfig(compute_meteor=True))
                except MissingTextMetricDependency:
                    tm = {}
                for k in ("bleu", "rouge1", "rouge2", "rougeL", "meteor"):
                    v = float(tm.get(k, 0.0))
                    raw[method][k].append(v)
                    row[k] = v

            for k in ("r1_no_citation", "r2_coarse_only", "r3_laterality_mismatch", "r4_bilateral_separation", "weighted_issue"):
                v = float(pm.get(k, 0.0))
                raw[method][k].append(v)
                row[k] = v

            # Grounding summary keys
            iou_u = float(gm.get("iou_union", 0.0))
            dice_u = float(gm.get("dice_union", 0.0))
            hit = float(gm.get("hit", gm.get("hit_rate", 0.0)))
            hit_at_8 = float(gm.get("hit_at_8", 0.0))
            cov_at_8 = float(gm.get("coverage_at_8", 0.0))
            lat_acc = float(gm.get("laterality_grounding_acc", gm.get("laterality_grounding_acc", 0.0)))

            raw[method]["iou_union"].append(iou_u)
            raw[method]["dice_union"].append(dice_u)
            raw[method]["hit"].append(hit)
            raw[method]["hit_at_8"].append(hit_at_8)
            raw[method]["coverage_at_8"].append(cov_at_8)
            raw[method]["laterality_grounding_acc"].append(lat_acc)

            row.update(
                {
                    "iou_union": iou_u,
                    "dice_union": dice_u,
                    "hit": hit,
                    "hit_at_8": hit_at_8,
                    "coverage_at_8": cov_at_8,
                    "laterality_grounding_acc": lat_acc,
                }
            )

            unsupported = float(issue_counts.get("U1_unsupported", 0) / denom)
            overclaim = float(issue_counts.get("O1_overclaim", 0) / denom)
            raw[method]["unsupported"].append(unsupported)
            raw[method]["overclaim"].append(overclaim)
            raw[method]["warm_time_s"].append(float(t1 - t0))
            raw[method]["n_frames_pred"].append(float(len(gen.frames)))
            n_citations = sum(len(v or []) for v in (gen.citations or {}).values())
            raw[method]["n_citations"].append(float(n_citations))

            row.update(
                {
                    "unsupported": unsupported,
                    "overclaim": overclaim,
                    "warm_time_s": float(t1 - t0),
                    "n_frames_pred": float(len(gen.frames)),
                    "n_citations": float(n_citations),
                }
            )

            per_sample.append(row)

        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(dl)}] processed")

    summary: Dict[str, Any] = {}
    for m in methods:
        vals = raw[m]
        summary[m] = {k: _mean(v) for k, v in vals.items() if v}
        summary[m]["n"] = int(max(len(next(iter(vals.values()))), 0) if vals else 0)

    save_results_json(
        {
            "config": asdict(cfg),
            "methods": methods,
            "summary": summary,
            "raw": raw,
            "per_sample": per_sample,
        },
        out_dir / "eval_external_predictions.json",
    )
    print(f"wrote: {out_dir / 'eval_external_predictions.json'}")


if __name__ == "__main__":
    main()
