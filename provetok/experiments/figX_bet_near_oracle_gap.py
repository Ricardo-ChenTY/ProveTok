"""Fig X: BET near-oracle gap (C0007).

Goal: show that our adaptive tokenization baseline (e.g. `bet_alg`) is close to
the per-sample oracle chosen from a fixed candidate pool under the same budget.

We define the oracle per sample as:
    oracle_metric(sample) = max_{method in pool} metric(method, sample)

and report the relative gap for our method:
    gap_ratio = (oracle - ours) / max(oracle, eps)

Primary metric: `combined` (same combined used elsewhere: nlg_weight*frame_f1 + grounding_weight*iou).
Secondary metric: `iou` (union-level grounding).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..eval.stats import bootstrap_mean_ci
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .run_baselines import BaselineRunConfig, run_baselines
from .utils import save_results_json


def _stable_seed(seed0: int, tag: str) -> int:
    h = hashlib.sha1(tag.encode("utf-8")).digest()
    return int(seed0) + int.from_bytes(h[:4], "little", signed=False) % 1_000_000


def _budget_dir_key(b: float) -> str:
    return f"{float(b):g}".replace(".", "_")


def _gap_ratio(oracle: float, ours: float, *, eps: float) -> float:
    o = float(oracle)
    if o <= float(eps):
        return 0.0
    return float(max(0.0, (o - float(ours)) / max(o, float(eps))))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_one(cfg_dict: Dict[str, Any], out_path: str) -> None:
    rep = run_baselines(BaselineRunConfig(**cfg_dict))
    save_results_json(rep, out_path)


def _stack_seed_sample(per_seed: Dict[int, Dict[str, Any]], *, method: str, key: str, seeds: List[int]) -> np.ndarray:
    mats = []
    for s in seeds:
        mats.append(per_seed[int(s)]["raw"][str(method)][str(key)])
    return np.asarray(mats, dtype=np.float64)  # (S, N)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute BET near-oracle gap across budgets and seeds.")
    ap.add_argument("--dataset-type", type=str, default="manifest", choices=["synthetic", "manifest"])
    ap.add_argument("--manifest", type=str, default="", help="Manifest path when dataset-type=manifest")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--pcg", type=str, default="toy", choices=["toy", "llama2"])
    ap.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])
    ap.add_argument("--methods", type=str, nargs="+", default=["fixed_grid", "roi_variance", "roi_crop", "provetok_lesionness", "bet_alg"])
    ap.add_argument("--ours", type=str, default="bet_alg", help="Method name to treat as ours.")
    ap.add_argument("--budgets", type=float, nargs="+", required=True, help="FLOPs total budgets (budget caps).")
    ap.add_argument("--b-gen", type=int, default=128)
    ap.add_argument("--n-verify", type=int, default=1)
    ap.add_argument("--costs-json", type=str, default="", help="Optional JSON with ComputeUnitCosts.")
    ap.add_argument("--selector-ratio", type=float, default=0.1)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--topk-citations", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--gap-threshold", type=float, default=0.15, help="Per-budget mean gap ratio threshold.")
    ap.add_argument("--min-pass-budgets", type=int, default=5, help="How many budgets must pass to prove C0007.")
    ap.add_argument("--lesionness-weights", type=str, default="", help="Optional lesionness_head.pt (for provetok_lesionness/bet_alg).")
    ap.add_argument("--lesionness-device", type=str, default="cpu")
    ap.add_argument("--lesionness-score-level-power", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output-dir", type=str, default="./outputs/figX_bet_near_oracle_gap")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    budgets = [float(b) for b in args.budgets]
    seeds = [int(s) for s in args.seeds]
    methods = [str(m) for m in args.methods]
    ours = str(args.ours)
    if ours not in methods:
        raise SystemExit(f"--ours must be included in --methods (ours={ours!r}, methods={methods})")

    per_budget: Dict[str, Dict[int, Dict[str, Any]]] = {}
    per_budget_dirs: Dict[str, Dict[int, str]] = {}
    pending: List[Dict[str, Any]] = []
    for b in budgets:
        b_key = f"{b:g}"
        per_budget[b_key] = {}
        per_budget_dirs[b_key] = {}
        for s in seeds:
            seed_dir = out_root / f"budget_{_budget_dir_key(b)}" / f"seed_{int(s)}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            out_path = seed_dir / "baselines.json"

            if args.resume and out_path.exists():
                try:
                    rep = _load_json(out_path)
                except Exception:
                    rep = None
                if isinstance(rep, dict) and rep:
                    per_budget[b_key][int(s)] = rep
                    per_budget_dirs[b_key][int(s)] = str(seed_dir)
                    continue

            cfg = BaselineRunConfig(
                dataset_type=str(args.dataset_type),
                manifest_path=str(args.manifest),
                split=str(args.split),
                n_samples=int(args.n_samples),
                seed=int(s),
                output_dir=str(seed_dir),
                flops_total=float(b),
                b_gen=int(args.b_gen),
                n_verify=int(args.n_verify),
                costs_json=str(args.costs_json),
                selector_ratio=float(args.selector_ratio),
                resize_shape=tuple(int(x) for x in args.resize_shape),
                pcg_backend=str(args.pcg),
                llama2_path=str(args.llama2_path),
                llama2_quant=str(args.llama2_quant),
                methods=list(methods),
                topk_citations=int(args.topk_citations),
                lesionness_weights=str(args.lesionness_weights),
                lesionness_device=str(args.lesionness_device),
                lesionness_score_level_power=float(args.lesionness_score_level_power),
                compute_text_metrics=False,  # keep cheap; this experiment is about grounding/selection
            )
            pending.append({"b_key": b_key, "seed": int(s), "seed_dir": str(seed_dir), "cfg_dict": cfg.__dict__, "out_path": str(out_path)})

    if pending:
        workers = max(1, int(args.workers))
        if workers == 1:
            for job in pending:
                _run_one(job["cfg_dict"], job["out_path"])
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_one, job["cfg_dict"], job["out_path"]) for job in pending]
                for fut in as_completed(futs):
                    fut.result()

        for job in pending:
            rep = _load_json(Path(str(job["out_path"])))
            per_budget[str(job["b_key"])][int(job["seed"])] = rep
            per_budget_dirs[str(job["b_key"])][int(job["seed"])] = str(job["seed_dir"])

    # Aggregate: near-oracle gap per budget.
    rows: List[Dict[str, Any]] = []
    rows_iou: List[Dict[str, Any]] = []
    pass_flags: List[bool] = []

    for bidx, b in enumerate(budgets):
        b_key = f"{b:g}"
        seed0 = seeds[0]
        per_seed = per_budget[b_key]

        if str(args.dataset_type) == "manifest":
            # Hierarchical: mean over seeds per sample index, then bootstrap over samples.
            combined_mean_by_method: Dict[str, np.ndarray] = {}
            iou_mean_by_method: Dict[str, np.ndarray] = {}
            for m in methods:
                combined = _stack_seed_sample(per_seed, method=m, key="combined", seeds=seeds).mean(axis=0)
                iou = _stack_seed_sample(per_seed, method=m, key="iou", seeds=seeds).mean(axis=0)
                combined_mean_by_method[m] = combined
                iou_mean_by_method[m] = iou

            oracle_combined = np.max(np.stack([combined_mean_by_method[m] for m in methods], axis=0), axis=0)
            oracle_iou = np.max(np.stack([iou_mean_by_method[m] for m in methods], axis=0), axis=0)
            ours_combined = combined_mean_by_method[ours]
            ours_iou = iou_mean_by_method[ours]

            gaps = [_gap_ratio(o, a, eps=1e-12) for o, a in zip(oracle_combined.tolist(), ours_combined.tolist())]
            gaps_iou = [_gap_ratio(o, a, eps=1e-12) for o, a in zip(oracle_iou.tolist(), ours_iou.tolist())]
        else:
            # Synthetic: seeds correspond to different samples; treat (seed, idx) as i.i.d.
            gaps = []
            gaps_iou = []
            for s in seeds:
                comb_by_m = {m: np.asarray(per_seed[int(s)]["raw"][m]["combined"], dtype=np.float64) for m in methods}
                iou_by_m = {m: np.asarray(per_seed[int(s)]["raw"][m]["iou"], dtype=np.float64) for m in methods}
                n = int(comb_by_m[ours].shape[0])
                for j in range(n):
                    oc = max(float(comb_by_m[m][j]) for m in methods)
                    oi = max(float(iou_by_m[m][j]) for m in methods)
                    gaps.append(_gap_ratio(oc, float(comb_by_m[ours][j]), eps=1e-12))
                    gaps_iou.append(_gap_ratio(oi, float(iou_by_m[ours][j]), eps=1e-12))

            oracle_combined = float("nan")
            oracle_iou = float("nan")
            ours_combined = float("nan")
            ours_iou = float("nan")

        gap_ci = bootstrap_mean_ci(gaps, n_boot=int(args.n_bootstrap), seed=_stable_seed(seed0, f"gap:{bidx}"), ci=float(args.ci))
        gap_iou_ci = bootstrap_mean_ci(gaps_iou, n_boot=int(args.n_bootstrap), seed=_stable_seed(seed0, f"gap_iou:{bidx}"), ci=float(args.ci))

        ours_mean = float(np.mean(ours_combined)) if isinstance(ours_combined, np.ndarray) else float("nan")
        oracle_mean = float(np.mean(oracle_combined)) if isinstance(oracle_combined, np.ndarray) else float("nan")
        ours_iou_mean = float(np.mean(ours_iou)) if isinstance(ours_iou, np.ndarray) else float("nan")
        oracle_iou_mean = float(np.mean(oracle_iou)) if isinstance(oracle_iou, np.ndarray) else float("nan")

        passed = bool(float(gap_ci.mean) <= float(args.gap_threshold) + 1e-12)
        pass_flags.append(passed)

        rows.append(
            {
                "budget": float(b),
                "ours_mean": ours_mean,
                "oracle_mean": oracle_mean,
                "gap_ratio": {"mean": float(gap_ci.mean), "ci_low": float(gap_ci.ci_low), "ci_high": float(gap_ci.ci_high)},
                "passed": bool(passed),
            }
        )
        rows_iou.append(
            {
                "budget": float(b),
                "ours_mean": ours_iou_mean,
                "oracle_mean": oracle_iou_mean,
                "gap_ratio": {"mean": float(gap_iou_ci.mean), "ci_low": float(gap_iou_ci.ci_low), "ci_high": float(gap_iou_ci.ci_high)},
            }
        )

    pass_count = int(sum(1 for x in pass_flags if x))
    proved = bool(pass_count >= int(args.min_pass_budgets))

    # Artifact meta.
    repo_root = Path(__file__).resolve().parents[2]
    data_revision = "synthetic"
    split_manifest_path = ""
    if args.dataset_type == "manifest":
        data_revision, split_manifest_path = try_manifest_revision(str(args.manifest))
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(seeds[0]),
        config={
            "dataset_type": args.dataset_type,
            "manifest_path": args.manifest,
            "split": args.split,
            "resize_shape": list(args.resize_shape),
            "pcg": args.pcg,
            "llama2_path": args.llama2_path,
            "llama2_quant": args.llama2_quant,
            "methods": methods,
            "ours": ours,
            "budgets": budgets,
            "b_gen": int(args.b_gen),
            "n_verify": int(args.n_verify),
            "costs_json": args.costs_json,
            "selector_ratio": float(args.selector_ratio),
            "n_samples": int(args.n_samples),
            "topk_citations": int(args.topk_citations),
            "seeds": seeds,
            "n_bootstrap": int(args.n_bootstrap),
            "ci": float(args.ci),
            "gap_threshold": float(args.gap_threshold),
            "min_pass_budgets": int(args.min_pass_budgets),
            "lesionness_weights": str(args.lesionness_weights),
            "lesionness_device": str(args.lesionness_device),
            "lesionness_score_level_power": float(args.lesionness_score_level_power),
        },
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    report = {
        "meta": meta.to_dict(),
        "claim_id": "C0007",
        "proved": bool(proved),
        "pass_count": int(pass_count),
        "required_pass_count": int(args.min_pass_budgets),
        "gap_threshold": float(args.gap_threshold),
        "primary_metric": "combined",
        "secondary_metric": "iou",
        "rows": rows,
        "rows_iou": rows_iou,
        "per_budget_seed_dirs": {bk: {str(s): d for s, d in dirs.items()} for bk, dirs in per_budget_dirs.items()},
    }

    out_json = out_root / "figX_bet_near_oracle_gap.json"
    save_results_json(report, str(out_json))
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()

