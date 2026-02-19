"""End-to-end Split-DPO closed-loop evaluation (pp.md §5.2).

This script evaluates ProveTok-Agent's write→verify→split loop under:
- heuristic split selection (default allocator), and
- an optional learned split policy trained via verifier-derived DPO.

It reports proof-centric reliability metrics (R1–R4 + WeightedIssue) and trace
statistics on either synthetic volumes or a manifest dataset.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..agent.loop import AgentConfig, run_provetok_agent
from ..bet.split_policy_dpo import load_dpo_split_policy
from ..data import ManifestDataset
from ..eval.metrics_proof import ProofWeights, compute_proof_metrics
from ..pcg.generator import ToyPCG
from ..verifier.pp_v1_1 import create_pp_verifier
from .utils import create_synthetic_volume, make_output_dir, save_results_json, set_seed


@dataclass(frozen=True)
class EvalConfig:
    seed: int = 42
    output_dir: str = "./outputs/figX_split_dpo_closed_loop"

    dataset_type: str = "synthetic"  # "synthetic" | "manifest"
    manifest_path: str = ""
    split: str = "test"
    max_samples: int = 0
    resize_shape: Tuple[int, int, int] = (64, 64, 64)

    n_samples: int = 30

    # Agent
    budget_tokens: int = 256
    init_level: int = 1
    max_depth: int = 3
    max_steps_per_finding: int = 8
    l_min: int = 2
    k_max_citations: int = 8
    emb_dim: int = 32

    # PCG
    topk_citations: int = 3

    # Proof weights
    w1_r1: float = 3.0
    w2_r2: float = 2.0
    w3_r3: float = 2.0
    w4_r4: float = 2.0

    # DPO policy (optional)
    dpo_ckpt: str = ""
    dpo_device: str = "cpu"
    dpo_require_blame: bool = True


def _iter_samples(cfg: EvalConfig):
    if str(cfg.dataset_type) == "manifest":
        if not cfg.manifest_path:
            raise SystemExit("--manifest is required when --dataset-type=manifest")
        ds = ManifestDataset(
            manifest_path=str(cfg.manifest_path),
            split=str(cfg.split),
            max_samples=int(cfg.max_samples),
            resize_shape=tuple(int(x) for x in cfg.resize_shape),
            seed=int(cfg.seed),
        )
        if len(ds) <= 0:
            raise SystemExit("ManifestDataset is empty (check --manifest/--split/--max-samples).")
        for i in range(min(int(cfg.n_samples), len(ds))):
            row = ds[int(i)]
            yield {
                "sample_id": row.get("sample_id", str(i)),
                "volume": row["volume"],
                "affine_zyx": row.get("affine_zyx", None),
            }
        return

    # synthetic
    for i in range(int(cfg.n_samples)):
        vol, _ = create_synthetic_volume(shape=tuple(int(x) for x in cfg.resize_shape), n_lesions=3, seed=int(cfg.seed) + 97 * i)
        yield {"sample_id": f"synthetic_{i}", "volume": vol, "affine_zyx": None}


def _run_one(
    *,
    volume: torch.Tensor,
    affine_zyx: Any,
    pcg: ToyPCG,
    verifier,
    agent_cfg: AgentConfig,
    proof_weights: ProofWeights,
    seed: int,
    split_cell_fn=None,
) -> Dict[str, Any]:
    res = run_provetok_agent(
        volume,
        generator_fn=pcg,
        verifier=verifier,
        cfg=agent_cfg,
        seed=int(seed),
        affine_zyx=affine_zyx,
        split_cell_fn=split_cell_fn,
    )
    m = compute_proof_metrics(res.generation, res.tokens, verifier=verifier, l_min=int(agent_cfg.l_min), weights=proof_weights)

    trace = list(getattr(res, "trace", []) or [])
    n_split = sum(1 for st in trace if str(getattr(st, "action", "")) == "split")
    n_despec = sum(1 for st in trace if str(getattr(st, "action", "")) == "despecify")

    return {
        "proof": {k: float(v) for k, v in (m or {}).items()},
        "trace": {
            "num_steps": int(len(trace)),
            "num_splits": int(n_split),
            "num_despecify": int(n_despec),
            "final_num_tokens": int(len(res.tokens or [])),
            "final_num_cells": int(len(getattr(res, "final_cells", []) or [])),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="FigX: Split-DPO closed-loop evaluation (pp.md §5.2).")
    ap.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest"])
    ap.add_argument("--manifest", type=str, default="", help="Manifest path when dataset-type=manifest")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])

    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="./outputs/figX_split_dpo_closed_loop")

    ap.add_argument("--budget-tokens", type=int, default=256)
    ap.add_argument("--init-level", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-steps-per-finding", type=int, default=8)
    ap.add_argument("--l-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--topk-citations", type=int, default=3)

    ap.add_argument("--proof-w1", type=float, default=3.0)
    ap.add_argument("--proof-w2", type=float, default=2.0)
    ap.add_argument("--proof-w3", type=float, default=2.0)
    ap.add_argument("--proof-w4", type=float, default=2.0)

    ap.add_argument("--dpo-ckpt", type=str, default="", help="Path to split_policy_dpo.pt checkpoint")
    ap.add_argument("--dpo-device", type=str, default="cpu")
    ap.add_argument("--dpo-require-blame", action="store_true", help="Require blamed-only split actions (recommended)")

    ap.add_argument("--debug-sample-id", type=str, default="", help="Run single-sample debug (manifest only) and exit")
    args = ap.parse_args()

    cfg = EvalConfig(
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        dataset_type=str(args.dataset_type),
        manifest_path=str(args.manifest),
        split=str(args.split),
        max_samples=int(args.max_samples),
        resize_shape=tuple(int(x) for x in args.resize_shape),
        n_samples=int(args.n_samples),
        budget_tokens=int(args.budget_tokens),
        init_level=int(args.init_level),
        max_depth=int(args.max_depth),
        max_steps_per_finding=int(args.max_steps_per_finding),
        l_min=int(args.l_min),
        k_max_citations=int(args.k_max),
        emb_dim=int(args.emb_dim),
        topk_citations=int(args.topk_citations),
        w1_r1=float(args.proof_w1),
        w2_r2=float(args.proof_w2),
        w3_r3=float(args.proof_w3),
        w4_r4=float(args.proof_w4),
        dpo_ckpt=str(args.dpo_ckpt),
        dpo_device=str(args.dpo_device),
        dpo_require_blame=bool(args.dpo_require_blame),
    )

    set_seed(int(cfg.seed))
    out_dir = Path(make_output_dir(str(cfg.output_dir), "figX_split_dpo_closed_loop"))
    out_dir.mkdir(parents=True, exist_ok=True)

    verifier = create_pp_verifier(l_min=int(cfg.l_min))
    pcg = ToyPCG(emb_dim=int(cfg.emb_dim), topk=int(cfg.topk_citations), seed=int(cfg.seed))

    agent_cfg = AgentConfig(
        budget_tokens=int(cfg.budget_tokens),
        init_level=int(cfg.init_level),
        max_depth=int(cfg.max_depth),
        max_steps_per_finding=int(cfg.max_steps_per_finding),
        l_min=int(cfg.l_min),
        k_max_citations=int(cfg.k_max_citations),
        emb_dim=int(cfg.emb_dim),
    )

    proof_weights = ProofWeights(
        w1_r1=float(cfg.w1_r1),
        w2_r2=float(cfg.w2_r2),
        w3_r3=float(cfg.w3_r3),
        w4_r4=float(cfg.w4_r4),
    )

    dpo_policy = None
    if cfg.dpo_ckpt:
        dpo_policy = load_dpo_split_policy(str(cfg.dpo_ckpt), device=str(cfg.dpo_device), require_blame=bool(cfg.dpo_require_blame))

    debug_sid = str(getattr(args, "debug_sample_id", "") or "").strip()
    if debug_sid:
        if str(cfg.dataset_type) != "manifest":
            raise SystemExit("--debug-sample-id requires --dataset-type=manifest")

        ds = ManifestDataset(
            manifest_path=str(cfg.manifest_path),
            split=str(cfg.split),
            max_samples=int(cfg.max_samples),
            resize_shape=tuple(int(x) for x in cfg.resize_shape),
            seed=int(cfg.seed),
        )
        idx = None
        for i, r in enumerate(ds.records):
            if str(getattr(r, "scan_hash", "")) == debug_sid:
                idx = int(i)
                break
        if idx is None:
            raise SystemExit(f"sample_id not found in manifest split: {debug_sid}")

        row = ds[int(idx)]
        vol = row["volume"]
        affine_zyx = row.get("affine_zyx", None)

        def _run_variant(name: str, split_fn):
            res = run_provetok_agent(
                vol,
                generator_fn=pcg,
                verifier=verifier,
                cfg=agent_cfg,
                seed=int(cfg.seed),
                affine_zyx=affine_zyx,
                split_cell_fn=split_fn,
            )
            m = compute_proof_metrics(res.generation, res.tokens, verifier=verifier, l_min=int(agent_cfg.l_min), weights=proof_weights)
            final_issues = verifier.verify(res.generation, res.tokens)
            frames = []
            for j, fr in enumerate(res.generation.frames or []):
                frames.append(
                    {
                        "frame_idx": int(j),
                        "finding": str(getattr(fr, "finding", "")),
                        "polarity": str(getattr(fr, "polarity", "")),
                        "laterality": str(getattr(fr, "laterality", "")),
                        "uncertain": bool(getattr(fr, "uncertain", False)),
                        "n_citations": int(len((res.generation.citations or {}).get(int(j), []) or [])),
                        "citations_ref": list((getattr(res.generation, "citations_ref", {}) or {}).get(int(j), []) or []),
                    }
                )
            return {
                "name": name,
                "proof": {k: float(v) for k, v in (m or {}).items()},
                "frames": frames,
                "issues": [
                    {
                        "frame_idx": int(getattr(x, "frame_idx", -1)),
                        "rule_id": str(getattr(x, "rule_id", "")),
                        "issue_type": str(getattr(x, "issue_type", "")),
                        "severity": int(getattr(x, "severity", 0)),
                        "message": str(getattr(x, "message", "")),
                        "evidence_trace": getattr(x, "evidence_trace", {}) or {},
                    }
                    for x in (final_issues or [])
                ],
                "trace": [asdict(st) for st in (res.trace or [])],
                "findings_lines": list(res.findings_lines or []),
                "impression": str(getattr(res, "impression", "")),
            }

        base = _run_variant("heuristic", None)
        dpo = _run_variant("dpo", dpo_policy.pick_cell if dpo_policy is not None else None)
        out_debug = {
            "sample_id": str(row.get("sample_id", debug_sid)),
            "config": asdict(cfg),
            "agent_cfg": asdict(agent_cfg),
            "heuristic": base,
            "dpo": dpo,
            "delta_weighted_issue": float(dpo["proof"].get("weighted_issue", 0.0) - base["proof"].get("weighted_issue", 0.0)),
        }
        debug_path = out_dir / f"debug_{debug_sid}.json"
        save_results_json(out_debug, str(debug_path))
        print(str(debug_path))
        return

    per_sample: List[Dict[str, Any]] = []

    for i, row in enumerate(_iter_samples(cfg)):
        sid = str(row.get("sample_id", i))
        vol = row["volume"]
        affine_zyx = row.get("affine_zyx", None)

        base = _run_one(
            volume=vol,
            affine_zyx=affine_zyx,
            pcg=pcg,
            verifier=verifier,
            agent_cfg=agent_cfg,
            proof_weights=proof_weights,
            seed=int(cfg.seed) + i,
            split_cell_fn=None,
        )
        out_row: Dict[str, Any] = {"sample_id": sid, "heuristic": base}

        if dpo_policy is not None:
            dpo = _run_one(
                volume=vol,
                affine_zyx=affine_zyx,
                pcg=pcg,
                verifier=verifier,
                agent_cfg=agent_cfg,
                proof_weights=proof_weights,
                seed=int(cfg.seed) + i,
                split_cell_fn=dpo_policy.pick_cell,
            )
            out_row["dpo"] = dpo
            out_row["delta_weighted_issue"] = float(dpo["proof"].get("weighted_issue", 0.0) - base["proof"].get("weighted_issue", 0.0))

        per_sample.append(out_row)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{int(cfg.n_samples)}] processed")

    # Aggregate
    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def _collect(path: List[str], variant: str) -> List[float]:
        out = []
        for r in per_sample:
            v = r.get(variant, {})
            cur = v
            ok = True
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    ok = False
                    break
                cur = cur[k]
            if ok:
                try:
                    out.append(float(cur))
                except Exception:
                    pass
        return out

    summary: Dict[str, Any] = {
        "heuristic": {
            "weighted_issue": _mean(_collect(["proof", "weighted_issue"], "heuristic")),
            "r2_coarse_only": _mean(_collect(["proof", "r2_coarse_only"], "heuristic")),
            "r3_laterality_mismatch": _mean(_collect(["proof", "r3_laterality_mismatch"], "heuristic")),
            "num_splits": _mean(_collect(["trace", "num_splits"], "heuristic")),
        }
    }
    if dpo_policy is not None:
        summary["dpo"] = {
            "weighted_issue": _mean(_collect(["proof", "weighted_issue"], "dpo")),
            "r2_coarse_only": _mean(_collect(["proof", "r2_coarse_only"], "dpo")),
            "r3_laterality_mismatch": _mean(_collect(["proof", "r3_laterality_mismatch"], "dpo")),
            "num_splits": _mean(_collect(["trace", "num_splits"], "dpo")),
            "delta_weighted_issue": _mean([float(r.get("delta_weighted_issue", 0.0)) for r in per_sample if "delta_weighted_issue" in r]),
        }

    out = {
        "config": asdict(cfg),
        "agent_cfg": asdict(agent_cfg),
        "summary": summary,
        "per_sample": per_sample,
    }
    save_results_json(out, str(out_dir / "figX_split_dpo_closed_loop.json"))
    print(str(out_dir / "figX_split_dpo_closed_loop.json"))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
