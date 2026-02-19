"""Fig 4: reliability-budget Pareto curves for ProveTok-Agent (pp.md v1.1, §7.1).

This runner evaluates system variants under token budgets B and aggregates
WeightedIssue (and R1-R4 rates) with hierarchical bootstrap CIs:
1) average over seeds per sample
2) bootstrap across samples

Variants (pp.md §6.3 / §7.1):
- nosplit: Proof+Verify-NoSplit
- heuristic: Proof+Verify+Split-Heuristic
- dpo: Proof+Verify+Split-DPO (requires a trained split policy checkpoint)

Outputs:
- <output_dir>/budget_<B>/seed_<seed>/agent_pareto.json (per budget/seed)
- <output_dir>/fig4_agent_pareto_multiseed.json (aggregated curve + CI)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..agent.loop import AgentConfig, run_provetok_agent
from ..bet.split_policy_dpo import DPOSplitPolicy, load_dpo_split_policy
from ..data.io import load_volume_and_affine
from ..data.manifest_schema import ManifestRecord, load_manifest
from ..eval.metrics_proof import ProofWeights, compute_proof_metrics
from ..eval.stats import bootstrap_mean_ci
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.pp_v1_1 import create_pp_verifier
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .utils import save_results_json, set_seed


@dataclass(frozen=True)
class RunnerConfig:
    dataset_type: str = "synthetic"  # "synthetic" | "manifest"
    manifest_path: str = ""
    split: str = "test"
    resize_shape: Tuple[int, int, int] = (64, 64, 64)
    max_samples: int = 0
    n_samples: int = 50

    budgets: Tuple[int, ...] = (64, 128, 256, 512)
    seeds: Tuple[int, ...] = (0, 1, 2)
    variants: Tuple[str, ...] = ("nosplit", "heuristic")

    # Agent
    init_level: int = 1
    max_depth: int = 3
    max_steps_per_finding: int = 32
    k_max_citations: int = 8
    l_min: int = 2
    emb_dim: int = 32

    # PCG
    pcg_backend: str = "toy"  # "toy" | "llama2"
    topk_citations: int = 3

    # Llama2 PCG (when pcg_backend=llama2)
    llama2_path: str = "/data/models/Llama-2-7b-chat-hf"
    llama2_quant: str = "fp16"  # "fp16" | "8bit"
    llama2_contract_mode: str = "full"  # "free_form" | "schema_only" | "schema_citations" | "full"
    llama2_citation_source: str = "score_override"  # "score_override" | "llm"
    llama2_max_frames: int = 1
    llama2_lora_adapter: str = ""
    llama2_lora_merge: bool = False
    b_gen: int = 128

    # Proof weights
    w1_r1: float = 3.0
    w2_r2: float = 2.0
    w3_r3: float = 2.0
    w4_r4: float = 2.0

    # DPO split policy (when variants include "dpo")
    dpo_ckpt: str = ""
    dpo_device: str = "cpu"
    dpo_require_blame: bool = True

    # IO
    output_dir: str = "./outputs/fig4_agent_pareto_multiseed"
    resume: bool = False
    n_bootstrap: int = 10_000
    ci: float = 0.95


def _resize_volume(vol: torch.Tensor, *, resize_shape: Tuple[int, int, int]) -> torch.Tensor:
    tgt = tuple(int(x) for x in resize_shape)
    if tuple(int(x) for x in vol.shape) == tgt:
        return vol
    x = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    y = F.interpolate(x, size=tgt, mode="trilinear", align_corners=False)
    return y[0, 0]


def _load_manifest_records(cfg: RunnerConfig) -> List[ManifestRecord]:
    if not cfg.manifest_path:
        raise SystemExit("--manifest is required when --dataset-type=manifest")
    recs = [r for r in load_manifest(str(cfg.manifest_path)) if str(r.split) == str(cfg.split)]
    recs = sorted(recs, key=lambda r: str(r.scan_hash))
    if int(cfg.max_samples) > 0:
        recs = recs[: int(cfg.max_samples)]
    if not recs:
        raise SystemExit("No manifest records found (check --manifest/--split/--max-samples).")
    return recs


def _iter_samples(cfg: RunnerConfig) -> List[Dict[str, Any]]:
    if str(cfg.dataset_type) == "manifest":
        recs = _load_manifest_records(cfg)
        out: List[Dict[str, Any]] = []
        for i, r in enumerate(recs[: int(cfg.n_samples)]):
            vol, aff = load_volume_and_affine(str(r.volume_path), seed=1337 + i)
            vol = _resize_volume(vol, resize_shape=cfg.resize_shape)
            out.append({"sample_id": str(r.scan_hash), "volume": vol, "affine_zyx": aff})
        return out

    # synthetic: deterministic per index
    from .utils import create_synthetic_volume

    out = []
    for i in range(int(cfg.n_samples)):
        vol, _ = create_synthetic_volume(shape=tuple(int(x) for x in cfg.resize_shape), n_lesions=3, seed=4242 + 97 * i)
        out.append({"sample_id": f"synthetic_{i}", "volume": vol, "affine_zyx": None})
    return out


_LLAMA2_PCG_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _build_pcg(cfg: RunnerConfig):
    if str(cfg.pcg_backend) == "llama2":
        key = (
            str(cfg.llama2_path),
            str(cfg.llama2_quant),
            str(cfg.llama2_contract_mode),
            str(cfg.llama2_citation_source),
            int(cfg.llama2_max_frames),
            int(cfg.topk_citations),
            int(cfg.b_gen),
            str(cfg.llama2_lora_adapter),
            bool(cfg.llama2_lora_merge),
        )
        pcg = _LLAMA2_PCG_CACHE.get(key)
        if pcg is not None:
            return pcg

        from ..pcg.llama2_pcg import Llama2PCG, Llama2PCGConfig

        pcg = Llama2PCG(
            Llama2PCGConfig(
                model_path=str(cfg.llama2_path),
                device="cuda",
                quantization=str(cfg.llama2_quant),
                max_new_tokens=max(128, int(cfg.b_gen)),
                temperature=0.0,
                topk_citations=int(cfg.topk_citations),
                contract_mode=str(cfg.llama2_contract_mode),
                citation_source=str(cfg.llama2_citation_source),
                max_frames=int(cfg.llama2_max_frames),
                lora_adapter_path=str(cfg.llama2_lora_adapter),
                lora_merge=bool(cfg.llama2_lora_merge),
            )
        )
        _LLAMA2_PCG_CACHE[key] = pcg
        return pcg

    from ..pcg.generator import ToyPCG

    return ToyPCG(emb_dim=int(cfg.emb_dim), topk=int(cfg.topk_citations), seed=0, citation_strategy="attn_score")


def _variant_split_fn(variant: str, *, dpo_policy: Optional[DPOSplitPolicy]):
    v = str(variant).strip().lower()
    if v == "nosplit":
        return lambda _cells, _tokens, _issues: None
    if v == "heuristic":
        return None
    if v == "dpo":
        if dpo_policy is None:
            raise RuntimeError("variant=dpo requires --dpo-ckpt")
        return dpo_policy.pick_cell
    raise ValueError(f"Unknown variant: {variant!r}")


def _mean_and_ci_from_seed_sample_matrix(x: np.ndarray, *, n_boot: int, seed: int, ci: float) -> Dict[str, float]:
    """Hierarchical mean CI: mean over seeds per sample, bootstrap across samples."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (S,N), got shape={x.shape}")
    if x.shape[1] == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if np.isnan(x).any():
        keep = ~np.isnan(x).any(axis=0)
        x = x[:, keep]
        if x.shape[1] == 0:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    per_sample = x.mean(axis=0)
    res = bootstrap_mean_ci(per_sample.tolist(), n_boot=int(n_boot), seed=int(seed), ci=float(ci))
    return {"mean": float(res.mean), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}


def _run_one_budget_seed(
    *,
    cfg: RunnerConfig,
    budget_tokens: int,
    seed: int,
    samples: List[Dict[str, Any]],
    pcg,
    verifier,
    weights: ProofWeights,
    dpo_policy: Optional[DPOSplitPolicy],
    out_path: Path,
) -> Dict[str, Any]:
    set_seed(int(seed))

    agent_cfg = AgentConfig(
        budget_tokens=int(budget_tokens),
        init_level=int(cfg.init_level),
        max_depth=int(cfg.max_depth),
        max_steps_per_finding=int(cfg.max_steps_per_finding),
        k_max_citations=int(cfg.k_max_citations),
        l_min=int(cfg.l_min),
        emb_dim=int(cfg.emb_dim),
    )

    variants = list(cfg.variants)
    sample_ids: List[str] = []
    raw: Dict[str, Dict[str, List[float]]] = {str(v): {} for v in variants}

    for i, row in enumerate(samples):
        sid = str(row.get("sample_id", i))
        sample_ids.append(sid)
        vol = row["volume"]
        aff = row.get("affine_zyx", None)

        for v in variants:
            split_fn = _variant_split_fn(str(v), dpo_policy=dpo_policy)
            res = run_provetok_agent(
                vol,
                generator_fn=pcg,
                verifier=verifier,
                cfg=agent_cfg,
                seed=int(seed) + 10_000 * int(budget_tokens) + i,
                affine_zyx=aff,
                split_cell_fn=split_fn,
            )
            pm = compute_proof_metrics(res.generation, res.tokens, verifier=verifier, l_min=int(cfg.l_min), weights=weights)

            for k in (
                "weighted_issue",
                "r1_no_citation",
                "r2_coarse_only",
                "r3_laterality_mismatch",
                "r4_bilateral_separation",
            ):
                raw[str(v)].setdefault(k, []).append(float(pm.get(k, 0.0)))

            trace = list(getattr(res, "trace", []) or [])
            n_split = sum(1 for st in trace if str(getattr(st, "action", "")) == "split")
            n_despec = sum(1 for st in trace if str(getattr(st, "action", "")) == "despecify")
            raw[str(v)].setdefault("num_splits", []).append(float(n_split))
            raw[str(v)].setdefault("num_despecify", []).append(float(n_despec))
            raw[str(v)].setdefault("final_num_tokens", []).append(float(len(res.tokens or [])))

        if (i + 1) % 10 == 0:
            print(f"[budget={budget_tokens} seed={seed}] processed {i+1}/{len(samples)}")

    data_revision, split_manifest_path = try_manifest_revision(str(cfg.manifest_path))
    meta = build_artifact_meta(
        repo_root=Path(__file__).resolve().parents[2],
        seed=int(seed),
        config={"budget_tokens": int(budget_tokens), **asdict(cfg)},
        rule_set_version=str(RULE_SET_VERSION),
        schema_version=str(SCHEMA_VERSION),
        taxonomy_version=str(TAXONOMY_VERSION),
        data_revision=str(data_revision),
        split_manifest_path=str(split_manifest_path),
    )

    rep = {
        "meta": meta.to_dict(),
        "budget_tokens": int(budget_tokens),
        "seed": int(seed),
        "sample_ids": list(sample_ids),
        "raw": raw,
    }
    save_results_json(rep, str(out_path))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig4: ProveTok-Agent reliability-budget Pareto curves (multi-seed).")
    ap.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest"])
    ap.add_argument("--manifest", type=str, default="", help="Manifest path when dataset-type=manifest")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=50)

    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--variants", type=str, nargs="+", default=[], help="Subset of {nosplit,heuristic,dpo}")

    ap.add_argument("--init-level", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-steps-per-finding", type=int, default=32)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--l-min", type=int, default=2)
    ap.add_argument("--emb-dim", type=int, default=32)

    ap.add_argument("--pcg", type=str, default="toy", choices=["toy", "llama2"])
    ap.add_argument("--topk-citations", type=int, default=3)

    ap.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])
    ap.add_argument("--llama2-contract-mode", type=str, default="full", choices=["free_form", "schema_only", "schema_citations", "full"])
    ap.add_argument("--llama2-citation-source", type=str, default="score_override", choices=["score_override", "llm"])
    ap.add_argument("--llama2-max-frames", type=int, default=1)
    ap.add_argument("--llama2-lora-adapter", type=str, default="", help="Optional LoRA/PEFT adapter path")
    ap.add_argument("--llama2-lora-merge", action="store_true", help="Merge LoRA adapter into base model (if supported)")
    ap.add_argument("--b-gen", type=int, default=128)

    ap.add_argument("--proof-w1", type=float, default=3.0)
    ap.add_argument("--proof-w2", type=float, default=2.0)
    ap.add_argument("--proof-w3", type=float, default=2.0)
    ap.add_argument("--proof-w4", type=float, default=2.0)

    ap.add_argument("--dpo-ckpt", type=str, default="", help="Path to split_policy_dpo.pt checkpoint")
    ap.add_argument("--dpo-device", type=str, default="cpu")
    ap.add_argument("--dpo-require-blame", action="store_true", help="Require blamed-only split actions (recommended)")

    ap.add_argument("--output-dir", type=str, default="./outputs/fig4_agent_pareto_multiseed")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--no-plot", action="store_true", help="(kept for parity; plotting not implemented here)")
    args = ap.parse_args()

    cfg = RunnerConfig(
        dataset_type=str(args.dataset_type),
        manifest_path=str(args.manifest),
        split=str(args.split),
        resize_shape=tuple(int(x) for x in args.resize_shape),
        max_samples=int(args.max_samples),
        n_samples=int(args.n_samples),
        budgets=tuple(int(b) for b in args.budgets),
        seeds=tuple(int(s) for s in args.seeds),
        variants=tuple(str(v) for v in (args.variants or [])) if args.variants else ("nosplit", "heuristic"),
        init_level=int(args.init_level),
        max_depth=int(args.max_depth),
        max_steps_per_finding=int(args.max_steps_per_finding),
        k_max_citations=int(args.k_max),
        l_min=int(args.l_min),
        emb_dim=int(args.emb_dim),
        pcg_backend=str(args.pcg),
        topk_citations=int(args.topk_citations),
        llama2_path=str(args.llama2_path),
        llama2_quant=str(args.llama2_quant),
        llama2_contract_mode=str(args.llama2_contract_mode),
        llama2_citation_source=str(args.llama2_citation_source),
        llama2_max_frames=int(args.llama2_max_frames),
        llama2_lora_adapter=str(args.llama2_lora_adapter),
        llama2_lora_merge=bool(args.llama2_lora_merge),
        b_gen=int(args.b_gen),
        w1_r1=float(args.proof_w1),
        w2_r2=float(args.proof_w2),
        w3_r3=float(args.proof_w3),
        w4_r4=float(args.proof_w4),
        dpo_ckpt=str(args.dpo_ckpt),
        dpo_device=str(args.dpo_device),
        dpo_require_blame=bool(args.dpo_require_blame),
        output_dir=str(args.output_dir),
        resume=bool(args.resume),
        n_bootstrap=int(args.n_bootstrap),
        ci=float(args.ci),
    )

    variants = list(cfg.variants)
    if not args.variants and cfg.dpo_ckpt:
        variants = ["nosplit", "heuristic", "dpo"]
    if "dpo" in [v.lower() for v in variants] and not cfg.dpo_ckpt:
        raise SystemExit("variants include 'dpo' but --dpo-ckpt is empty")
    cfg = RunnerConfig(**{**asdict(cfg), "variants": tuple(variants)})

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Reduce OpenMP oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    samples = _iter_samples(cfg)
    verifier = create_pp_verifier(l_min=int(cfg.l_min))
    pcg = _build_pcg(cfg)
    weights = ProofWeights(w1_r1=float(cfg.w1_r1), w2_r2=float(cfg.w2_r2), w3_r3=float(cfg.w3_r3), w4_r4=float(cfg.w4_r4))

    dpo_policy = None
    if cfg.dpo_ckpt:
        dpo_policy = load_dpo_split_policy(str(cfg.dpo_ckpt), device=str(cfg.dpo_device), require_blame=bool(cfg.dpo_require_blame))

    per_budget: Dict[int, Dict[int, Dict[str, Any]]] = {}
    per_budget_dirs: Dict[int, Dict[int, str]] = {}

    for b in cfg.budgets:
        per_budget[int(b)] = {}
        per_budget_dirs[int(b)] = {}
        for s in cfg.seeds:
            out_dir = Path(cfg.output_dir) / f"budget_{int(b)}" / f"seed_{int(s)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "agent_pareto.json"

            if cfg.resume and out_path.exists():
                try:
                    rep = json.loads(out_path.read_text(encoding="utf-8"))
                except Exception:
                    rep = None
                if isinstance(rep, dict) and rep.get("raw"):
                    per_budget[int(b)][int(s)] = rep
                    per_budget_dirs[int(b)][int(s)] = str(out_dir)
                    continue

            rep = _run_one_budget_seed(
                cfg=cfg,
                budget_tokens=int(b),
                seed=int(s),
                samples=samples,
                pcg=pcg,
                verifier=verifier,
                weights=weights,
                dpo_policy=dpo_policy,
                out_path=out_path,
            )
            per_budget[int(b)][int(s)] = rep
            per_budget_dirs[int(b)][int(s)] = str(out_dir)

    any_b = int(cfg.budgets[0])
    any_s = int(cfg.seeds[0])
    variants0 = sorted(per_budget[any_b][any_s]["raw"].keys())
    metric_keys = sorted(per_budget[any_b][any_s]["raw"][variants0[0]].keys())

    metrics_out: Dict[str, Dict[str, List[Dict[str, float]]]] = {k: {v: [] for v in variants0} for k in metric_keys}
    for bidx, b in enumerate(cfg.budgets):
        for v in variants0:
            for k in metric_keys:
                mats = []
                for s in cfg.seeds:
                    mats.append(per_budget[int(b)][int(s)]["raw"][v][k])
                arr = np.asarray(mats, dtype=np.float64)  # (S,N)
                stable = hashlib.sha1(f"{v}:{k}".encode("utf-8")).digest()
                stable_seed = int.from_bytes(stable[:4], "little", signed=False)
                ci_rec = _mean_and_ci_from_seed_sample_matrix(
                    arr,
                    n_boot=int(cfg.n_bootstrap),
                    seed=int(cfg.seeds[0]) + 1000 * bidx + (stable_seed % 997),
                    ci=float(cfg.ci),
                )
                metrics_out[k][v].append(ci_rec)

    data_revision, split_manifest_path = try_manifest_revision(str(cfg.manifest_path))
    meta = build_artifact_meta(
        repo_root=Path(__file__).resolve().parents[2],
        seed=int(cfg.seeds[0]),
        config=asdict(cfg),
        rule_set_version=str(RULE_SET_VERSION),
        schema_version=str(SCHEMA_VERSION),
        taxonomy_version=str(TAXONOMY_VERSION),
        data_revision=str(data_revision),
        split_manifest_path=str(split_manifest_path),
    )

    out = {
        "meta": meta.to_dict(),
        "budgets": [int(b) for b in cfg.budgets],
        "variants": variants0,
        "per_budget_dirs": {str(k): {str(sk): sv for sk, sv in v.items()} for k, v in per_budget_dirs.items()},
        "metrics": metrics_out,
    }
    out_path = Path(cfg.output_dir) / "fig4_agent_pareto_multiseed.json"
    save_results_json(out, str(out_path))
    print(str(out_path))


if __name__ == "__main__":
    main()
