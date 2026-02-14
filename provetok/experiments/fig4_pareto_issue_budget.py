"""Figure 4 (Main): reliability–budget Pareto curve (pp.md v1.1).

This runner evaluates ProveTok-Agent variants across token budgets B and reports
pp.md proof metrics (R1–R4 + WeightedIssue) with hierarchical bootstrap CIs:
1) mean over seeds per sample
2) bootstrap across samples

Outputs:
- <output_dir>/fig4_pareto_issue_budget.json
- <output_dir>/fig4_pareto_issue_budget.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..agent.loop import AgentConfig, run_provetok_agent
from ..bet.split_policy_dpo import load_dpo_split_policy
from ..eval.metrics_proof import ProofWeights, attach_posthoc_citations, compute_proof_metrics
from ..pcg.generator import ToyPCG
from ..pcg.llama2_pcg import Llama2PCG, Llama2PCGConfig
from ..pcg.text_contract import enforce_pp_contract
from ..types import Generation, Token
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.pp_v1_1 import PP_RULE_SET_VERSION, create_pp_verifier
from .utils import create_synthetic_volume, save_results_json, set_seed
from ..data.io import load_volume
from ..data.manifest_schema import load_manifest, split_records


def _hier_mean_ci(
    x_seed_by_sample: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    ci: float,
) -> Dict[str, float]:
    from ..eval.stats import bootstrap_mean_ci

    if x_seed_by_sample.ndim != 2:
        raise ValueError(f"expected 2D array (S,N), got shape={tuple(x_seed_by_sample.shape)}")
    if x_seed_by_sample.shape[1] == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    per_sample = x_seed_by_sample.mean(axis=0)
    res = bootstrap_mean_ci(per_sample.tolist(), n_boot=int(n_boot), seed=int(seed), ci=float(ci))
    return {"mean": float(res.mean), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}


def _init_cells(*, init_level: int, budget_tokens: int, max_depth: int) -> List[Any]:
    from ..grid.cells import Cell, root_cell

    init_level = int(min(max_depth, max(0, int(init_level))))
    while init_level > 0:
        n = 2 ** int(init_level)
        if int(n * n * n) <= int(budget_tokens):
            break
        init_level -= 1
    if init_level <= 0:
        return [root_cell()]
    n = 2 ** int(init_level)
    return [Cell(level=init_level, ix=ix, iy=iy, iz=iz) for ix in range(n) for iy in range(n) for iz in range(n)]


def _fit_init_level(*, init_level: int, budget_tokens: int, max_depth: int) -> int:
    """Match agent init-level fitting logic (ensures init token coverage fits budget)."""
    init_level = int(min(max_depth, max(0, int(init_level))))
    while init_level > 0:
        n = 2 ** int(init_level)
        if int(n * n * n) <= int(budget_tokens):
            break
        init_level -= 1
    return int(init_level)


def _proof_no_verify(
    volume: torch.Tensor,
    *,
    generator_fn,
    emb_dim: int,
    budget_tokens: int,
    init_level: int,
    max_depth: int,
    k_max: int,
) -> Tuple[List[Token], Generation]:
    from ..bet.tokenize import TokenEncoder

    cells = _init_cells(init_level=int(init_level), budget_tokens=int(budget_tokens), max_depth=int(max_depth))
    enc = TokenEncoder(volume=volume, emb_dim=int(emb_dim), seed=0)
    tokens = enc.encode(cells)
    gen = generator_fn(tokens)
    gen = enforce_pp_contract(gen, tokens, k_max=int(k_max))
    gen = attach_posthoc_citations(gen, tokens, k_max=int(k_max), positive_only=True, overwrite_empty_only=True)
    gen = enforce_pp_contract(gen, tokens, k_max=int(k_max))
    return tokens, gen


@dataclass(frozen=True)
class RunConfig:
    dataset_type: str = "synthetic"  # synthetic | manifest | path
    manifest_path: str = ""
    split: str = "test"
    volume_path: str = ""
    resize_shape: Tuple[int, int, int] = (64, 64, 64)

    n_samples: int = 100
    data_seed: int = 0
    seeds: Tuple[int, ...] = (0, 1, 2)

    budgets: Tuple[int, ...] = (64, 128, 256, 512)
    init_level: int = 1
    max_depth: int = 3
    l_min: int = 2
    k_max: int = 8
    max_steps_per_finding: int = 32
    emb_dim: int = 32
    topk_citations: int = 3

    pcg_backend: str = "toy"  # toy | llama2
    llama2_path: str = "/data/models/Llama-2-7b-chat-hf"
    llama2_device: str = "cuda"
    llama2_quant: str = "fp16"

    include_no_verify: bool = False
    include_split_dpo: bool = False
    dpo_ckpt: str = ""
    dpo_device: str = "cpu"
    require_blame: bool = True

    n_bootstrap: int = 10_000
    ci: float = 0.95
    output_dir: str = "./outputs/fig4_pareto_issue_budget"


def _iter_volumes(cfg: RunConfig) -> Iterable[Tuple[str, torch.Tensor, str]]:
    """Yield (sample_id, volume, data_revision)."""
    if cfg.dataset_type == "path":
        vol = load_volume(cfg.volume_path, seed=int(cfg.data_seed))
        yield ("volume_path", vol, "path")
        return

    if cfg.dataset_type == "manifest":
        if not cfg.manifest_path:
            raise ValueError("--manifest is required when --dataset-type=manifest")
        records = split_records(load_manifest(cfg.manifest_path), split=str(cfg.split))
        if not records:
            raise RuntimeError(f"No records for split={cfg.split!r} in manifest={cfg.manifest_path!r}")
        records = sorted(records, key=lambda r: r.scan_hash)
        for r in records[: max(0, int(cfg.n_samples))]:
            vol = load_volume(str(r.volume_path), seed=int(cfg.data_seed))
            yield (str(r.scan_hash), vol, "manifest")
        return

    # synthetic
    for i in range(max(0, int(cfg.n_samples))):
        seed = int(cfg.data_seed) + 10_000 * (0 if str(cfg.split) == "train" else 1) + int(i)
        vol, _ = create_synthetic_volume(shape=tuple(int(x) for x in cfg.resize_shape), n_lesions=3, seed=int(seed))
        yield (f"synthetic_{cfg.split}_{i}", vol, "synthetic")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig4: WeightedIssue vs token budget (pp.md v1.1).")
    ap.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest", "path"])
    ap.add_argument("--manifest", type=str, default="")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--volume-path", type=str, default="")
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])

    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--init-level", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--l-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--max-steps-per-finding", type=int, default=32)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--topk-citations", type=int, default=3)

    ap.add_argument("--pcg", type=str, default="toy", choices=["toy", "llama2"])
    ap.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--llama2-device", type=str, default="cuda")
    ap.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])

    ap.add_argument("--include-no-verify", action="store_true")
    ap.add_argument("--include-split-dpo", action="store_true")
    ap.add_argument("--dpo-ckpt", type=str, default="")
    ap.add_argument("--dpo-device", type=str, default="cpu")
    ap.add_argument("--require-blame", action="store_true")

    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--output-dir", type=str, default="./outputs/fig4_pareto_issue_budget")
    args = ap.parse_args()

    cfg = RunConfig(
        dataset_type=str(args.dataset_type),
        manifest_path=str(args.manifest),
        split=str(args.split),
        volume_path=str(args.volume_path),
        resize_shape=tuple(int(x) for x in args.resize_shape),
        n_samples=int(args.n_samples),
        data_seed=int(args.data_seed),
        seeds=tuple(int(x) for x in args.seeds),
        budgets=tuple(int(x) for x in args.budgets),
        init_level=int(args.init_level),
        max_depth=int(args.max_depth),
        l_min=int(args.l_min),
        k_max=int(args.k_max),
        max_steps_per_finding=int(args.max_steps_per_finding),
        emb_dim=int(args.emb_dim),
        topk_citations=int(args.topk_citations),
        pcg_backend=str(args.pcg),
        llama2_path=str(args.llama2_path),
        llama2_device=str(args.llama2_device),
        llama2_quant=str(args.llama2_quant),
        include_no_verify=bool(args.include_no_verify),
        include_split_dpo=bool(args.include_split_dpo),
        dpo_ckpt=str(args.dpo_ckpt),
        dpo_device=str(args.dpo_device),
        require_blame=bool(args.require_blame),
        n_bootstrap=int(args.n_bootstrap),
        ci=float(args.ci),
        output_dir=str(args.output_dir),
    )

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.pcg_backend == "llama2":
        pcg = Llama2PCG(
            Llama2PCGConfig(
                model_path=str(cfg.llama2_path),
                device=str(cfg.llama2_device),
                quantization=str(cfg.llama2_quant),
                temperature=0.0,
                topk_citations=int(cfg.topk_citations),
                max_new_tokens=512,
            )
        )
    else:
        pcg = None

    verifier = create_pp_verifier(l_min=int(cfg.l_min))
    weights = ProofWeights()

    dpo_policy = None
    if cfg.include_split_dpo:
        if not cfg.dpo_ckpt:
            raise ValueError("--dpo-ckpt is required when --include-split-dpo is set")
        dpo_policy = load_dpo_split_policy(str(cfg.dpo_ckpt), device=str(cfg.dpo_device), require_blame=bool(cfg.require_blame))

    methods: List[str] = []
    if cfg.include_no_verify:
        methods.append("Proof-NoVerify")
    methods.extend(["Proof+Verify-NoSplit", "+Split-Heuristic"])
    if cfg.include_split_dpo:
        methods.append("+Split-DPO")

    # raw[method][metric][budget] -> (S,N) matrix
    metric_keys = ["weighted_issue", "r1_no_citation", "r2_coarse_only", "r3_laterality_mismatch", "r4_bilateral_separation"]
    raw: Dict[str, Dict[str, Dict[int, List[List[float]]]]] = {m: {k: {} for k in metric_keys} for m in methods}

    volumes = list(_iter_volumes(cfg))
    if not volumes:
        raise RuntimeError("No volumes to run.")

    for budget in cfg.budgets:
        init_level_eff = _fit_init_level(init_level=int(cfg.init_level), budget_tokens=int(budget), max_depth=int(cfg.max_depth))
        for m in methods:
            for k in metric_keys:
                raw[m][k][int(budget)] = []

        for s in cfg.seeds:
            set_seed(int(s))
            if cfg.pcg_backend == "toy":
                pcg_seed = int(s)
                pcg_fn = ToyPCG(emb_dim=int(cfg.emb_dim), topk=int(cfg.topk_citations), seed=int(pcg_seed))
            else:
                if pcg is None:
                    raise RuntimeError("llama2 pcg not initialized")
                pcg_fn = pcg

            # Per-seed lists aligned with volumes order.
            per_seed: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in metric_keys} for m in methods}
            for _sample_id, vol, _rev in volumes:
                if "Proof-NoVerify" in methods:
                    toks0, gen0 = _proof_no_verify(
                        vol,
                        generator_fn=lambda toks: pcg_fn(toks),
                        emb_dim=int(cfg.emb_dim),
                        budget_tokens=int(budget),
                        init_level=int(init_level_eff),
                        max_depth=int(cfg.max_depth),
                        k_max=int(cfg.k_max),
                    )
                    met0 = compute_proof_metrics(gen0, toks0, verifier=verifier, l_min=int(cfg.l_min), weights=weights)
                    for k in metric_keys:
                        per_seed["Proof-NoVerify"][k].append(float(met0.get(k, 0.0)))

                res_no_split = run_provetok_agent(
                    volume=vol,
                    generator_fn=lambda toks: pcg_fn(toks),
                    verifier=verifier,
                    cfg=AgentConfig(
                        budget_tokens=int(budget),
                        emb_dim=int(cfg.emb_dim),
                        init_level=int(init_level_eff),
                        max_depth=int(init_level_eff),  # no split
                        max_steps_per_finding=int(cfg.max_steps_per_finding),
                        k_max_citations=int(cfg.k_max),
                        l_min=int(cfg.l_min),
                    ),
                    seed=int(s),
                )
                met1 = compute_proof_metrics(res_no_split.generation, res_no_split.tokens, verifier=verifier, l_min=int(cfg.l_min), weights=weights)
                for k in metric_keys:
                    per_seed["Proof+Verify-NoSplit"][k].append(float(met1.get(k, 0.0)))

                res_h = run_provetok_agent(
                    volume=vol,
                    generator_fn=lambda toks: pcg_fn(toks),
                    verifier=verifier,
                    cfg=AgentConfig(
                        budget_tokens=int(budget),
                        emb_dim=int(cfg.emb_dim),
                        init_level=int(init_level_eff),
                        max_depth=int(cfg.max_depth),
                        max_steps_per_finding=int(cfg.max_steps_per_finding),
                        k_max_citations=int(cfg.k_max),
                        l_min=int(cfg.l_min),
                    ),
                    seed=int(s),
                )
                met2 = compute_proof_metrics(res_h.generation, res_h.tokens, verifier=verifier, l_min=int(cfg.l_min), weights=weights)
                for k in metric_keys:
                    per_seed["+Split-Heuristic"][k].append(float(met2.get(k, 0.0)))

                if "+Split-DPO" in methods and dpo_policy is not None:
                    res_dpo = run_provetok_agent(
                        volume=vol,
                        generator_fn=lambda toks: pcg_fn(toks),
                        verifier=verifier,
                        cfg=AgentConfig(
                            budget_tokens=int(budget),
                            emb_dim=int(cfg.emb_dim),
                            init_level=int(init_level_eff),
                            max_depth=int(cfg.max_depth),
                            max_steps_per_finding=int(cfg.max_steps_per_finding),
                            k_max_citations=int(cfg.k_max),
                            l_min=int(cfg.l_min),
                        ),
                        seed=int(s),
                        split_cell_fn=lambda cand_cells, toks, issues: dpo_policy.pick_cell(cand_cells, toks, issues),
                    )
                    met3 = compute_proof_metrics(res_dpo.generation, res_dpo.tokens, verifier=verifier, l_min=int(cfg.l_min), weights=weights)
                    for k in metric_keys:
                        per_seed["+Split-DPO"][k].append(float(met3.get(k, 0.0)))

            # Append this seed's vector to raw matrices.
            for m in methods:
                for k in metric_keys:
                    raw[m][k][int(budget)].append([float(x) for x in per_seed[m][k]])

    # Aggregate CIs
    agg: Dict[str, Dict[str, List[Dict[str, float]]]] = {k: {m: [] for m in methods} for k in metric_keys}
    for bidx, budget in enumerate(cfg.budgets):
        for m in methods:
            for k in metric_keys:
                mat = np.asarray(raw[m][k][int(budget)], dtype=np.float64)  # (S,N)
                stable = hashlib.sha1(f"{m}:{k}".encode("utf-8")).digest()
                stable_seed = int(cfg.data_seed) + 1000 * bidx + int.from_bytes(stable[:4], "little", signed=False) % 997
                agg[k][m].append({"budget": int(budget), **_hier_mean_ci(mat, n_boot=int(cfg.n_bootstrap), seed=stable_seed, ci=float(cfg.ci))})

    # Meta
    repo_root = Path(__file__).resolve().parents[2]
    data_revision = "synthetic"
    split_manifest_path = ""
    if cfg.dataset_type == "manifest":
        data_revision, split_manifest_path = try_manifest_revision(str(cfg.manifest_path))

    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(cfg.data_seed),
        config=asdict(cfg),
        rule_set_version=str(PP_RULE_SET_VERSION),
        schema_version="pp_v1.1",
        taxonomy_version="pp_v1.1",
        data_revision=str(data_revision),
        split_manifest_path=str(split_manifest_path),
    )

    out_json = {
        "meta": meta.to_dict(),
        "methods": list(methods),
        "metric_keys": list(metric_keys),
        "raw": raw,
        "agg": agg,
    }
    save_results_json(out_json, str(out_dir / "fig4_pareto_issue_budget.json"))

    # CSV (WeightedIssue)
    with (out_dir / "fig4_pareto_issue_budget.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["budget", "method", "mean", "ci_low", "ci_high"])
        for m in methods:
            for rec in agg["weighted_issue"][m]:
                w.writerow([int(rec["budget"]), str(m), float(rec["mean"]), float(rec["ci_low"]), float(rec["ci_high"])])


if __name__ == "__main__":
    main()
