"""Figure 5: qualitative case study bundle (pp.md v1.1).

Produces a per-case artifact with:
- Findings lines + citations
- Proof-object (token table + citations_ref)
- Verifier issues before/after the agent loop
- A 2-panel axial overlay (coarse vs refined citations)

This is a scaffold: it runs on synthetic volumes by default and can optionally
consume a manifest (no dataset completion required).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..agent.loop import AgentConfig, run_provetok_agent
from ..bet.split_policy_dpo import load_dpo_split_policy
from ..bet.tokenize import TokenEncoder
from ..grid.cells import Cell, parse_cell_id, root_cell
from ..pcg.generator import ToyPCG
from ..pcg.llama2_pcg import Llama2PCG, Llama2PCGConfig
from ..pcg.proof_object import build_proof_object_from_generation
from ..pcg.text_contract import enforce_pp_contract, render_findings, render_impression
from ..eval.metrics_proof import attach_posthoc_citations, compute_proof_metrics
from ..types import Generation, Token
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.pp_contract import check_impression_no_new_cite
from ..verifier.pp_v1_1 import PP_RULE_SET_VERSION, create_pp_verifier
from .utils import create_synthetic_volume, save_results_json, set_seed
from ..data.io import load_volume
from ..data.manifest_schema import ManifestRecord, load_manifest, split_records
from ..eval.metrics_grounding import tokens_to_mask


def _fit_init_level(*, init_level: int, budget_tokens: int, max_depth: int) -> int:
    init_level = int(min(max_depth, max(0, int(init_level))))
    while init_level > 0:
        n = 2 ** int(init_level)
        if int(n * n * n) <= int(budget_tokens):
            break
        init_level -= 1
    return int(init_level)


def _init_cells(*, init_level: int, budget_tokens: int, max_depth: int) -> List[Cell]:
    init_level_eff = _fit_init_level(init_level=int(init_level), budget_tokens=int(budget_tokens), max_depth=int(max_depth))
    if init_level_eff <= 0:
        return [root_cell()]
    n = 2 ** int(init_level_eff)
    return [Cell(level=init_level_eff, ix=ix, iy=iy, iz=iz) for ix in range(n) for iy in range(n) for iz in range(n)]


def _resize_volume(vol: torch.Tensor, *, resize_shape: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(int(x) for x in vol.shape) == tuple(int(x) for x in resize_shape):
        return vol
    x = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    y = F.interpolate(x, size=tuple(int(x) for x in resize_shape), mode="trilinear", align_corners=False)
    return y[0, 0]


def _pick_manifest_record(records: Sequence[ManifestRecord], *, sample_id: str, idx: int) -> ManifestRecord:
    pool = sorted(list(records), key=lambda r: r.scan_hash)
    if sample_id:
        wanted = str(sample_id)
        hit = next((r for r in pool if str(r.scan_hash) == wanted), None)
        if hit is None:
            raise RuntimeError(f"--sample-id={wanted!r} not found in split records")
        return hit
    if not pool:
        raise RuntimeError("Empty manifest split records")
    return pool[int(max(0, min(int(idx), len(pool) - 1)))]


def _cell_ancestor(cell: Cell, *, level: int) -> Cell:
    """Return the ancestor of `cell` at `level` by truncating the octree path."""
    level = int(level)
    if int(cell.level) <= level:
        return cell
    shift = int(cell.level) - level
    return Cell(level=level, ix=int(cell.ix) >> shift, iy=int(cell.iy) >> shift, iz=int(cell.iz) >> shift)


def _select_target_finding_idx(trace: Sequence[Any], *, default: int = 0) -> int:
    for step in trace or []:
        try:
            if str(getattr(step, "action", "")) == "split":
                return int(getattr(step, "finding_idx", default))
        except Exception:
            continue
    return int(default)


def _save_overlay_png(
    *,
    out_path: Path,
    volume: torch.Tensor,
    coarse_mask: np.ndarray,
    fine_mask: np.ndarray,
    title_left: str,
    title_right: str,
    z: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Plotting requires matplotlib (see requirements.txt).") from e

    vol = volume.detach().cpu().float().numpy()
    vol = np.clip(vol, -1000.0, 1000.0) / 1000.0
    z = int(max(0, min(int(z), int(vol.shape[0] - 1))))

    img = vol[z]
    cm = coarse_mask[z] if isinstance(coarse_mask, np.ndarray) and coarse_mask.ndim == 3 else np.zeros_like(img, dtype=bool)
    fm = fine_mask[z] if isinstance(fine_mask, np.ndarray) and fine_mask.ndim == 3 else np.zeros_like(img, dtype=bool)

    fig = plt.figure(figsize=(10, 5), dpi=180)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img, cmap="gray", vmin=-1.0, vmax=1.0)
    if cm.any():
        ax1.imshow(np.ma.masked_where(~cm, cm), cmap="Blues", alpha=0.25)
    ax1.set_title(title_left)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img, cmap="gray", vmin=-1.0, vmax=1.0)
    if fm.any():
        ax2.imshow(np.ma.masked_where(~fm, fm), cmap="Blues", alpha=0.25)
    ax2.set_title(title_right)
    ax2.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class RunConfig:
    dataset_type: str = "synthetic"  # synthetic | manifest | path
    manifest_path: str = ""
    split: str = "test"
    sample_id: str = ""
    sample_idx: int = 0
    volume_path: str = ""
    resize_shape: Tuple[int, int, int] = (64, 64, 64)

    budget_tokens: int = 256
    init_level: int = 1
    max_depth: int = 3
    l_min: int = 2
    k_max: int = 8
    max_steps_per_finding: int = 32
    emb_dim: int = 32
    topk_citations: int = 3

    method: str = "heuristic"  # heuristic | dpo
    dpo_ckpt: str = ""
    dpo_device: str = "cpu"
    require_blame: bool = True

    pcg_backend: str = "toy"  # toy | llama2
    seed: int = 0
    data_seed: int = 0
    llama2_path: str = "/data/models/Llama-2-7b-chat-hf"
    llama2_device: str = "cuda"
    llama2_quant: str = "fp16"

    output_dir: str = "./outputs/fig5_case_study"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig5: case study bundle (pp.md v1.1).")
    ap.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest", "path"])
    ap.add_argument("--manifest", type=str, default="")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--sample-id", type=str, default="")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--volume-path", type=str, default="")
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])

    ap.add_argument("--budget-tokens", type=int, default=256)
    ap.add_argument("--init-level", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--l-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--max-steps-per-finding", type=int, default=32)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--topk-citations", type=int, default=3)

    ap.add_argument("--method", type=str, default="heuristic", choices=["heuristic", "dpo"])
    ap.add_argument("--dpo-ckpt", type=str, default="")
    ap.add_argument("--dpo-device", type=str, default="cpu")
    ap.add_argument("--require-blame", action="store_true")

    ap.add_argument("--pcg", type=str, default="toy", choices=["toy", "llama2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--llama2-device", type=str, default="cuda")
    ap.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])

    ap.add_argument("--output-dir", type=str, default="./outputs/fig5_case_study")
    args = ap.parse_args()

    cfg = RunConfig(
        dataset_type=str(args.dataset_type),
        manifest_path=str(args.manifest),
        split=str(args.split),
        sample_id=str(args.sample_id),
        sample_idx=int(args.sample_idx),
        volume_path=str(args.volume_path),
        resize_shape=tuple(int(x) for x in args.resize_shape),
        budget_tokens=int(args.budget_tokens),
        init_level=int(args.init_level),
        max_depth=int(args.max_depth),
        l_min=int(args.l_min),
        k_max=int(args.k_max),
        max_steps_per_finding=int(args.max_steps_per_finding),
        emb_dim=int(args.emb_dim),
        topk_citations=int(args.topk_citations),
        method=str(args.method),
        dpo_ckpt=str(args.dpo_ckpt),
        dpo_device=str(args.dpo_device),
        require_blame=bool(args.require_blame),
        pcg_backend=str(args.pcg),
        seed=int(args.seed),
        data_seed=int(args.data_seed),
        llama2_path=str(args.llama2_path),
        llama2_device=str(args.llama2_device),
        llama2_quant=str(args.llama2_quant),
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

    set_seed(int(cfg.seed))

    # Load one volume
    sample_id = "synthetic"
    data_revision = "synthetic"
    split_manifest_path = ""
    if cfg.dataset_type == "path":
        volume = load_volume(cfg.volume_path, seed=int(cfg.data_seed))
        volume = _resize_volume(volume, resize_shape=cfg.resize_shape)
        sample_id = "volume_path"
        data_revision = "path"
    elif cfg.dataset_type == "manifest":
        if not cfg.manifest_path:
            raise ValueError("--manifest is required for dataset-type=manifest")
        records = split_records(load_manifest(cfg.manifest_path), split=str(cfg.split))
        rec = _pick_manifest_record(records, sample_id=str(cfg.sample_id), idx=int(cfg.sample_idx))
        sample_id = str(rec.scan_hash)
        volume = load_volume(str(rec.volume_path), seed=int(cfg.data_seed))
        volume = _resize_volume(volume, resize_shape=cfg.resize_shape)
        data_revision, split_manifest_path = try_manifest_revision(str(cfg.manifest_path))
        data_revision = str(data_revision)
    else:
        seed0 = int(cfg.data_seed) + 10_000 * (0 if str(cfg.split) == "train" else 1) + int(cfg.sample_idx)
        volume, _ = create_synthetic_volume(shape=cfg.resize_shape, n_lesions=3, seed=int(seed0))
        sample_id = f"synthetic_{cfg.split}_{int(cfg.sample_idx)}"
        data_revision = "synthetic"

    # PCG backend
    if cfg.pcg_backend == "llama2":
        pcg_fn = Llama2PCG(
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
        pcg_fn = ToyPCG(emb_dim=int(cfg.emb_dim), topk=int(cfg.topk_citations), seed=int(cfg.seed))

    verifier = create_pp_verifier(l_min=int(cfg.l_min))

    init_level_eff = _fit_init_level(init_level=int(cfg.init_level), budget_tokens=int(cfg.budget_tokens), max_depth=int(cfg.max_depth))
    init_cells = _init_cells(init_level=int(init_level_eff), budget_tokens=int(cfg.budget_tokens), max_depth=int(cfg.max_depth))
    enc0 = TokenEncoder(volume=volume, emb_dim=int(cfg.emb_dim), seed=int(cfg.seed))
    tokens0 = enc0.encode(init_cells)

    # Before: single-pass generation at coarse tokens
    gen_before = pcg_fn(tokens0)
    gen_before = enforce_pp_contract(gen_before, tokens0, k_max=int(cfg.k_max))
    gen_before = attach_posthoc_citations(gen_before, tokens0, k_max=int(cfg.k_max), positive_only=True, overwrite_empty_only=True)
    gen_before = enforce_pp_contract(gen_before, tokens0, k_max=int(cfg.k_max))
    before_lines = render_findings(gen_before, k_max=int(cfg.k_max))
    before_imp = render_impression(before_lines)
    before_metrics = compute_proof_metrics(gen_before, tokens0, verifier=verifier, l_min=int(cfg.l_min))
    before_issues = verifier.verify(gen_before, tokens0)

    # After: agent loop
    split_cell_fn = None
    if cfg.method == "dpo":
        if not cfg.dpo_ckpt:
            raise ValueError("--dpo-ckpt is required for --method=dpo")
        policy = load_dpo_split_policy(str(cfg.dpo_ckpt), device=str(cfg.dpo_device), require_blame=bool(cfg.require_blame))
        split_cell_fn = lambda cand_cells, toks, issues: policy.pick_cell(cand_cells, toks, issues)

    agent_res = run_provetok_agent(
        volume=volume,
        generator_fn=lambda toks: pcg_fn(toks),
        verifier=verifier,
        cfg=AgentConfig(
            budget_tokens=int(cfg.budget_tokens),
            emb_dim=int(cfg.emb_dim),
            init_level=int(init_level_eff),
            max_depth=int(cfg.max_depth),
            max_steps_per_finding=int(cfg.max_steps_per_finding),
            k_max_citations=int(cfg.k_max),
            l_min=int(cfg.l_min),
        ),
        seed=int(cfg.seed),
        split_cell_fn=split_cell_fn,
    )

    after_metrics = compute_proof_metrics(agent_res.generation, agent_res.tokens, verifier=verifier, l_min=int(cfg.l_min))
    proof_obj = build_proof_object_from_generation(agent_res.generation, agent_res.tokens, k_max=int(cfg.k_max), impression=str(agent_res.impression))
    r0 = check_impression_no_new_cite(findings_lines=agent_res.findings_lines, impression=str(agent_res.impression))

    # Visualization: pick one finding index (first split if any) and show coarse vs refined citations.
    target_idx = _select_target_finding_idx(agent_res.trace, default=0)
    final_cites = list((agent_res.generation.citations or {}).get(int(target_idx), []) or [])

    # Fine mask: cited tokens at final state.
    tok_by_id = {int(t.token_id): t for t in agent_res.tokens}
    cited_tokens_fine = [tok_by_id[int(tid)] for tid in final_cites if int(tid) in tok_by_id]
    fine_mask = tokens_to_mask(cited_tokens_fine, volume_shape=tuple(int(x) for x in volume.shape))

    # Coarse mask: ancestors at init_level_eff.
    tok0_by_cell = {str(t.cell_id): t for t in tokens0}
    coarse_cells: Dict[str, Token] = {}
    for t in cited_tokens_fine:
        c = parse_cell_id(str(t.cell_id))
        if c is None:
            continue
        anc = _cell_ancestor(c, level=int(init_level_eff))
        anc_id = anc.id()
        tok0 = tok0_by_cell.get(str(anc_id))
        if tok0 is not None:
            coarse_cells[str(anc_id)] = tok0
    coarse_mask = tokens_to_mask(list(coarse_cells.values()), volume_shape=tuple(int(x) for x in volume.shape))

    union = np.logical_or(coarse_mask, fine_mask)
    if union.any():
        z = int(np.argmax(union.sum(axis=(1, 2))))
    else:
        z = int(volume.shape[0] // 2)

    overlay_path = out_dir / "fig5_overlay.png"
    _save_overlay_png(
        out_path=overlay_path,
        volume=volume,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
        title_left=f"Coarse (level={init_level_eff}) cites for finding[{target_idx}]",
        title_right=f"Refined (final leaves) cites for finding[{target_idx}]",
        z=int(z),
    )

    # Meta + bundle
    repo_root = Path(__file__).resolve().parents[2]
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(cfg.seed),
        config=asdict(cfg),
        rule_set_version=str(PP_RULE_SET_VERSION),
        schema_version="pp_v1.1",
        taxonomy_version="pp_v1.1",
        data_revision=str(data_revision),
        split_manifest_path=str(split_manifest_path),
    )

    bundle: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "sample_id": str(sample_id),
        "target_finding_idx": int(target_idx),
        "before": {
            "findings_lines": list(before_lines),
            "impression": str(before_imp),
            "metrics": {k: float(v) for k, v in before_metrics.items()},
            "issues": [i.__dict__ for i in before_issues],
        },
        "after": {
            "findings_lines": list(agent_res.findings_lines),
            "impression": str(agent_res.impression),
            "metrics": {k: float(v) for k, v in after_metrics.items()},
            "issues": [i.__dict__ for i in agent_res.issues],
            "contract_issues": [] if r0 is None else [r0.__dict__],
            "trace": [getattr(t, "__dict__", {}) for t in agent_res.trace],
            "proof_object": proof_obj.to_dict(),
        },
        "viz": {
            "overlay_png": str(overlay_path),
            "axial_z": int(z),
        },
    }

    save_results_json(bundle, str(out_dir / "fig5_case_study_bundle.json"))


if __name__ == "__main__":
    main()

