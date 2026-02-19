"""Train a lightweight split policy via one-step verifier-derived DPO (pp.md §5.2).

This script supports:
- synthetic volumes (default; reproducible without datasets), and
- manifest-driven real volumes (dataset_type=manifest).

It generates preference pairs via one-step lookahead (split → rewrite → verify),
then trains a pointer-style action scorer with the discrete-action DPO objective.
"""

from __future__ import annotations

import argparse
import time
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..bet.split_policy_dpo import SplitFeaturizerConfig, SplitPolicyNet, dpo_loss, featurize_split_actions
from ..bet.tokenize import TokenEncoder
from ..data import ManifestDataset
from ..grid.cells import Cell, root_cell, split
from ..pcg.generator import ToyPCG
from ..pcg.text_contract import enforce_pp_contract
from ..types import Generation, Issue
from ..verifier.pp_v1_1 import create_pp_verifier
from ..eval.metrics_proof import compute_proof_metrics
from .utils import create_synthetic_volume, save_results_json, set_seed


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    output_dir: str = "./outputs/train_split_policy_dpo"

    # Data source
    dataset_type: str = "synthetic"  # "synthetic" | "manifest"
    manifest_path: str = ""
    split: str = "train"
    max_samples: int = 0
    resize_shape: Tuple[int, int, int] = (64, 64, 64)

    # Synthetic data
    vol_shape: Tuple[int, int, int] = (64, 64, 64)
    n_lesions: int = 3

    # Preference data
    n_states: int = 256
    n_candidates: int = 8
    init_level: int = 1
    max_depth: int = 3
    budget_tokens: int = 256
    l_min: int = 2
    k_max_citations: int = 8

    # Toy generator
    emb_dim: int = 32
    topk_citations: int = 3

    # DPO
    beta: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 3
    batch_size: int = 16
    hidden_dim: int = 64

    device: str = "cpu"


def _slice_generation(gen: Generation, *, frame_idx: int) -> Generation:
    if not gen.frames:
        return Generation(frames=[], citations={}, q={}, refusal={}, text="")
    i = int(frame_idx)
    if i < 0 or i >= len(gen.frames):
        return Generation(frames=[], citations={}, q={}, refusal={}, text="")
    fr = gen.frames[i]
    cites = {0: list((gen.citations or {}).get(i, []) or [])}
    q = {0: float((gen.q or {}).get(i, getattr(fr, "confidence", 0.5)))}
    refusal = {0: bool((gen.refusal or {}).get(i, False))}
    cites_ref = {0: list((gen.citations_ref or {}).get(i, []) or [])} if getattr(gen, "citations_ref", None) else {}
    return Generation(frames=[fr], citations=cites, q=q, refusal=refusal, citations_ref=cites_ref, text="")


def _collect_blame_refs(issues: List[Issue]) -> List[str]:
    refs: List[str] = []
    for iss in issues:
        trace = getattr(iss, "evidence_trace", {}) or {}
        for r in (trace.get("blame_refs", []) or []):
            if isinstance(r, str) and r:
                refs.append(str(r))
    # Stable unique ordering
    seen = set()
    out = []
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _ref_logits_from_features(feats: torch.Tensor) -> torch.Tensor:
    """Heuristic reference policy logits (deterministic, auditable)."""
    # Feature indices (see featurize_split_actions):
    # 0 level_norm, 1 score, 2 unc, 9 crosses, 11 is_r2, 12 is_r3, 13 is_r4
    level = feats[:, 0]
    score = feats[:, 1]
    unc = feats[:, 2]
    crosses = feats[:, 9]
    is_r2 = feats[:, 11]
    is_r3 = feats[:, 12]
    is_r4 = feats[:, 13]
    return 1.5 * unc + 1.0 * score + 2.0 * is_r2 * (1.0 - level) + 1.0 * is_r3 * crosses + 0.5 * is_r4


def _init_cells(*, init_level: int, budget_tokens: int, max_depth: int) -> List[Cell]:
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


def _load_manifest_dataset(cfg: TrainConfig) -> ManifestDataset:
    if not cfg.manifest_path:
        raise SystemExit("--manifest is required when --dataset-type=manifest")
    return ManifestDataset(
        manifest_path=str(cfg.manifest_path),
        split=str(cfg.split),
        max_samples=int(cfg.max_samples),
        resize_shape=tuple(int(x) for x in cfg.resize_shape),
        seed=int(cfg.seed),
    )


def _gen_pref_samples(cfg: TrainConfig) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(int(cfg.seed))
    t_start = time.perf_counter()
    verifier = create_pp_verifier(l_min=int(cfg.l_min))
    pcg = ToyPCG(emb_dim=int(cfg.emb_dim), topk=int(cfg.topk_citations), seed=int(cfg.seed))

    ds: Optional[ManifestDataset] = None
    if str(cfg.dataset_type) == "manifest":
        ds = _load_manifest_dataset(cfg)
        if len(ds) <= 0:
            raise SystemExit("ManifestDataset is empty (check --manifest/--split/--max-samples).")

    samples: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = int(cfg.n_states) * 20

    while len(samples) < int(cfg.n_states) and attempts < max_attempts:
        attempts += 1

        affine_zyx = None
        if ds is not None:
            idx = int(rng.randint(0, len(ds)))
            row = ds[int(idx)]
            vol = row["volume"]
            affine_zyx = row.get("affine_zyx", None)
        else:
            vol_seed = int(cfg.seed) + attempts * 17
            vol, _ = create_synthetic_volume(shape=tuple(int(x) for x in cfg.vol_shape), n_lesions=int(cfg.n_lesions), seed=int(vol_seed))

        cells = _init_cells(init_level=int(cfg.init_level), budget_tokens=int(cfg.budget_tokens), max_depth=int(cfg.max_depth))
        enc = TokenEncoder(volume=vol, emb_dim=int(cfg.emb_dim), seed=int(cfg.seed), affine_zyx=affine_zyx)
        tokens = enc.encode(cells)
        gen_full = pcg(tokens)
        if not gen_full.frames:
            continue

        # Pick a frame index that yields at least one verifier issue.
        frame_order = list(range(len(gen_full.frames)))
        rng.shuffle(frame_order)
        picked_idx: Optional[int] = None
        picked_issues: List[Issue] = []
        for idx in frame_order:
            gen_one = enforce_pp_contract(_slice_generation(gen_full, frame_idx=int(idx)), tokens, k_max=int(cfg.k_max_citations))
            issues = verifier.verify(gen_one, tokens)
            if issues:
                picked_idx = int(idx)
                picked_issues = list(issues)
                break
        if picked_idx is None:
            continue

        blame_refs = _collect_blame_refs(picked_issues)
        if not blame_refs:
            continue

        tok_by_ref = {str(getattr(t, "ref", t.cell_id)): t for t in tokens}
        cell_by_id = {c.id(): c for c in cells}
        candidate_cells: List[Cell] = []
        for ref in blame_refs:
            t = tok_by_ref.get(str(ref))
            if t is None:
                continue
            c = cell_by_id.get(str(t.cell_id))
            if c is None:
                continue
            if int(c.level) >= int(cfg.max_depth):
                continue
            candidate_cells.append(c)

        # Dedup + deterministic order.
        uniq = {c.id(): c for c in candidate_cells}
        candidate_cells = sorted(list(uniq.values()), key=lambda c: c.id())
        if not candidate_cells:
            continue

        if len(candidate_cells) > int(cfg.n_candidates):
            # Deterministic subset via stable RNG choice over sorted list.
            idxs = rng.choice(len(candidate_cells), size=int(cfg.n_candidates), replace=False).tolist()
            candidate_cells = [candidate_cells[int(i)] for i in sorted(idxs)]

        # Base metric for the picked state (before any split).
        gen_one_base = enforce_pp_contract(_slice_generation(gen_full, frame_idx=int(picked_idx)), tokens, k_max=int(cfg.k_max_citations))
        base_m = compute_proof_metrics(gen_one_base, tokens, l_min=int(cfg.l_min))
        base_w = float(base_m.get("weighted_issue", 0.0))

        # Evaluate one-step lookahead for each candidate action.
        action_after_w: List[float] = []
        action_delta_w: List[float] = []
        action_n_cites: List[int] = []

        for cand in candidate_cells:
            cells2 = [c for c in cells if c.id() != cand.id()] + split(cand)
            if (len(cells2)) > int(cfg.budget_tokens):
                action_after_w.append(float("inf"))
                action_delta_w.append(float("-inf"))
                action_n_cites.append(0)
                continue

            tokens2 = enc.encode(cells2)
            gen2 = pcg(tokens2)
            gen_one2 = enforce_pp_contract(_slice_generation(gen2, frame_idx=int(picked_idx)), tokens2, k_max=int(cfg.k_max_citations))
            m2 = compute_proof_metrics(gen_one2, tokens2, l_min=int(cfg.l_min))
            w2 = float(m2.get("weighted_issue", 0.0))
            action_after_w.append(w2)
            action_delta_w.append(float(base_w - w2))
            n_c = len(list((gen_one2.citations or {}).get(0, []) or []))
            action_n_cites.append(int(n_c))

        # Choose preferred action: maximize issue reduction; tie-break by lower post-split issue,
        # then fewer citations.
        idxs = list(range(len(candidate_cells)))
        best = min(idxs, key=lambda i: (-action_delta_w[i], action_after_w[i], action_n_cites[i], i))
        worst = min(idxs, key=lambda i: (action_delta_w[i], -action_after_w[i], -action_n_cites[i], i))
        if int(best) == int(worst):
            continue

        budget_frac = float(len(cells)) / float(max(1, int(cfg.budget_tokens)))
        step_frac = 0.0
        feats, ordered = featurize_split_actions(
            cells=candidate_cells,
            tokens=tokens,
            issues=picked_issues,
            cfg=SplitFeaturizerConfig(max_depth=int(cfg.max_depth)),
            budget_frac=float(budget_frac),
            step_frac=float(step_frac),
        )
        if not ordered or feats.shape[0] != len(candidate_cells):
            continue

        # Align indices: featurizer sorts by cell.id, which matches candidate_cells sort above.
        chosen_idx = int(best)
        rejected_idx = int(worst)
        ref_logits = _ref_logits_from_features(feats).detach().cpu()

        samples.append(
            {
                "features": feats.detach().cpu(),
                "ref_logits": ref_logits,
                "chosen_idx": chosen_idx,
                "rejected_idx": rejected_idx,
                "frame_idx": int(picked_idx),
                "base_weighted_issue": float(base_w),
                "action_after_weighted_issue": [float(x) for x in action_after_w],
                "action_delta_weighted_issue": [float(x) for x in action_delta_w],
                "action_n_citations": [int(x) for x in action_n_cites],
                "budget_frac": float(budget_frac),
                "step_frac": float(step_frac),
            }
        )
        
        # Progress logs: preference generation can be slow on real data; emit ETA.
        if len(samples) <= 5 or len(samples) % 25 == 0 or len(samples) == int(cfg.n_states):
            elapsed = max(1e-6, float(time.perf_counter() - t_start))
            rate = float(len(samples)) / elapsed
            eta_s = float(int(cfg.n_states) - len(samples)) / max(rate, 1e-6)
            print(
                f"[prefs] {len(samples)}/{int(cfg.n_states)} attempts={attempts} "
                f"rate={rate:.3f}/s eta={eta_s/60.0:.1f}m"
            )

    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a split policy via verifier-derived DPO (synthetic/manifest).")

    # Data
    ap.add_argument("--dataset-type", type=str, default="synthetic", choices=["synthetic", "manifest"])
    ap.add_argument("--manifest", type=str, default="", help="Manifest path when dataset-type=manifest")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Split for manifest dataset")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap for manifest records")
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64], help="Resize (D,H,W) for manifest volumes")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default="./outputs/train_split_policy_dpo")
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--n-states", type=int, default=256)
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--init-level", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--budget-tokens", type=int, default=256)
    ap.add_argument("--l-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)

    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--topk-citations", type=int, default=3)

    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=64)
    args = ap.parse_args()

    cfg = TrainConfig(
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        device=str(args.device),
        dataset_type=str(args.dataset_type),
        manifest_path=str(args.manifest),
        split=str(args.split),
        max_samples=int(args.max_samples),
        resize_shape=tuple(int(x) for x in args.resize_shape),
        n_states=int(args.n_states),
        n_candidates=int(args.n_candidates),
        init_level=int(args.init_level),
        max_depth=int(args.max_depth),
        budget_tokens=int(args.budget_tokens),
        l_min=int(args.l_min),
        k_max_citations=int(args.k_max),
        emb_dim=int(args.emb_dim),
        topk_citations=int(args.topk_citations),
        beta=float(args.beta),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
    )
    set_seed(int(cfg.seed))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefs = _gen_pref_samples(cfg)
    if not prefs:
        raise SystemExit("No preference samples generated (try increasing --n-states or adjusting --l-min/--init-level).")

    # Train policy
    model = SplitPolicyNet(in_dim=20, hidden_dim=int(cfg.hidden_dim)).to(torch.device(str(cfg.device)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    def batch_iter():
        idxs = np.arange(len(prefs))
        np.random.shuffle(idxs)
        bs = int(cfg.batch_size)
        for i in range(0, len(idxs), bs):
            yield idxs[i : i + bs].tolist()

    for ep in range(int(cfg.epochs)):
        losses = []
        for batch_ids in batch_iter():
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for j in batch_ids:
                row = prefs[int(j)]
                feats = row["features"].to(torch.device(str(cfg.device)))
                ref_logits = row["ref_logits"].to(torch.device(str(cfg.device)))
                chosen_idx = int(row["chosen_idx"])
                rejected_idx = int(row["rejected_idx"])
                pol_logits = model(feats)
                loss_j = dpo_loss(
                    policy_logits=pol_logits,
                    ref_logits=ref_logits,
                    chosen_idx=chosen_idx,
                    rejected_idx=rejected_idx,
                    beta=float(cfg.beta),
                )
                loss = loss + loss_j
            loss = loss / max(1, len(batch_ids))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        print(f"[epoch {ep+1}/{cfg.epochs}] loss={float(np.mean(losses)):.6f}")

    ckpt_path = out_dir / "split_policy_dpo.pt"
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt_path)
    torch.save(prefs, out_dir / "prefs.pt")

    save_results_json(
        {"config": asdict(cfg), "n_prefs": len(prefs), "ckpt": str(ckpt_path)},
        str(out_dir / "train_summary.json"),
    )


if __name__ == "__main__":
    # Reduce OpenMP oversubscription in multiprocess environments.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
