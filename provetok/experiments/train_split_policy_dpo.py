"""Train a lightweight split policy via one-step verifier-derived DPO (pp.md §5.2).

This is a scaffold intended for reproducible experimentation:
- generates preference pairs via one-step lookahead (split → rewrite → verify)
- trains a pointer-style action scorer with the discrete-action DPO objective

Default data source is synthetic volumes (no dataset completion required).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..bet.split_policy_dpo import SplitFeaturizerConfig, SplitPolicyNet, dpo_loss, featurize_split_actions
from ..bet.tokenize import TokenEncoder
from ..grid.cells import Cell, root_cell, split
from ..pcg.finding_generator import SlicedGenerationFindingGenerator
from ..pcg.generator import ToyPCG
from ..pcg.text_contract import enforce_pp_contract
from ..types import Generation, Issue, Token
from ..verifier.pp_v1_1 import create_pp_verifier
from ..eval.metrics_proof import compute_proof_metrics
from .utils import create_synthetic_volume, save_results_json, set_seed


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    output_dir: str = "./outputs/train_split_policy_dpo"

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
    # 0 level_norm, 1 score, 2 unc, 6 sz_n, 7 sy_n, 8 sx_n, 9 crosses, 11 is_r2, 12 is_r3, 13 is_r4
    level = feats[:, 0]
    score = feats[:, 1]
    unc = feats[:, 2]
    vol = feats[:, 6] * feats[:, 7] * feats[:, 8]
    crosses = feats[:, 9]
    is_r2 = feats[:, 11]
    is_r3 = feats[:, 12]
    is_r4 = feats[:, 13]
    # Align with pp.md §5.1 priorities: R2 coarse-first, R3 boundary-first, tie-break by volume.
    return 2.0 * is_r2 * (1.0 - level) + 2.0 * is_r3 * crosses + 0.5 * is_r4 + 0.25 * vol + 0.5 * score + 0.25 * unc


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


def _gen_pref_samples(cfg: TrainConfig) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(int(cfg.seed))
    verifier = create_pp_verifier(l_min=int(cfg.l_min))
    pcg = ToyPCG(emb_dim=int(cfg.emb_dim), topk=int(cfg.topk_citations), seed=int(cfg.seed))
    fg = SlicedGenerationFindingGenerator(
        lambda toks: pcg(toks),
        k_max=int(cfg.k_max_citations),
        l_min=int(cfg.l_min),
        verifier=verifier,
    )

    samples: List[Dict[str, Any]] = []
    attempts = 0
    while len(samples) < int(cfg.n_states) and attempts < int(cfg.n_states) * 10:
        attempts += 1
        vol_seed = int(cfg.seed) + attempts * 17
        vol, _ = create_synthetic_volume(shape=tuple(int(x) for x in cfg.vol_shape), n_lesions=int(cfg.n_lesions), seed=int(vol_seed))
        cells = _init_cells(init_level=int(cfg.init_level), budget_tokens=int(cfg.budget_tokens), max_depth=int(cfg.max_depth))
        enc = TokenEncoder(volume=vol, emb_dim=int(cfg.emb_dim), seed=int(cfg.seed))
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

        # Evaluate one-step lookahead for each candidate action.
        action_scores: List[float] = []
        for cand in candidate_cells:
            cells2 = [c for c in cells if c.id() != cand.id()] + split(cand)
            if (len(cells2)) > int(cfg.budget_tokens):
                action_scores.append(float("inf"))
                continue
            tokens2 = enc.encode(cells2)
            # pp.md one-step lookahead: split → rewrite current finding → verify.
            line2 = fg.rewrite_one(tokens2, finding_idx=int(picked_idx), issues=picked_issues)
            gen_one2 = enforce_pp_contract(line2.to_single_generation(), tokens2, k_max=int(cfg.k_max_citations))
            m = compute_proof_metrics(gen_one2, tokens2, l_min=int(cfg.l_min))
            action_scores.append(float(m.get("weighted_issue", 0.0)))

        best = int(np.argmin(np.asarray(action_scores, dtype=np.float64)))
        worst = int(np.argmax(np.asarray(action_scores, dtype=np.float64)))
        if best == worst:
            continue

        feats, ordered = featurize_split_actions(
            cells=candidate_cells,
            tokens=tokens,
            issues=picked_issues,
            cfg=SplitFeaturizerConfig(max_depth=int(cfg.max_depth)),
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
                "action_scores": [float(x) for x in action_scores],
            }
        )

    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a split policy via verifier-derived DPO (synthetic scaffold).")
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
