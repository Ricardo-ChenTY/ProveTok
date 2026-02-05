"""Train a lightweight token-level lesionness head on ReXGroundingCT-style masks.

Goal (C0004 helper): learn a score function `p(lesion | token)` so that ProveTok can
focus tokenization/citations on lesion regions, improving pixel-level grounding.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..bet.tokenize import encode_tokens
from ..data import make_dataloader
from ..eval.metrics_grounding import union_lesion_masks
from ..grid.cells import Cell
from ..models.lesionness_head import LesionnessHead, LesionnessHeadConfig, save_lesionness_head
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .utils import save_results_json, set_seed


@dataclass(frozen=True)
class TrainConfig:
    manifest_path: str
    resize_shape: Tuple[int, int, int] = (64, 64, 64)
    train_split: str = "train"
    val_split: str = "val"
    max_train_samples: int = 500
    max_val_samples: int = 200
    candidate_level: int = 4
    candidate_levels: Tuple[int, ...] = ()
    tokens_per_sample: int = 256
    pos_fraction: float = 0.5
    emb_dim: int = 32
    hidden_dim: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 5
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./outputs/train_lesionness_head"


def _candidate_cells(level: int) -> List[Cell]:
    n = 2 ** int(level)
    return [Cell(level=int(level), ix=ix, iy=iy, iz=iz) for ix in range(n) for iy in range(n) for iz in range(n)]


def _idx_to_cell(level: int, idx: int) -> Cell:
    n = 2 ** int(level)
    nn = n * n
    ix = int(idx // nn)
    rem = int(idx % nn)
    iy = int(rem // n)
    iz = int(rem % n)
    return Cell(level=int(level), ix=ix, iy=iy, iz=iz)


def _pos_indices_from_mask(lesion_union: np.ndarray, *, level: int) -> List[int]:
    """Map lesion voxels to unique cell indices at a given level.

    This is substantially faster than scanning all cells and slicing the mask.
    """
    if not isinstance(lesion_union, np.ndarray) or lesion_union.ndim != 3:
        return []
    coords = np.argwhere(lesion_union.astype(bool))
    if coords.size == 0:
        return []
    D, H, W = (int(x) for x in lesion_union.shape)
    n = 2 ** int(level)
    # coords are (z,y,x) in (D,H,W)
    iz = (coords[:, 0] * n) // max(D, 1)
    iy = (coords[:, 1] * n) // max(H, 1)
    ix = (coords[:, 2] * n) // max(W, 1)
    iz = np.clip(iz, 0, n - 1)
    iy = np.clip(iy, 0, n - 1)
    ix = np.clip(ix, 0, n - 1)
    idx = (ix * (n * n) + iy * n + iz).astype(np.int64)
    return sorted(set(int(x) for x in idx.tolist()))


def _sample_neg_indices(
    *,
    rng: np.random.RandomState,
    total: int,
    exclude: set[int],
    k: int,
) -> List[int]:
    out: List[int] = []
    used: set[int] = set()
    # Rejection sampling is fine here because exclude is small compared to total.
    while len(out) < int(k):
        cand = int(rng.randint(0, int(total)))
        if cand in exclude or cand in used:
            continue
        used.add(cand)
        out.append(cand)
    return out


def _collect_examples(cfg: TrainConfig, *, split: str, max_samples: int) -> Dict[str, Any]:
    dl = make_dataloader(
        {
            "dataset_type": "manifest",
            "manifest_path": cfg.manifest_path,
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": int(max_samples),
            "resize_shape": tuple(int(x) for x in cfg.resize_shape),
        },
        split=str(split),
    )

    levels = list(int(x) for x in (cfg.candidate_levels or (int(cfg.candidate_level),)))
    levels = sorted(set(levels))
    if not levels:
        raise ValueError("candidate_levels is empty")
    if int(cfg.tokens_per_sample) < len(levels):
        raise ValueError(f"tokens_per_sample={cfg.tokens_per_sample} must be >= number of levels={len(levels)}")

    tokens_per_level: Dict[int, int] = {}
    base = int(cfg.tokens_per_sample) // len(levels)
    rem = int(cfg.tokens_per_sample) % len(levels)
    for i, lvl in enumerate(levels):
        tokens_per_level[int(lvl)] = int(base + (1 if i < rem else 0))

    X: List[torch.Tensor] = []
    y: List[float] = []
    token_levels: List[int] = []
    stats: Dict[str, Any] = {
        "num_samples": 0,
        "tokens_per_sample": int(cfg.tokens_per_sample),
        "levels": [int(x) for x in levels],
        "tokens_per_level": {str(k): int(v) for k, v in tokens_per_level.items()},
        "pos_counts": [],
        "neg_counts": [],
        "pos_counts_by_level": {str(lvl): [] for lvl in levels},
        "neg_counts_by_level": {str(lvl): [] for lvl in levels},
    }

    for sample_idx, batch in enumerate(dl):
        volume = batch["volume"][0]  # (D,H,W)
        lesion_masks = batch.get("lesion_masks", [{}])[0] or {}
        vol_shape = tuple(int(x) for x in volume.shape)

        lesion_union = union_lesion_masks(lesion_masks, vol_shape)
        if isinstance(lesion_union, torch.Tensor):
            lesion_union = lesion_union.detach().cpu().numpy()

        rng = np.random.RandomState(int(cfg.seed) + 10_000 * (0 if split == "train" else 1) + int(sample_idx))
        chosen_cells: List[Cell] = []
        label_by_cell_id: Dict[str, float] = {}
        n_pos_total = 0
        n_neg_total = 0

        for lvl in levels:
            per_lvl = int(tokens_per_level[int(lvl)])
            if per_lvl <= 0:
                continue
            pos_idx = _pos_indices_from_mask(lesion_union, level=int(lvl))
            pos_set = set(pos_idx)

            want_pos = int(round(float(per_lvl) * float(cfg.pos_fraction)))
            want_pos = max(0, min(want_pos, int(per_lvl)))
            n_pos = min(len(pos_idx), want_pos)
            n_neg = int(per_lvl) - int(n_pos)

            pos_sel = rng.choice(pos_idx, size=n_pos, replace=False).tolist() if n_pos > 0 else []
            n = 2 ** int(lvl)
            total_cells = int(n * n * n)
            neg_sel = _sample_neg_indices(rng=rng, total=total_cells, exclude=pos_set, k=n_neg) if n_neg > 0 else []

            chosen_idx = pos_sel + neg_sel
            rng.shuffle(chosen_idx)
            chosen_cells.extend([_idx_to_cell(int(lvl), int(i)) for i in chosen_idx])
            for i in pos_sel:
                label_by_cell_id[_idx_to_cell(int(lvl), int(i)).id()] = 1.0

            n_pos_total += int(n_pos)
            n_neg_total += int(n_neg)
            stats["pos_counts_by_level"][str(int(lvl))].append(int(n_pos))
            stats["neg_counts_by_level"][str(int(lvl))].append(int(n_neg))

        tokens = encode_tokens(volume, chosen_cells, emb_dim=int(cfg.emb_dim), seed=int(cfg.seed) + 123 * sample_idx)
        for t in tokens:
            X.append(t.embedding.float().cpu())
            y.append(float(label_by_cell_id.get(t.cell_id, 0.0)))
            token_levels.append(int(t.level))

        stats["num_samples"] += 1
        stats["pos_counts"].append(int(n_pos_total))
        stats["neg_counts"].append(int(n_neg_total))

    if not X:
        raise RuntimeError(f"No training examples collected for split={split!r}. Check masks/manifest.")

    X_t = torch.stack(X, dim=0)
    y_t = torch.tensor(y, dtype=torch.float32)
    levels_t = torch.tensor(token_levels, dtype=torch.int64)
    stats["num_tokens"] = int(y_t.numel())
    stats["pos_rate"] = float(y_t.mean().item())
    stats["level_counts"] = {str(int(lvl)): int((levels_t == int(lvl)).sum().item()) for lvl in levels}
    return {"X": X_t, "y": y_t, "levels": levels_t, "stats": stats}


@torch.no_grad()
def _eval_binary_classifier(model: LesionnessHead, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    tp = fp = tn = fn = 0
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(float(loss.item()))
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).to(dtype=torch.int64)
        y_int = (yb >= 0.5).to(dtype=torch.int64)
        tp += int(((pred == 1) & (y_int == 1)).sum().item())
        fp += int(((pred == 1) & (y_int == 0)).sum().item())
        tn += int(((pred == 0) & (y_int == 0)).sum().item())
        fn += int(((pred == 0) & (y_int == 1)).sum().item())

    denom = max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    acc = (tp + tn) / denom
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a token-level lesionness head on manifest masks (ReXGroundingCT).")
    ap.add_argument("--manifest", type=str, required=True, help="Manifest JSONL/CSV path (must contain mask_path in extras).")
    ap.add_argument("--smoke", action="store_true", help="Quick sanity run (small samples/epochs on CPU).")
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64], help="Resize (D,H,W) for volumes/masks.")
    ap.add_argument("--train-split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--val-split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--max-train-samples", type=int, default=500)
    ap.add_argument("--max-val-samples", type=int, default=200)
    ap.add_argument("--candidate-level", type=int, default=4, help="Grid level for candidate cells (2^L per axis).")
    ap.add_argument("--levels", type=int, nargs="+", default=None, help="Optional list of candidate levels to mix (overrides --candidate-level).")
    ap.add_argument("--tokens-per-sample", type=int, default=256, help="How many candidate tokens to sample per volume.")
    ap.add_argument("--pos-fraction", type=float, default=0.5, help="Target fraction of positive tokens per volume.")
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--output-dir", type=str, default="./outputs/train_lesionness_head")
    args = ap.parse_args()

    if bool(args.smoke):
        cfg = TrainConfig(
            manifest_path=str(args.manifest),
            resize_shape=(32, 32, 32),
            train_split=str(args.train_split),
            val_split=str(args.val_split),
            max_train_samples=min(10, int(args.max_train_samples)),
            max_val_samples=min(5, int(args.max_val_samples)),
            candidate_level=min(3, int(args.candidate_level)),
            candidate_levels=tuple(int(x) for x in (args.levels or [])),
            tokens_per_sample=min(64, int(args.tokens_per_sample)),
            pos_fraction=float(args.pos_fraction),
            emb_dim=int(args.emb_dim),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            batch_size=min(256, int(args.batch_size)),
            epochs=1,
            seed=int(args.seed),
            device="cpu",
            output_dir=str(args.output_dir),
        )
    else:
        cfg = TrainConfig(
            manifest_path=str(args.manifest),
            resize_shape=tuple(int(x) for x in args.resize_shape),
            train_split=str(args.train_split),
            val_split=str(args.val_split),
            max_train_samples=int(args.max_train_samples),
            max_val_samples=int(args.max_val_samples),
            candidate_level=int(args.candidate_level),
            candidate_levels=tuple(int(x) for x in (args.levels or [])),
            tokens_per_sample=int(args.tokens_per_sample),
            pos_fraction=float(args.pos_fraction),
            emb_dim=int(args.emb_dim),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            seed=int(args.seed),
            device=str(args.device),
            output_dir=str(args.output_dir),
        )

    set_seed(int(cfg.seed))
    os.makedirs(cfg.output_dir, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
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

    train = _collect_examples(cfg, split=cfg.train_split, max_samples=int(cfg.max_train_samples))
    val = _collect_examples(cfg, split=cfg.val_split, max_samples=int(cfg.max_val_samples))

    X_train: torch.Tensor = train["X"]
    y_train: torch.Tensor = train["y"]
    levels_train: torch.Tensor = train["levels"]
    X_val: torch.Tensor = val["X"]
    y_val: torch.Tensor = val["y"]
    levels_val: torch.Tensor = val["levels"]

    device = torch.device(cfg.device)
    model = LesionnessHead(LesionnessHeadConfig(emb_dim=cfg.emb_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)).to(device)

    # Handle class imbalance.
    pos = float(y_train.sum().item())
    neg = float(y_train.numel() - y_train.sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=int(cfg.batch_size), shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=int(cfg.batch_size), shuffle=False)

    history: List[Dict[str, Any]] = []
    for epoch in range(int(cfg.epochs)):
        model.train()
        losses: List[float] = []
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_eval = _eval_binary_classifier(model, DataLoader(TensorDataset(X_train, y_train), batch_size=int(cfg.batch_size)), device)
        val_eval = _eval_binary_classifier(model, val_dl, device)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "train_eval": train_eval,
                "val_eval": val_eval,
            }
        )
        print(f"[epoch {epoch}] train_loss={history[-1]['train_loss']:.4f} val_f1={val_eval['f1']:.4f} val_loss={val_eval['loss']:.4f}")

    # Per-level final evaluation (helps debug multi-level scoring issues).
    per_level_val: Dict[str, Dict[str, float]] = {}
    for lvl in sorted(set(int(x) for x in levels_val.detach().cpu().tolist())):
        mask = levels_val == int(lvl)
        if int(mask.sum().item()) == 0:
            continue
        dl_lvl = DataLoader(TensorDataset(X_val[mask], y_val[mask]), batch_size=int(cfg.batch_size), shuffle=False)
        per_level_val[str(int(lvl))] = _eval_binary_classifier(model, dl_lvl, device)

    # Save weights and report.
    weights_path = os.path.join(cfg.output_dir, "lesionness_head.pt")
    save_lesionness_head(weights_path, model.to("cpu"))

    report: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "config": asdict(cfg),
        "train_stats": train["stats"],
        "val_stats": val["stats"],
        "per_level_val_eval": per_level_val,
        "pos_weight": float(pos_weight.detach().cpu().item()),
        "history": history,
        "best_val_f1": float(max((h["val_eval"]["f1"] for h in history), default=0.0)),
        "weights_path": weights_path,
    }

    out_path = os.path.join(cfg.output_dir, "train_lesionness_head.json")
    save_results_json(report, out_path)
    print(f"Saved -> {out_path}")
    print(f"Weights -> {weights_path}")


if __name__ == "__main__":
    main()
