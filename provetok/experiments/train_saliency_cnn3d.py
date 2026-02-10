"""Train a lightweight 3D saliency CNN to predict union lesion masks.

Motivation: strengthen C0003 (optional) omega_perm counterfactual on real data
without using GT masks at inference time (train on train split, evaluate on test).
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

from ..data import make_dataloader
from ..eval.metrics_grounding import union_lesion_masks
from ..models.saliency_cnn3d import SaliencyCNN3D, SaliencyCNN3DConfig, save_saliency_cnn3d
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
    num_workers: int = 4
    clip_hu: Tuple[float, float] = (-1000.0, 1000.0)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 3
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    base_channels: int = 16
    num_layers: int = 4
    dropout: float = 0.0
    loss: str = "bce_pos_weight"
    pos_weight_max: float = 0.0  # 0 => no clamp
    dice_weight: float = 0.5
    eval_topk_ratio: float = 0.005
    output_dir: str = "./outputs/train_saliency_cnn3d"


def _preprocess_volume(vol: torch.Tensor, *, clip_hu: Tuple[float, float]) -> torch.Tensor:
    v = vol.float()
    lo, hi = (float(clip_hu[0]), float(clip_hu[1]))
    v = v.clamp(min=lo, max=hi)
    # Map HU to roughly [-1, 1] range for stability.
    v = v / max(abs(lo), abs(hi), 1.0)
    return v


def _collect(cfg: TrainConfig, *, split: str, max_samples: int) -> Dict[str, Any]:
    dl = make_dataloader(
        {
            "dataset_type": "manifest",
            "manifest_path": cfg.manifest_path,
            "batch_size": 1,
            "num_workers": int(cfg.num_workers),
            "max_samples": int(max_samples),
            "resize_shape": tuple(int(x) for x in cfg.resize_shape),
        },
        split=str(split),
    )

    X: List[torch.Tensor] = []
    y: List[torch.Tensor] = []
    stats: Dict[str, Any] = {"num_samples": 0, "pos_voxels": [], "total_voxels": []}

    for batch in dl:
        vol = batch["volume"][0]  # (D,H,W)
        lesion_masks = batch.get("lesion_masks", [{}])[0] or {}
        vol_shape = tuple(int(x) for x in vol.shape)

        lesion_union = union_lesion_masks(lesion_masks, vol_shape)
        if isinstance(lesion_union, torch.Tensor):
            lesion_union = lesion_union.detach().cpu().numpy()

        x = _preprocess_volume(vol, clip_hu=cfg.clip_hu).unsqueeze(0)  # (1,D,H,W)
        m = torch.from_numpy(lesion_union.astype(np.float32)).unsqueeze(0)  # (1,D,H,W)
        X.append(x.cpu())
        y.append(m.cpu())

        stats["num_samples"] += 1
        stats["pos_voxels"].append(int(lesion_union.astype(bool).sum()))
        stats["total_voxels"].append(int(np.prod(vol_shape)))

        if stats["num_samples"] == 1 or (stats["num_samples"] % 25) == 0:
            print(f"[train_saliency_cnn3d] loaded {stats['num_samples']}/{max_samples} split={split}", flush=True)

    if not X:
        raise RuntimeError(f"No samples loaded for split={split!r}. Check manifest/masks.")

    X_t = torch.stack(X, dim=0)  # (N,1,D,H,W)
    y_t = torch.stack(y, dim=0)  # (N,1,D,H,W)
    return {"X": X_t, "y": y_t, "stats": stats}


@torch.no_grad()
def _eval(model: SaliencyCNN3D, dl: DataLoader, device: torch.device, *, topk_ratio: float) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    # Simple thresholded voxel F1 (proxy)
    tp = fp = fn = 0
    dice_topk_vals: List[float] = []
    iou_topk_vals: List[float] = []
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="mean")
        losses.append(float(loss.item()))
        pred = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.int32)
        yt = (yb >= 0.5).to(dtype=torch.int32)
        tp += int(((pred == 1) & (yt == 1)).sum().item())
        fp += int(((pred == 1) & (yt == 0)).sum().item())
        fn += int(((pred == 0) & (yt == 1)).sum().item())

        # Top-k Dice/IoU at union-mask granularity (matches submission-style ratio thresholding).
        prob = torch.sigmoid(logits).detach()
        b, c, d, h, w = prob.shape
        assert c == 1
        flat = prob.view(b, -1)
        k = max(1, min(int(round(float(topk_ratio) * float(flat.shape[1]))), int(flat.shape[1])))
        # For each sample, pick top-k voxels.
        topk_idx = torch.topk(flat, k=k, dim=1, largest=True).indices
        pred_topk = torch.zeros_like(flat, dtype=torch.bool)
        pred_topk.scatter_(1, topk_idx, True)
        pred_topk = pred_topk.view(b, 1, d, h, w)
        gt = (yb >= 0.5)
        inter = (pred_topk & gt).sum(dim=(1, 2, 3, 4)).float()
        pred_sum = pred_topk.sum(dim=(1, 2, 3, 4)).float()
        gt_sum = gt.sum(dim=(1, 2, 3, 4)).float()
        union = (pred_topk | gt).sum(dim=(1, 2, 3, 4)).float()
        dice = (2.0 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
        iou = (inter + 1e-6) / (union + 1e-6)
        dice_topk_vals.extend([float(x) for x in dice.detach().cpu().tolist()])
        iou_topk_vals.extend([float(x) for x in iou.detach().cpu().tolist()])

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) <= 0 else (2.0 * prec * rec / (prec + rec))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "dice_topk": float(np.mean(dice_topk_vals)) if dice_topk_vals else 0.0,
        "iou_topk": float(np.mean(iou_topk_vals)) if iou_topk_vals else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a 3D saliency CNN (union mask segmentation).")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--smoke", action="store_true", help="Quick sanity run (small samples/epochs).")
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--train-split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--val-split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--max-train-samples", type=int, default=500)
    ap.add_argument("--max-val-samples", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for manifest loading.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument(
        "--loss",
        type=str,
        default="bce_pos_weight",
        choices=["bce_pos_weight", "bce", "bce_dice"],
        help="Training loss. `bce_pos_weight` matches legacy behavior; `bce_dice` is more robust under imbalance.",
    )
    ap.add_argument("--pos-weight-max", type=float, default=0.0, help="Clamp auto pos_weight to this value (0 => no clamp).")
    ap.add_argument("--dice-weight", type=float, default=0.5, help="Weight for Dice loss when --loss=bce_dice.")
    ap.add_argument("--eval-topk-ratio", type=float, default=0.005, help="Top-k ratio for dice_topk/iou_topk eval.")
    ap.add_argument("--output-dir", type=str, default="./outputs/train_saliency_cnn3d")
    args = ap.parse_args()

    if bool(args.smoke):
        cfg = TrainConfig(
            manifest_path=str(args.manifest),
            resize_shape=(32, 32, 32),
            train_split=str(args.train_split),
            val_split=str(args.val_split),
            max_train_samples=min(20, int(args.max_train_samples)),
            max_val_samples=min(10, int(args.max_val_samples)),
            num_workers=0,
            batch_size=1,
            epochs=1,
            seed=int(args.seed),
            device="cpu",
            base_channels=min(8, int(args.base_channels)),
            num_layers=min(2, int(args.num_layers)),
            dropout=float(args.dropout),
            loss=str(args.loss),
            pos_weight_max=float(args.pos_weight_max),
            dice_weight=float(args.dice_weight),
            eval_topk_ratio=float(args.eval_topk_ratio),
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
            num_workers=int(args.num_workers),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            seed=int(args.seed),
            device=str(args.device),
            base_channels=int(args.base_channels),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            loss=str(args.loss),
            pos_weight_max=float(args.pos_weight_max),
            dice_weight=float(args.dice_weight),
            eval_topk_ratio=float(args.eval_topk_ratio),
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

    train = _collect(cfg, split=cfg.train_split, max_samples=int(cfg.max_train_samples))
    val = _collect(cfg, split=cfg.val_split, max_samples=int(cfg.max_val_samples))

    X_train: torch.Tensor = train["X"]
    y_train: torch.Tensor = train["y"]
    X_val: torch.Tensor = val["X"]
    y_val: torch.Tensor = val["y"]

    device = torch.device(cfg.device)
    model = SaliencyCNN3D(
        SaliencyCNN3DConfig(
            in_channels=1,
            base_channels=int(cfg.base_channels),
            num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout),
        )
    ).to(device)

    # Global pos_weight for BCE to handle severe class imbalance.
    pos = float(y_train.sum().item())
    neg = float(y_train.numel() - y_train.sum().item())
    auto_pos_weight = float(neg / max(pos, 1.0))
    if float(cfg.pos_weight_max) and float(cfg.pos_weight_max) > 0:
        auto_pos_weight = float(min(auto_pos_weight, float(cfg.pos_weight_max)))
    pos_weight = torch.tensor([auto_pos_weight], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=int(cfg.batch_size), shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=max(1, int(cfg.batch_size)), shuffle=False)

    history: List[Dict[str, Any]] = []
    best_val_score = -1e9
    best_path = Path(cfg.output_dir) / "saliency_cnn3d.pt"

    for epoch in range(int(cfg.epochs)):
        model.train()
        losses: List[float] = []
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            if str(cfg.loss) == "bce":
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="mean")
            elif str(cfg.loss) == "bce_dice":
                bce = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="mean")
                prob = torch.sigmoid(logits)
                # Soft dice loss (per-sample) to handle extreme imbalance.
                inter = (prob * yb).sum(dim=(1, 2, 3, 4))
                denom = prob.sum(dim=(1, 2, 3, 4)) + yb.sum(dim=(1, 2, 3, 4))
                dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
                dice_loss = 1.0 - dice.mean()
                w = float(max(0.0, min(1.0, float(cfg.dice_weight))))
                loss = (1.0 - w) * bce + w * dice_loss
            else:
                # Legacy: BCE with auto pos_weight.
                loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_eval = _eval(
            model,
            DataLoader(TensorDataset(X_train, y_train), batch_size=max(1, int(cfg.batch_size))),
            device,
            topk_ratio=float(cfg.eval_topk_ratio),
        )
        val_eval = _eval(model, val_dl, device, topk_ratio=float(cfg.eval_topk_ratio))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "train_eval": train_eval,
                "val_eval": val_eval,
            }
        )

        # Prefer dice_topk as the selection criterion (aligns with submission-style thresholding).
        score = float(val_eval.get("dice_topk", 0.0))
        if score > best_val_score:
            best_val_score = float(score)
            save_saliency_cnn3d(str(best_path), model, extra={"meta": meta.to_dict(), "epoch": int(epoch)})

    report: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "train_stats": train["stats"],
        "val_stats": val["stats"],
        "pos_weight": float(pos_weight.detach().cpu().item()),
        "history": history,
        "best_val_score": float(best_val_score),
        "best_val_score_metric": "dice_topk",
        "weights_path": str(best_path),
    }
    out_json = Path(cfg.output_dir) / "train_saliency_cnn3d.json"
    save_results_json(report, str(out_json))
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
