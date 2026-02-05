"""Train a reproducible "strong report generation baseline" (ct2rep_strong) on manifest-driven data.

This is a paper-grade improvement over the placeholder `ct2rep_like`:
- Learns query vectors and polarity classification from report-derived frames
- Emits token citations via learned attention weights
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..baselines.tokenizers import FixedGridTokenizer
from ..data import make_dataloader
from ..eval.compute_budget import ComputeUnitCosts, match_b_enc_for_total_flops
from ..eval.metrics_frames import compute_frame_f1
from ..models.ct2rep_strong import CT2RepStrong, CT2RepStrongConfig, save_ct2rep_strong
from ..pcg.schema import FINDINGS
from ..pcg.schema_version import SCHEMA_VERSION
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .utils import make_output_dir, save_results_json, set_seed


@dataclass(frozen=True)
class TrainConfig:
    manifest_path: str
    resize_shape: Tuple[int, int, int] = (64, 64, 64)
    train_split: str = "train"
    val_split: str = "val"
    max_train_samples: int = 500
    max_val_samples: int = 200
    max_depth: int = 6
    budget_tokens: int = 128
    flops_total: float = 0.0
    costs_json: str = ""
    emb_dim: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 5
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    topk_citations: int = 3
    output_dir: str = "./outputs/train_ct2rep_strong"
    export_weights: str = ""


def _labels_from_frames(frames: List[Any]) -> torch.Tensor:
    """Return (K,) labels in {0:none, 1:present, 2:absent} for FINDINGS."""
    K = len(FINDINGS)
    y = torch.zeros((K,), dtype=torch.long)
    for fr in frames or []:
        finding = getattr(fr, "finding", None)
        polarity = getattr(fr, "polarity", None)
        if not isinstance(finding, str) or finding not in FINDINGS:
            continue
        k = FINDINGS.index(finding)
        if str(polarity) == "present":
            y[k] = 1
        elif str(polarity) == "absent" and y[k].item() == 0:
            y[k] = 2
    return y


@torch.no_grad()
def _predict_frames(tokens_emb: torch.Tensor, logits: torch.Tensor) -> List[Any]:
    from ..types import Frame

    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)  # (K,)
    out: List[Any] = []
    for k, cls in enumerate(preds.tolist()):
        if int(cls) == 0:
            continue
        pol = "present" if int(cls) == 1 else "absent"
        conf = float(probs[k, int(cls)].item())
        out.append(Frame(finding=str(FINDINGS[k]), polarity=pol, laterality="unspecified", confidence=conf))
    return out


def _b_enc(cfg: TrainConfig, *, costs: ComputeUnitCosts) -> int:
    if float(cfg.flops_total) > 0.0:
        return match_b_enc_for_total_flops(
            flops_total=float(cfg.flops_total),
            b_gen=128,
            n_verify=1,
            costs=costs,
            flops_extra=0.0,
            min_b_enc=1,
            max_b_enc=4096,
        )
    return int(cfg.budget_tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ct2rep_strong (learned baseline) on manifest splits.")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--smoke", action="store_true", help="Quick sanity run (small samples/epochs on CPU).")
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--train-split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--val-split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--max-train-samples", type=int, default=500)
    ap.add_argument("--max-val-samples", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--budget-tokens", type=int, default=128)
    ap.add_argument("--flops-total", type=float, default=0.0)
    ap.add_argument("--costs-json", type=str, default="")
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default="./outputs/train_ct2rep_strong")
    ap.add_argument("--export-weights", type=str, default="", help="Optional path to copy final weights to (stable path).")
    args = ap.parse_args()

    device = str(args.device) if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(
        manifest_path=str(args.manifest),
        resize_shape=tuple(int(x) for x in args.resize_shape),
        train_split=str(args.train_split),
        val_split=str(args.val_split),
        max_train_samples=int(args.max_train_samples),
        max_val_samples=int(args.max_val_samples),
        max_depth=int(args.max_depth),
        budget_tokens=int(args.budget_tokens),
        flops_total=float(args.flops_total),
        costs_json=str(args.costs_json),
        emb_dim=int(args.emb_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=1,
        seed=int(args.seed),
        device=device,
        output_dir=str(args.output_dir),
        export_weights=str(args.export_weights),
    )
    if bool(args.smoke):
        cfg = TrainConfig(
            manifest_path=str(args.manifest),
            resize_shape=(32, 32, 32),
            train_split=str(args.train_split),
            val_split=str(args.val_split),
            max_train_samples=20,
            max_val_samples=20,
            max_depth=5,
            budget_tokens=64,
            flops_total=0.0,
            costs_json=str(args.costs_json),
            emb_dim=32,
            lr=2e-3,
            weight_decay=0.0,
            batch_size=1,
            epochs=1,
            seed=int(args.seed),
            device="cpu",
            output_dir=str(args.output_dir),
            export_weights=str(args.export_weights),
        )

    set_seed(int(cfg.seed))

    out_dir = make_output_dir(cfg.output_dir, "train_ct2rep_strong")
    cfg = TrainConfig(**{**asdict(cfg), "output_dir": out_dir})
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

    costs = ComputeUnitCosts.from_json(cfg.costs_json) if cfg.costs_json else ComputeUnitCosts()
    b_enc = int(_b_enc(cfg, costs=costs))

    dl_train = make_dataloader(
        {
            "dataset_type": "manifest",
            "manifest_path": cfg.manifest_path,
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": int(cfg.max_train_samples),
            "resize_shape": tuple(int(x) for x in cfg.resize_shape),
        },
        split=str(cfg.train_split),
    )
    dl_val = make_dataloader(
        {
            "dataset_type": "manifest",
            "manifest_path": cfg.manifest_path,
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": int(cfg.max_val_samples),
            "resize_shape": tuple(int(x) for x in cfg.resize_shape),
        },
        split=str(cfg.val_split),
    )

    model = CT2RepStrong(CT2RepStrongConfig(emb_dim=int(cfg.emb_dim), num_findings=len(FINDINGS), num_classes=3, dropout=0.1))
    model = model.to(torch.device(cfg.device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    class_weights = torch.tensor([0.25, 1.0, 1.0], dtype=torch.float32, device=torch.device(cfg.device))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    tok = FixedGridTokenizer(max_depth=int(cfg.max_depth))
    best_val_f1 = -1.0
    best_path = Path(cfg.output_dir) / "ct2rep_strong.pt"

    history: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "config": asdict(cfg),
        "b_enc": int(b_enc),
        "epochs": [],
    }

    for epoch in range(int(cfg.epochs)):
        model.train()
        train_losses: List[float] = []
        for i, batch in enumerate(dl_train):
            vol = batch["volume"][0]  # (D,H,W)
            frames = batch.get("frames", [[]])[0] or []
            sample_seed = int(cfg.seed) + int(epoch) * 100_000 + int(i)
            tokens = tok.tokenize(vol, budget_tokens=int(b_enc), emb_dim=int(cfg.emb_dim), seed=int(sample_seed))
            if not tokens:
                continue
            emb = torch.stack([t.embedding for t in tokens], dim=0).to(torch.device(cfg.device))
            y = _labels_from_frames(frames).to(torch.device(cfg.device))
            logits, _att = model(emb)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        # Val
        model.eval()
        val_losses: List[float] = []
        val_f1s: List[float] = []
        for i, batch in enumerate(dl_val):
            vol = batch["volume"][0]
            frames = batch.get("frames", [[]])[0] or []
            sample_seed = int(cfg.seed) + 10_000 + int(epoch) * 100_000 + int(i)
            tokens = tok.tokenize(vol, budget_tokens=int(b_enc), emb_dim=int(cfg.emb_dim), seed=int(sample_seed))
            if not tokens:
                continue
            emb = torch.stack([t.embedding for t in tokens], dim=0).to(torch.device(cfg.device))
            y = _labels_from_frames(frames).to(torch.device(cfg.device))
            logits, _att = model(emb)
            loss = loss_fn(logits, y)
            val_losses.append(float(loss.item()))
            pred_frames = _predict_frames(emb, logits)
            val_f1s.append(float(compute_frame_f1(pred_frames, frames, threshold=0.3).f1))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_f1 = float(np.mean(val_f1s)) if val_f1s else 0.0

        history["epochs"].append(
            {"epoch": int(epoch), "train_loss": train_loss, "val_loss": val_loss, "val_frame_f1": val_f1}
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_ct2rep_strong(str(best_path), model, extra={"best_val_frame_f1": float(best_val_f1)})

    history["best_val_frame_f1"] = float(best_val_f1)
    history["weights_path"] = str(best_path)

    if cfg.export_weights:
        dst = Path(str(cfg.export_weights))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best_path, dst)
        history["export_weights_path"] = str(dst)
        print(f"Exported weights -> {dst}")

    save_results_json(history, str(Path(cfg.output_dir) / "train_ct2rep_strong.json"))

    print(f"Saved -> {best_path}")


if __name__ == "__main__":
    main()
