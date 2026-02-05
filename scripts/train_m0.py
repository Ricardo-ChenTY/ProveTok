from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

# Ensure repo root is on sys.path when running as `python scripts/train_m0.py ...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.training.trainer import Trainer, TrainerConfig
from provetok.utils.config import load_yaml
from provetok.utils.seed import set_seed


def _pick_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified training entry (M0 scaffold) using provetok.training.Trainer.")
    ap.add_argument("--config", type=str, default="configs/m0.yaml")
    ap.add_argument("--stage", type=str, default="M0", help="Training stage: M0|M1|M2|M3")
    ap.add_argument("--device", type=str, default="", help="Override device: cpu|cuda")
    ap.add_argument("--max-steps", type=int, default=0, help="Override stage max_steps (0 keeps config-derived)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device = _pick_device(str(cfg.get("device", "auto")))
    if args.device:
        device = str(args.device)

    num_samples = int(cfg.get("data", {}).get("num_samples", 64))
    vol_shape = list(cfg.get("data", {}).get("vol_shape", [64, 128, 128]))
    batch_size = int(cfg.get("train", {}).get("batch_size", 4))
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    log_every = int(cfg.get("train", {}).get("log_every", 10))

    steps_per_epoch = int(math.ceil(num_samples / max(batch_size, 1)))
    derived_max_steps = int(epochs * steps_per_epoch)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else derived_max_steps

    output_dir = str(cfg.get("train", {}).get("save_dir", "./outputs"))

    trainer_cfg = TrainerConfig(
        stage=str(args.stage),
        device=device,
        output_dir=output_dir,
        seed=seed,
        emb_dim=int(cfg.get("model", {}).get("emb_dim", 32)),
        dataset_cfg={
            "dataset_type": str(cfg.get("data", {}).get("type", "synthetic")),
            "num_samples": num_samples,
            "vol_shape": vol_shape,
            "batch_size": batch_size,
            "seed": seed,
            "num_workers": 0,
        },
        overrides={
            "max_steps": max_steps,
            "log_every": log_every,
            # avoid slow eval/saves for the default smoke-like config
            "eval_every": max(10_000_000, max_steps + 1),
            "save_every": max(10_000_000, max_steps + 1),
        },
    )

    trainer = Trainer(trainer_cfg)
    out = trainer.train()
    print(out)


if __name__ == "__main__":
    main()

