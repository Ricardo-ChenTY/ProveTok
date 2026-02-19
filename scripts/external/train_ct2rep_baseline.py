#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _default_ct2rep_args() -> SimpleNamespace:
    # Mirror defaults in CT2Rep/main.py:parse_agrs (only fields CT2Rep uses).
    return SimpleNamespace(
        # Data
        max_seq_length=200,
        threshold=3,
        num_workers=2,
        batch_size=2,
        dataset_name="ct_dataset",
        # Model
        d_model=512,
        d_ff=512,
        d_vf=512,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
        logit_layers=1,
        bos_idx=0,
        eos_idx=0,
        pad_idx=0,
        use_bn=0,
        drop_prob_lm=0.5,
        rm_num_slots=3,
        rm_num_heads=8,
        rm_d_model=512,
        # Sampling
        sample_method="beam_search",
        beam_size=3,
        temperature=1.0,
        sample_n=1,
        group_size=1,
        output_logsoftmax=1,
        decoding_constraint=0,
        block_trigrams=1,
        # Trainer
        n_gpu=1,
        epochs=100,
        save_dir="results/",
        record_dir="records/",
        save_period=1,
        monitor_mode="max",
        monitor_metric="BLEU_4",
        early_stop=50,
        # Optim
        optim="Adam",
        lr_ve=5e-5,
        lr_ed=1e-4,
        weight_decay=5e-5,
        amsgrad=True,
        # Scheduler
        lr_scheduler="StepLR",
        step_size=50,
        gamma=0.1,
        # Paths
        xlsxfile="",
        trainfolder="",
        validfolder="",
        resume=None,
    )


def _load_npz_volume_as_ct2rep_tensor(path: str):
    import numpy as np
    import torch
    import torch.nn.functional as F

    z = np.load(path)
    if "arr_0" in z:
        arr = np.asarray(z["arr_0"], dtype=np.float32)
    elif "volume" in z:
        # ProveTok RRG-DPO preprocess stores (D,H,W) raw-ish CT values. A common
        # convention is HU = raw - 1024. CT2Rep expects arr_0 in HU/1000.
        vol = np.asarray(z["volume"], dtype=np.float32)
        arr = (vol - 1024.0) / 1000.0
    else:
        arr = np.asarray(z[z.files[0]], dtype=np.float32)

    # CT2Rep expects (H,W,D) during crop/pad, then permutes back to (D,H,W).
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr.shape} for {path}")
    arr = arr.transpose(1, 2, 0)

    # Back to HU.
    hu = arr * 1000.0
    hu = hu.clip(-1000.0, 200.0)
    img = ((hu + 400.0) / 600.0).astype(np.float32)

    tensor = torch.tensor(img)

    # Center crop/pad to (480,480,240) as in CT2Rep.
    target_h, target_w, target_d = (480, 480, 240)
    h, w, d = tensor.shape

    h_start = max((h - target_h) // 2, 0)
    w_start = max((w - target_w) // 2, 0)
    d_start = max((d - target_d) // 2, 0)

    tensor = tensor[h_start : h_start + min(target_h, h), w_start : w_start + min(target_w, w), d_start : d_start + min(target_d, d)]

    pad_h_before = (target_h - tensor.size(0)) // 2
    pad_h_after = target_h - tensor.size(0) - pad_h_before

    pad_w_before = (target_w - tensor.size(1)) // 2
    pad_w_after = target_w - tensor.size(1) - pad_w_before

    pad_d_before = (target_d - tensor.size(2)) // 2
    pad_d_after = target_d - tensor.size(2) - pad_d_before

    tensor = F.pad(
        tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1.0,
    )

    tensor = tensor.permute(2, 0, 1)  # (D,H,W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    return tensor[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train CT2Rep baseline using a ProveTok-built CT2Rep-style folder dataset.\n\n"
            "Prereqs:\n"
            "- Clone CT2Rep: https://github.com/ibrahimethemhamamci/CT2Rep\n"
            "- Build dataset via scripts/external/build_ct2rep_dataset_from_manifest.py\n"
        )
    )
    ap.add_argument(
        "--ct2rep-root",
        type=str,
        default="/data/tiasha/external/CT2Rep/CT2Rep",
        help="Path to CT2Rep code root that contains modules/ and models/",
    )
    ap.add_argument("--xlsxfile", type=str, required=True)
    ap.add_argument("--trainfolder", type=str, required=True)
    ap.add_argument("--validfolder", type=str, required=True)
    ap.add_argument("--save-dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--n-gpu", type=int, default=1)

    ap.add_argument("--max-seq-length", type=int, default=200)
    ap.add_argument("--threshold", type=int, default=3)

    ap.add_argument("--lr-ve", type=float, default=5e-5)
    ap.add_argument("--lr-ed", type=float, default=1e-4)

    ap.add_argument("--save-period", type=int, default=5)
    ap.add_argument("--early-stop", type=int, default=10)
    ap.add_argument("--resume", type=str, default="")

    args_in = ap.parse_args()

    ct2rep_root = Path(args_in.ct2rep_root)
    if not ct2rep_root.exists():
        raise SystemExit(f"Missing CT2Rep root: {ct2rep_root}")

    repo_root = ct2rep_root.parent
    ctvit_root = repo_root / "ctvit"

    sys.path.insert(0, str(ctvit_root))
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(ct2rep_root))

    from modules.tokenizers import Tokenizer  # type: ignore
    from modules.dataloaders import R2DataLoader  # type: ignore
    from modules.metrics import compute_scores  # type: ignore
    from modules.optimizers import build_lr_scheduler, build_optimizer  # type: ignore
    from modules.trainer import Trainer  # type: ignore
    from modules.loss import compute_loss  # type: ignore
    from models.ct2rep import CT2RepModel  # type: ignore
    from modules.data_ct import CTReportDataset  # type: ignore

    class CTReportDatasetProveTok(CTReportDataset):
        def nii_img_to_tensor(self, path, transform):  # noqa: ARG002
            return _load_npz_volume_as_ct2rep_tensor(path)

    args = _default_ct2rep_args()
    args.xlsxfile = str(Path(args_in.xlsxfile).resolve())
    args.trainfolder = str(Path(args_in.trainfolder).resolve())
    args.validfolder = str(Path(args_in.validfolder).resolve())

    args.epochs = int(args_in.epochs)
    args.batch_size = int(args_in.batch_size)
    args.num_workers = int(args_in.num_workers)
    args.n_gpu = int(args_in.n_gpu)

    args.max_seq_length = int(args_in.max_seq_length)
    args.threshold = int(args_in.threshold)

    args.lr_ve = float(args_in.lr_ve)
    args.lr_ed = float(args_in.lr_ed)

    args.save_dir = str(Path(args_in.save_dir).resolve())
    args.save_period = int(args_in.save_period)
    args.early_stop = int(args_in.early_stop)
    args.resume = str(args_in.resume) if str(args_in.resume) else None

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Keep CT2Rep's tokenizer side effects (idx2token.json/token2idx.json) out of the repo.
    os.chdir(str(save_dir))

    with open(save_dir / "ct2rep_train_args.json", "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    tokenizer = Tokenizer(args)

    train_ds = CTReportDatasetProveTok(args, data_folder=args.trainfolder, xlsx_file=args.xlsxfile, tokenizer=tokenizer, num_frames=2)
    valid_ds = CTReportDatasetProveTok(args, data_folder=args.validfolder, xlsx_file=args.xlsxfile, tokenizer=tokenizer, num_frames=2)

    train_dl = R2DataLoader(args, train_ds, tokenizer, split="train", shuffle=True)
    val_dl = R2DataLoader(args, valid_ds, tokenizer, split="val", shuffle=False)

    model = CT2RepModel(args, tokenizer)

    criterion = compute_loss
    metrics = compute_scores

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dl, val_dl, val_dl)
    trainer.train()


if __name__ == "__main__":
    main()
