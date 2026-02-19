#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _resolve_ct2rep_ckpt_path(ckpt_arg: str, *, orig_cwd: Path) -> Path:
    p = Path(ckpt_arg).expanduser()
    if not p.is_absolute():
        p = (orig_cwd / p).resolve()
    else:
        p = p.resolve()

    if p.is_file():
        return p

    parent = p.parent
    candidates: list[tuple[int, Path]] = []
    for c in parent.glob("current_checkpoint_*.pth"):
        try:
            epoch = int(c.stem.split("_")[-1])
        except Exception:
            epoch = -1
        candidates.append((epoch, c))

    if candidates:
        epoch, best = max(candidates, key=lambda t: t[0])
        print(
            f"[warn] checkpoint not found: {p}. Falling back to {best} (epoch={epoch}).",
            file=sys.stderr,
        )
        return best

    raise FileNotFoundError(f"Checkpoint not found: {p} (and no current_checkpoint_*.pth in {parent})")


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

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr.shape} for {path}")

    # CT2Rep expects (H,W,D) during crop/pad, then permutes back to (D,H,W).
    arr = arr.transpose(1, 2, 0)

    hu = arr * 1000.0
    hu = hu.clip(-1000.0, 200.0)
    img = ((hu + 400.0) / 600.0).astype(np.float32)

    tensor = torch.tensor(img)

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


def _load_train_args_json(path: str) -> SimpleNamespace:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    d = json.loads(p.read_text(encoding="utf-8"))
    ns = _default_ct2rep_args()
    for k, v in d.items():
        if hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CT2Rep inference and export ProveTok pred_jsonl.")
    ap.add_argument(
        "--ct2rep-root",
        type=str,
        default="/data/tiasha/external/CT2Rep/CT2Rep",
        help="Path to CT2Rep code root that contains modules/ and models/",
    )
    ap.add_argument("--train-args-json", type=str, default="", help="Optional ct2rep_train_args.json from training")
    ap.add_argument("--xlsxfile", type=str, required=True)
    ap.add_argument("--data-folder", type=str, required=True, help="CT2Rep-style folder root for the split")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to CT2Rep checkpoint (.pth)")
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--method", type=str, default="ct2rep")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args_in = ap.parse_args()
    orig_cwd = Path.cwd()

    ct2rep_root = Path(args_in.ct2rep_root)
    if not ct2rep_root.exists():
        raise SystemExit(f"Missing CT2Rep root: {ct2rep_root}")

    repo_root = ct2rep_root.parent
    ctvit_root = repo_root / "ctvit"

    sys.path.insert(0, str(ctvit_root))
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(ct2rep_root))

    import torch

    from modules.tokenizers import Tokenizer  # type: ignore
    from modules.dataloaders import R2DataLoader  # type: ignore
    from models.ct2rep import CT2RepModel  # type: ignore
    from modules.data_ct import CTReportDataset  # type: ignore

    class CTReportDatasetProveTok(CTReportDataset):
        def nii_img_to_tensor(self, path, transform):  # noqa: ARG002
            return _load_npz_volume_as_ct2rep_tensor(path)

    if str(args_in.train_args_json):
        args = _load_train_args_json(args_in.train_args_json)
    else:
        args = _default_ct2rep_args()

    args.xlsxfile = str(Path(args_in.xlsxfile).resolve())
    args.batch_size = int(args_in.batch_size)
    args.num_workers = int(args_in.num_workers)
    args.n_gpu = 1 if (args_in.device == "cuda" and torch.cuda.is_available()) else 0

    out_jsonl = Path(args_in.out_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Keep tokenizer side effects out of the repo.
    os.chdir(str(out_jsonl.parent))

    tokenizer = Tokenizer(args)

    ds = CTReportDatasetProveTok(
        args,
        data_folder=str(Path(args_in.data_folder).resolve()),
        xlsx_file=args.xlsxfile,
        tokenizer=tokenizer,
        num_frames=2,
    )
    dl = R2DataLoader(args, ds, tokenizer, split="test", shuffle=False)

    model = CT2RepModel(args, tokenizer)

    ckpt_path = _resolve_ct2rep_ckpt_path(args_in.ckpt, orig_cwd=orig_cwd)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)

    device = torch.device("cuda:0" if (args_in.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    model.eval()

    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for images_id, images, _reports_ids, _reports_masks in dl:
                images = images.to(device)
                output = model(images, mode="sample")
                reports = model.tokenizer.decode_batch(output.cpu().numpy())

                for img_id, pred_text in zip(images_id, reports):
                    sample_id = Path(str(img_id)).stem
                    rec = {"sample_id": sample_id, "method": str(args_in.method), "pred_text": str(pred_text)}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1

    print(
        json.dumps(
            {"out_jsonl": str(out_jsonl), "n": n, "method": str(args_in.method), "ckpt": str(ckpt_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
