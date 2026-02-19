from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class TrainCfg:
    model_path: str
    train_jsonl: str
    output_dir: str

    seed: int = 0
    device: str = "cuda"
    dtype: str = "float16"  # float16|bfloat16
    max_seq_len: int = 1024

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    # Optim
    lr: float = 5e-5
    weight_decay: float = 0.0
    max_steps: int = 10
    batch_size: int = 1
    grad_accum: int = 1
    max_grad_norm: float = 1.0


class JsonlSFTDataset(Dataset):
    def __init__(self, path: str):
        self.path = str(path)
        rows: List[Dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                rows.append(json.loads(s))
        if not rows:
            raise ValueError(f"Empty jsonl: {self.path}")
        self.rows = rows

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, idx: int) -> Dict:
        return dict(self.rows[int(idx)])


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encode_example(
    tokenizer,
    *,
    prompt: str,
    completion: str,
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    # Important: prompts already contain Llama-2 chat tokens (<s>[INST] ... [/INST]).
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids

    eos_id = int(tokenizer.eos_token_id)
    input_ids = prompt_ids + completion_ids + [eos_id]
    if len(input_ids) > int(max_seq_len):
        input_ids = input_ids[-int(max_seq_len) :]

    # Labels: mask prompt tokens; train on completion tokens + eos.
    n_prompt = len(prompt_ids)
    labels = [-100] * n_prompt + completion_ids + [eos_id]
    if len(labels) > int(max_seq_len):
        labels = labels[-int(max_seq_len) :]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _collate(batch: List[Dict], tokenizer, *, max_seq_len: int) -> Dict[str, torch.Tensor]:
    feats = []
    for row in batch:
        feats.append(
            _encode_example(
                tokenizer,
                prompt=str(row.get("prompt", "")),
                completion=str(row.get("completion", "")),
                max_seq_len=int(max_seq_len),
            )
        )

    pad_id = int(tokenizer.pad_token_id)
    max_len = max(int(x["input_ids"].shape[0]) for x in feats)
    max_len = min(int(max_len), int(max_seq_len))

    def pad_1d(x: torch.Tensor, *, pad_val: int) -> torch.Tensor:
        if int(x.shape[0]) >= int(max_len):
            return x[:max_len]
        pad = torch.full((int(max_len) - int(x.shape[0]),), int(pad_val), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    out = {
        "input_ids": torch.stack([pad_1d(x["input_ids"], pad_val=pad_id) for x in feats], dim=0),
        "attention_mask": torch.stack([pad_1d(x["attention_mask"], pad_val=0) for x in feats], dim=0),
        "labels": torch.stack([pad_1d(x["labels"], pad_val=-100) for x in feats], dim=0),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA smoke SFT for Llama2PCG (pp.md §6.6).")
    ap.add_argument("--model-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--train-jsonl", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--max-seq-len", type=int, default=1024)

    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", type=str, nargs="+", default=["q_proj", "v_proj"])

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)

    args = ap.parse_args()

    cfg = TrainCfg(
        model_path=str(args.model_path),
        train_jsonl=str(args.train_jsonl),
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
        max_seq_len=int(args.max_seq_len),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=tuple(str(x) for x in args.target_modules),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        grad_accum=int(args.grad_accum),
        max_grad_norm=float(args.max_grad_norm),
    )

    _set_seed(int(cfg.seed))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base model/tokenizer.
    dtype = torch.float16 if cfg.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device(cfg.device)
    # Load directly to GPU when possible.
    device_map = {"": 0} if str(cfg.device).startswith("cuda") else None
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # Attach LoRA.
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing optional dependency `peft`. Install in the Py3.11 env: pip install peft") from e

    lora_cfg = LoraConfig(
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if str(cfg.device).startswith("cuda"):
        model.to(device)

    # Optional memory saver.
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    # Some transformer models require this for gradient checkpointing to work with LoRA.
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    ds = JsonlSFTDataset(cfg.train_jsonl)
    dl = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: _collate(b, tokenizer, max_seq_len=int(cfg.max_seq_len)),
    )

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    model.train()
    t0 = time.perf_counter()
    step = 0
    losses: List[float] = []
    opt.zero_grad(set_to_none=True)

    while step < int(cfg.max_steps):
        for batch in dl:
            if step >= int(cfg.max_steps):
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if loss is None:
                raise RuntimeError("model returned loss=None")

            loss = loss / float(max(1, int(cfg.grad_accum)))
            loss.backward()

            if (step + 1) % int(cfg.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
                opt.step()
                opt.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().float().cpu().item()))
            if step == 0 or (step + 1) % 2 == 0:
                elapsed = max(1e-6, float(time.perf_counter() - t0))
                print(f"[step {step+1}/{cfg.max_steps}] loss={losses[-1]:.4f} elapsed={elapsed:.1f}s")

            step += 1

    # Save adapter.
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    summary = {
        "config": asdict(cfg),
        "n_rows": int(len(ds)),
        "final_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(str(out_dir))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
