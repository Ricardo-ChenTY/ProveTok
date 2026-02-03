from __future__ import annotations
import os
import math
import argparse
from typing import Dict, Any
import torch
from torch import nn
from torch.optim import AdamW
from rich import print as rprint

from provetok.utils.config import load_yaml
from provetok.utils.seed import set_seed
from provetok.data.dataset import make_dataloader
from provetok.pcg import ToyPCG
from provetok.verifier import verify
from provetok.bet import run_refine_loop

def pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def toy_loss_from_artifact(artifact: Dict[str, Any]) -> torch.Tensor:
    # 一个占位 loss：issue 越多 loss 越大；refusal 少一点也算好
    num_issues = len(artifact.get("issues", []))
    num_refusal = sum(1 for v in artifact.get("refusal", {}).values() if v)
    return torch.tensor(float(num_issues + 0.5 * num_refusal))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/m0.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = pick_device(cfg.get("device", "auto"))

    dl = make_dataloader(
        {
            "num_samples": cfg["data"]["num_samples"],
            "vol_shape": cfg["data"]["vol_shape"],
            "seed": cfg["seed"],
            "batch_size": cfg["train"]["batch_size"],
        },
        split="train",
    )

    # 注意：当前 pipeline 还是用 ToyPCG + verifier + refine loop，因此这里的“训练”只是 sanity check
    pcg = ToyPCG(emb_dim=int(cfg["model"]["emb_dim"]), topk=int(cfg["model"]["cite_topk"]), seed=int(cfg["seed"]))

    # 预留：未来你会把 pcg 换成 nn.Module，然后对其参数做优化
    dummy = nn.Parameter(torch.zeros(()))
    opt = AdamW([dummy], lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    save_dir = cfg["train"].get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    for epoch in range(int(cfg["train"]["epochs"])):
        for batch in dl:
            vols = batch["volume"].to(device)  # (B,D,H,W)
            loss_total = 0.0
            for b in range(vols.shape[0]):
                vol = vols[b]
                res = run_refine_loop(
                    volume=vol,
                    budget_tokens=64,
                    steps=3,
                    generator_fn=lambda tokens: pcg(tokens),
                    verifier_fn=lambda gen, tokens: verify(gen, tokens),
                    emb_dim=int(cfg["model"]["emb_dim"]),
                    seed=int(cfg["seed"]) + int(batch["sample_id"][b]),
                )
                artifact = {
                    "issues": [i.__dict__ for i in res.issues],
                    "refusal": res.gen.refusal,
                }
                loss_total = loss_total + toy_loss_from_artifact(artifact).to(device)

            loss = loss_total / vols.shape[0]
            # attach dummy so graph exists
            loss = loss + dummy * 0.0

            opt.zero_grad()
            loss.backward()
            opt.step()

            if global_step % int(cfg["train"]["log_every"]) == 0:
                rprint(f"[cyan]step={global_step}[/cyan] loss={float(loss.item()):.4f} device={device}")

            global_step += 1

    ckpt_path = os.path.join(save_dir, "m0_sanity.pt")
    torch.save({"dummy": dummy.detach().cpu(), "cfg": cfg}, ckpt_path)
    rprint(f"[green]Saved checkpoint -> {ckpt_path}[/green]")

if __name__ == "__main__":
    main()
