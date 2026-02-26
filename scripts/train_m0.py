from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import re

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


def _normalize_dataset_type(raw: str) -> str:
    t = str(raw or "").strip().lower()
    if t in ("real", "manifest"):
        return "manifest"
    if t in ("ct_rate",):
        return "ct_rate"
    return "synthetic"


def _validate_manifest_contract(
    manifest_path: str,
    *,
    require_mask: bool = False,
    require_anatomy: bool = False,
    max_probe: int = 256,
) -> None:
    p = Path(str(manifest_path))
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    n = 0
    bad_volume = 0
    bad_report = 0
    bad_mask = 0
    bad_anatomy = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            ln = str(line).strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if not isinstance(obj, dict):
                continue
            n += 1
            vol = str(obj.get("volume_path", obj.get("volume", "")) or "").strip()
            rpt = str(obj.get("report_text", obj.get("report", "")) or "").strip()
            if not vol:
                bad_volume += 1
            if not rpt:
                bad_report += 1
            if require_mask:
                has_mask = bool(
                    str(obj.get("mask_path", obj.get("lesion_mask_path", "")) or "").strip()
                    or (isinstance(obj.get("mask_paths"), list) and len(obj.get("mask_paths", [])) > 0)
                    or (isinstance(obj.get("lesion_mask_paths"), list) and len(obj.get("lesion_mask_paths", [])) > 0)
                )
                if not has_mask:
                    bad_mask += 1
            if require_anatomy:
                has_anatomy = bool(
                    str(
                        obj.get("anatomy_label_path", "")
                        or obj.get("atlas_label_path", "")
                        or obj.get("totalseg_label_path", "")
                        or obj.get("tsseg_label_path", "")
                    ).strip()
                )
                if not has_anatomy:
                    bad_anatomy += 1
            if n >= int(max_probe):
                break

    if n <= 0:
        raise ValueError(f"Manifest is empty: {p}")
    if bad_volume > 0 or bad_report > 0 or bad_mask > 0 or bad_anatomy > 0:
        raise ValueError(
            f"Manifest contract check failed on first {n} rows: "
            f"missing volume={bad_volume}, report_text={bad_report}, "
            f"mask={bad_mask}, anatomy={bad_anatomy}."
        )


def _find_latest_checkpoint(stage_dir: Path) -> Path | None:
    final_ckpt = stage_dir / "ckpt_final.pt"
    if final_ckpt.exists():
        return final_ckpt

    best_step = -1
    best_path: Path | None = None
    for p in stage_dir.glob("ckpt_step*.pt"):
        m = re.search(r"ckpt_step(\d+)\.pt$", str(p.name))
        if m is None:
            continue
        st = int(m.group(1))
        if st > best_step:
            best_step = st
            best_path = p
    return best_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified training entry (M0 scaffold) using provetok.training.Trainer.")
    ap.add_argument("--config", type=str, default="configs/m0.yaml")
    ap.add_argument("--stage", type=str, default="M0", help="Training stage: M0|M1|M2|M3")
    ap.add_argument("--device", type=str, default="", help="Override device: cpu|cuda")
    ap.add_argument("--max-steps", type=int, default=0, help="Override stage max_steps (0 keeps config-derived)")
    ap.add_argument("--resume-from", type=str, default="", help="Optional checkpoint path to resume from.")
    ap.add_argument("--auto-resume", action="store_true", help="Auto-resume from latest checkpoint under output_dir/<stage>/.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device = _pick_device(str(cfg.get("device", "auto")))
    if args.device:
        device = str(args.device)

    dataset_type = _normalize_dataset_type(str(cfg.get("data", {}).get("type", "synthetic")))
    manifest_path = str(cfg.get("data", {}).get("manifest_path", ""))
    use_ground_loss = float(cfg.get("train", {}).get("lambda_ground", 0.0)) > 0.0 or bool(
        cfg.get("data", {}).get("require_mask_contract", False)
    )
    use_r5 = bool(cfg.get("verifier", {}).get("enable_r5", False))
    if dataset_type == "manifest":
        _validate_manifest_contract(
            manifest_path,
            require_mask=bool(use_ground_loss),
            require_anatomy=bool(use_r5),
        )

    num_samples = int(cfg.get("data", {}).get("num_samples", 64))
    max_samples = int(cfg.get("data", {}).get("max_samples", num_samples))
    vol_shape = list(cfg.get("data", {}).get("vol_shape", [64, 128, 128]))
    batch_size = int(cfg.get("train", {}).get("batch_size", 4))
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    log_every = int(cfg.get("train", {}).get("log_every", 10))

    steps_per_epoch = int(math.ceil(num_samples / max(batch_size, 1)))
    derived_max_steps = int(epochs * steps_per_epoch)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else derived_max_steps

    run_name = str(cfg.get("run_name", "") or "").strip()
    base_out = str(cfg.get("train", {}).get("save_dir", "./outputs"))
    output_dir = str(Path(base_out) / run_name) if run_name else base_out

    trainer_cfg = TrainerConfig(
        stage=str(args.stage),
        device=device,
        output_dir=output_dir,
        seed=seed,
        emb_dim=int(cfg.get("model", {}).get("emb_dim", 32)),
        dataset_cfg={
            "dataset_type": dataset_type,
            "manifest_path": manifest_path,
            "num_samples": num_samples,
            "vol_shape": vol_shape,
            "resize_shape": tuple(int(x) for x in vol_shape),
            "max_samples": max_samples,
            "batch_size": batch_size,
            "seed": seed,
            "num_workers": int(cfg.get("data", {}).get("num_workers", 16)),      # 释放 CPU，启用多线程读图防止 GPU 饿死
        },
        encoder_cfg={
            "backend": str(cfg.get("model", {}).get("encoder_backend", "none")),
            "img_size": int(vol_shape[0]) if len(vol_shape) > 0 else 128,
            "in_channels": int(cfg.get("model", {}).get("in_channels", 1)),
            "feature_size": int(cfg.get("model", {}).get("feature_size", 48)),
            "out_channels": int(cfg.get("model", {}).get("out_channels", 14)),
            "checkpoint_path": str(cfg.get("model", {}).get("encoder_ckpt", "")),
            "emb_dim": int(cfg.get("model", {}).get("emb_dim", 32)),
        },
        verifier_cfg={
            "mode": str(cfg.get("verifier", {}).get("mode", "rules")),
            "l_min": int(cfg.get("verifier", {}).get("l_min", 2)),
            "enable_r5": bool(cfg.get("verifier", {}).get("enable_r5", False)),
            "enable_r6": bool(cfg.get("verifier", {}).get("enable_r6", False)),
            "r6_threshold": float(cfg.get("verifier", {}).get("r6_threshold", 0.3)),
            "semantic_rerank_on_violation": bool(cfg.get("verifier", {}).get("semantic_rerank_on_violation", False)),
            "semantic_rerank_topk": int(cfg.get("verifier", {}).get("semantic_rerank_topk", 3)),
            "semantic_rerank_rule_ids": list(cfg.get("verifier", {}).get("semantic_rerank_rule_ids", ["R5", "R6"])),
            "despecify_on_remaining_issues": bool(cfg.get("verifier", {}).get("despecify_on_remaining_issues", False)),
            "despecify_confidence_cap": float(cfg.get("verifier", {}).get("despecify_confidence_cap", 0.6)),
        },
        pcg_cfg={
            "top_k_citations": int(cfg.get("model", {}).get("cite_topk", 3)),
            "tau_refuse": float(cfg.get("model", {}).get("tau_refuse", 0.55)),
        },
        overrides={
            "max_steps": max_steps,
            "log_every": log_every,
            "lr": float(cfg.get("train", {}).get("lr", 1e-3)),
            "weight_decay": float(cfg.get("train", {}).get("weight_decay", 0.01)),
            "batch_size": batch_size,
            "budget_tokens": int(cfg.get("bet", {}).get("budget_tokens", cfg.get("train", {}).get("budget_tokens", 128))),
            "bet_steps": int(cfg.get("bet", {}).get("steps", cfg.get("train", {}).get("bet_steps", 5))),
            "max_depth": int(cfg.get("bet", {}).get("max_depth", cfg.get("train", {}).get("max_depth", 4))),
            "epsilon": float(cfg.get("bet", {}).get("epsilon", cfg.get("train", {}).get("epsilon", 0.01))),
            "verifier_refresh_period": int(
                cfg.get("bet", {}).get("verifier_refresh_period", cfg.get("train", {}).get("verifier_refresh_period", 1))
            ),
            # avoid slow eval/saves for the default smoke-like config
            "eval_every": max(10_000_000, max_steps + 1),
            "save_every": max(10_000_000, max_steps + 1),
        },
    )

    trainer = Trainer(trainer_cfg)
    resume_path = str(args.resume_from or "").strip()
    if not resume_path and bool(args.auto_resume):
        stage_dir = Path(output_dir) / str(args.stage)
        latest = _find_latest_checkpoint(stage_dir)
        if latest is not None:
            resume_path = str(latest)

    if resume_path:
        rp = Path(resume_path).resolve()
        if not rp.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {rp}")
        print(f"[train_m0] resuming from: {rp}")
        trainer.load_checkpoint(str(rp))

    out = trainer.train()
    print(out)


if __name__ == "__main__":
    main()

