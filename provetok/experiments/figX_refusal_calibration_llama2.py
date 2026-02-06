"""Fig X (Pilot): Refusal calibration in a LLaMA-2 PCG setting.

This is an *oral-ready pilot* to answer:
- "Is refusal just gaming metrics?"
- "Is tau_refuse tuned post-hoc on test?"

Contract (minimal):
1) Generate dev(val) / test(test) samples with `pcg=llama2`.
2) Select `tau_refuse` on dev once, then freeze it for test.
3) Report unsupported_rate / refusal_rate / refusal_ece (+ critical_miss_rate) for:
   - baseline (no refusal)
   - calibrated refusal (fixed tau on test)

Notes:
- This pilot intentionally keeps the q_k definition simple and auditable. We set
  q_k based on whether a frame is supported by the verifier (material
  unsupported issues: severity>=2). This makes refusal interpretable as
  "refuse when verifier says evidence is insufficient" for the pilot.
- We do *not* apply a per-generation refusal cap here because LLaMA-2 outputs
  one frame by default; instead we enforce a global refusal_rate constraint.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..bet.encoders.simple_cnn3d import SimpleCNN3D
from ..bet.refine_loop import run_refine_loop
from ..data import make_dataloader
from ..eval.compute_budget import ComputeUnitCosts, match_b_enc_for_total_flops
from ..pcg.llama2_pcg import Llama2PCG, Llama2PCGConfig
from ..pcg.refusal import RefusalCalibrator, apply_refusal_to_generation
from ..pcg.schema_version import SCHEMA_VERSION
from ..types import Frame, Generation, Issue, Token
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier import verify
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION
from .utils import make_output_dir, save_results_json, set_seed


@dataclass
class Llama2RefusalPilotConfig:
    manifest_path: str
    dev_split: str = "val"
    test_split: str = "test"
    resize_shape: Tuple[int, int, int] = (64, 64, 64)
    seed: int = 0

    # Budgeting (FLOPs-matched total budget)
    flops_total: float = 5_000_000.0
    costs_json: str = "outputs/compute_costs.json"
    b_gen: int = 128
    n_verify: int = 1

    # Generation
    llama2_path: str = "/data/models/Llama-2-7b-chat-hf"
    llama2_quant: str = "fp16"  # fp16 | 8bit
    pcg_refresh_period: int = 5
    max_steps: int = 10

    # Dataset sampling
    n_samples: int = 50
    n_samples_test: int = 50

    # Calibration search / gates
    max_refusal_rate: float = 0.20
    max_refusal_ece: float = 0.15
    max_critical_miss_rate: float = 0.05
    candidate_taus: List[float] = None  # type: ignore[assignment]

    output_dir: str = "outputs/D1-llama2-refusal-pilot"


def _load_manifest_samples(
    manifest_path: str,
    *,
    split: str,
    resize_shape: Tuple[int, int, int],
    max_samples: int,
) -> List[Dict[str, Any]]:
    dl = make_dataloader(
        {
            "dataset_type": "manifest",
            "manifest_path": str(manifest_path),
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": int(max_samples),
            "resize_shape": tuple(int(x) for x in resize_shape),
        },
        split=str(split),
    )
    out: List[Dict[str, Any]] = []
    for batch in dl:
        out.append(
            {
                "volume": batch["volume"][0],
                "gt_frames": batch.get("frames", [[]])[0] or [],
            }
        )
    if not out:
        raise RuntimeError(f"No samples loaded for split={split!r}. Check manifest or split names.")
    return out


def _unsupported_frame_idxs(issues: List[Issue]) -> set[int]:
    idxs: set[int] = set()
    for iss in issues or []:
        try:
            sev = int(getattr(iss, "severity", 0))
        except Exception:
            sev = 0
        if sev < 2:
            continue
        if "unsupported" not in str(getattr(iss, "issue_type", "")).lower():
            continue
        try:
            idxs.add(int(getattr(iss, "frame_idx", -1)))
        except Exception:
            pass
    return idxs


def _override_q_from_issues(gen: Generation, issues: List[Issue]) -> Generation:
    """Define q_k in a simple, auditable way for the pilot.

    q_k=1 for frames without material unsupported issues, else 0.
    """
    unsupported = _unsupported_frame_idxs(issues)
    q = {}
    for k, fr in enumerate(gen.frames):
        if str(getattr(fr, "polarity", "")) not in ("present", "positive"):
            # Refusal is only meaningful for asserted positive claims; keep q high for others.
            q[int(k)] = 1.0
            continue
        q[int(k)] = 0.0 if int(k) in unsupported else 1.0
    return Generation(frames=gen.frames, citations=gen.citations, q=q, refusal=gen.refusal, text=gen.text)


def _run_llama2_once(
    *,
    volume: torch.Tensor,
    gt_frames: List[Frame],
    b_enc: int,
    pcg: Llama2PCG,
    encoder: torch.nn.Module,
    cfg: Llama2RefusalPilotConfig,
    sample_seed: int,
) -> Dict[str, Any]:
    res = run_refine_loop(
        volume=volume,
        budget_tokens=int(b_enc),
        steps=int(cfg.max_steps),
        generator_fn=lambda toks: pcg(toks),
        verifier_fn=lambda gen, toks: verify(gen, toks),
        emb_dim=32,
        seed=int(sample_seed),
        encoder=encoder,
        require_full_budget=False,
        use_evidence_head=False,
        pcg_refresh_period=int(cfg.pcg_refresh_period),
    )
    gen = _override_q_from_issues(res.gen, res.issues)
    return {"gen": gen, "tokens": res.tokens, "issues": res.issues, "gt_frames": gt_frames}


def _metrics_to_dict(m) -> Dict[str, Any]:  # noqa: ANN001
    return {
        "unsupported_rate": float(m.unsupported_rate),
        "critical_miss_rate": float(m.critical_miss_rate),
        "refusal_rate": float(m.refusal_rate),
        "refusal_ece": float(m.refusal_ece),
        "reliability_bins": m.reliability_bins,
        "critical_refusal_count": int(m.critical_refusal_count),
        "total_critical_count": int(m.total_critical_count),
    }


def _apply_refusal_and_reverify(
    *,
    calibrator: RefusalCalibrator,
    gens: List[Generation],
    toks: List[List[Token]],
) -> Tuple[List[Generation], List[List[Issue]]]:
    out_gens: List[Generation] = []
    out_issues: List[List[Issue]] = []
    for g, t in zip(gens, toks):
        gu = apply_refusal_to_generation(g, calibrator, max_refusal_rate=None)
        out_gens.append(gu)
        out_issues.append(verify(gu, t))
    return out_gens, out_issues


def _select_tau(
    *,
    gens_dev: List[Generation],
    toks_dev: List[List[Token]],
    gts_dev: List[List[Frame]],
    issues_dev: List[List[Issue]],
    cfg: Llama2RefusalPilotConfig,
) -> Tuple[float, Dict[str, Any]]:
    if cfg.candidate_taus is None:
        candidate_taus = sorted({0.0, *[i / 10.0 for i in range(1, 10)], 0.95, 0.99, 1.0})
    else:
        candidate_taus = [float(x) for x in cfg.candidate_taus]

    best_tau = 0.0
    best = None
    best_u = float("inf")
    best_rr = float("inf")

    for tau in candidate_taus:
        calibrator = RefusalCalibrator(
            tau_refuse=float(tau),
            max_critical_miss_rate=float(cfg.max_critical_miss_rate),
            num_bins=10,
        )
        gens_u, issues_u = _apply_refusal_and_reverify(calibrator=calibrator, gens=gens_dev, toks=toks_dev)
        metrics = calibrator.compute_calibration_metrics(gens_u, gts_dev, issues_u)

        ok = True
        ok = ok and (float(metrics.critical_miss_rate) <= float(cfg.max_critical_miss_rate) + 1e-12)
        ok = ok and (float(metrics.refusal_rate) <= float(cfg.max_refusal_rate) + 1e-12)
        ok = ok and (float(metrics.refusal_ece) <= float(cfg.max_refusal_ece) + 1e-12)
        if not ok:
            continue

        u = float(metrics.unsupported_rate)
        rr = float(metrics.refusal_rate)
        is_better = (u < best_u - 1e-12) or (abs(u - best_u) <= 1e-12 and rr < best_rr - 1e-12)
        if is_better:
            best_tau = float(tau)
            best = metrics
            best_u = float(u)
            best_rr = float(rr)

    if best is None:
        # Fall back to "no refusal" if nothing satisfies gates.
        calibrator = RefusalCalibrator(
            tau_refuse=0.0,
            max_critical_miss_rate=float(cfg.max_critical_miss_rate),
            num_bins=10,
        )
        gens_u, issues_u = _apply_refusal_and_reverify(calibrator=calibrator, gens=gens_dev, toks=toks_dev)
        best = calibrator.compute_calibration_metrics(gens_u, gts_dev, issues_u)
        best_tau = 0.0

    return float(best_tau), _metrics_to_dict(best)


def run_pilot(cfg: Llama2RefusalPilotConfig) -> Dict[str, Any]:
    set_seed(int(cfg.seed))

    repo_root = Path(__file__).resolve().parents[2]
    data_revision, split_manifest_path = try_manifest_revision(str(cfg.manifest_path))
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

    costs = ComputeUnitCosts.from_json(str(cfg.costs_json)) if str(cfg.costs_json) else ComputeUnitCosts()
    b_enc = match_b_enc_for_total_flops(
        flops_total=float(cfg.flops_total),
        b_gen=int(cfg.b_gen),
        n_verify=int(cfg.n_verify),
        costs=costs,
        min_b_enc=1,
        max_b_enc=4096,
    )

    encoder = SimpleCNN3D(in_channels=1, emb_dim=32).to("cuda").eval()
    pcg = Llama2PCG(
        Llama2PCGConfig(
            model_path=str(cfg.llama2_path),
            device="cuda",
            quantization=str(cfg.llama2_quant),
            max_new_tokens=max(128, int(cfg.b_gen)),
            temperature=0.0,
            topk_citations=3,
            tau_refuse=0.55,
        )
    )

    dev_samples = _load_manifest_samples(
        str(cfg.manifest_path),
        split=str(cfg.dev_split),
        resize_shape=tuple(int(x) for x in cfg.resize_shape),
        max_samples=int(cfg.n_samples),
    )
    test_samples = _load_manifest_samples(
        str(cfg.manifest_path),
        split=str(cfg.test_split),
        resize_shape=tuple(int(x) for x in cfg.resize_shape),
        max_samples=int(cfg.n_samples_test),
    )

    def _run_split(samples: List[Dict[str, Any]], *, seed_offset: int) -> Tuple[List[Generation], List[List[Token]], List[List[Issue]], List[List[Frame]]]:
        gens: List[Generation] = []
        toks: List[List[Token]] = []
        issues: List[List[Issue]] = []
        gts: List[List[Frame]] = []
        for i, s in enumerate(samples):
            out = _run_llama2_once(
                volume=s["volume"],
                gt_frames=s["gt_frames"],
                b_enc=int(b_enc),
                pcg=pcg,
                encoder=encoder,
                cfg=cfg,
                sample_seed=int(cfg.seed) + int(seed_offset) + int(i),
            )
            gens.append(out["gen"])
            toks.append(out["tokens"])
            issues.append(out["issues"])
            gts.append(out["gt_frames"])
        return gens, toks, issues, gts

    gens_dev, toks_dev, issues_dev, gts_dev = _run_split(dev_samples, seed_offset=0)
    gens_test, toks_test, issues_test, gts_test = _run_split(test_samples, seed_offset=10_000)

    tau, best_metrics_dev = _select_tau(
        gens_dev=gens_dev,
        toks_dev=toks_dev,
        gts_dev=gts_dev,
        issues_dev=issues_dev,
        cfg=cfg,
    )

    # Dev baseline vs calibrated
    cal_dev = RefusalCalibrator(tau_refuse=float(tau), max_critical_miss_rate=float(cfg.max_critical_miss_rate), num_bins=10)
    gens_dev_cal, issues_dev_cal = _apply_refusal_and_reverify(calibrator=cal_dev, gens=gens_dev, toks=toks_dev)
    m_dev_cal = cal_dev.compute_calibration_metrics(gens_dev_cal, gts_dev, issues_dev_cal)
    m_dev_base = cal_dev.compute_calibration_metrics(gens_dev, gts_dev, issues_dev)

    # Test baseline vs calibrated (fixed tau)
    gens_test_cal, issues_test_cal = _apply_refusal_and_reverify(calibrator=cal_dev, gens=gens_test, toks=toks_test)
    m_test_cal = cal_dev.compute_calibration_metrics(gens_test_cal, gts_test, issues_test_cal)
    m_test_base = cal_dev.compute_calibration_metrics(gens_test, gts_test, issues_test)

    policy = {
        "tau_refuse": float(tau),
        "max_critical_miss_rate": float(cfg.max_critical_miss_rate),
        "max_refusal_rate": float(cfg.max_refusal_rate),
        "num_bins": 10,
        # For this pilot, q_k is defined deterministically from verifier support; keep explicit.
        "q_definition": "verifier_support_binary(severity>=2)",
    }

    report: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "config": asdict(cfg),
        "budget": {
            "mode": "flops",
            "flops_total": float(cfg.flops_total),
            "b_enc": int(b_enc),
            "b_gen": int(cfg.b_gen),
            "n_verify": int(cfg.n_verify),
        },
        "best_tau": float(tau),
        "refusal_policy": policy,
        "best_metrics_dev": best_metrics_dev,
        "dev": {
            "split": str(cfg.dev_split),
            "no_refusal": _metrics_to_dict(m_dev_base),
            "calibrated": _metrics_to_dict(m_dev_cal),
        },
        "test": {
            "split": str(cfg.test_split),
            "no_refusal": _metrics_to_dict(m_test_base),
            "calibrated": _metrics_to_dict(m_test_cal),
        },
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Pilot: refusal calibration with LLaMA-2 PCG (dev→test frozen tau).")
    ap.add_argument("--manifest", type=str, required=True, help="Manifest jsonl path.")
    ap.add_argument("--dev-split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--test-split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--n-samples-test", type=int, default=50)
    ap.add_argument("--flops-total", type=float, default=5_000_000.0)
    ap.add_argument("--costs-json", type=str, default="outputs/compute_costs.json")
    ap.add_argument("--b-gen", type=int, default=128)
    ap.add_argument("--n-verify", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--pcg-refresh-period", type=int, default=5)
    ap.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])
    ap.add_argument("--max-refusal-rate", type=float, default=0.20)
    ap.add_argument("--max-refusal-ece", type=float, default=0.15)
    ap.add_argument("--max-critical-miss-rate", type=float, default=0.05)
    ap.add_argument("--candidate-taus", type=float, nargs="*", default=None)
    ap.add_argument("--output-dir", type=str, default="outputs/D1-llama2-refusal-pilot")
    args = ap.parse_args()

    cfg = Llama2RefusalPilotConfig(
        manifest_path=str(args.manifest),
        dev_split=str(args.dev_split),
        test_split=str(args.test_split),
        resize_shape=tuple(int(x) for x in args.resize_shape),
        seed=int(args.seed),
        n_samples=int(args.n_samples),
        n_samples_test=int(args.n_samples_test),
        flops_total=float(args.flops_total),
        costs_json=str(args.costs_json),
        b_gen=int(args.b_gen),
        n_verify=int(args.n_verify),
        max_steps=int(args.max_steps),
        pcg_refresh_period=int(args.pcg_refresh_period),
        llama2_path=str(args.llama2_path),
        llama2_quant=str(args.llama2_quant),
        max_refusal_rate=float(args.max_refusal_rate),
        max_refusal_ece=float(args.max_refusal_ece),
        max_critical_miss_rate=float(args.max_critical_miss_rate),
        candidate_taus=list(args.candidate_taus) if args.candidate_taus is not None else None,
        output_dir=str(args.output_dir),
    )

    out_dir = make_output_dir(cfg.output_dir, "figX_refusal_calibration_llama2_pilot")
    base_dir = Path(cfg.output_dir)
    cfg = Llama2RefusalPilotConfig(**{**asdict(cfg), "output_dir": out_dir})

    report = run_pilot(cfg)
    os.makedirs(out_dir, exist_ok=True)
    save_results_json(report, os.path.join(out_dir, "figX_refusal_calibration_llama2_pilot.json"))
    save_results_json(report["refusal_policy"], os.path.join(out_dir, "refusal_policy.json"))

    # Stable "latest" at base output dir.
    base_dir.mkdir(parents=True, exist_ok=True)
    save_results_json(report, str(base_dir / "figX_refusal_calibration_llama2_pilot.json"))
    save_results_json(report["refusal_policy"], str(base_dir / "refusal_policy.json"))
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()

