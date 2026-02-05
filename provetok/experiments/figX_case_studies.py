"""Fig X: Oral-ready case studies (qualitative + auditable artifacts).

This runner produces 3–5 per-sample audit bundles that can be opened during an
oral Q&A to explain:
- which Ω regions are covered by tokens
- which top-k citations were used
- what the verifier flagged (unsupported/overclaim/etc.)
- whether refusal was triggered and why

Outputs per case:
- <output_dir>/case_<scan_hash>/case.json
- <output_dir>/case_<scan_hash>/case.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..bet.encoders.simple_cnn3d import SimpleCNN3D
from ..bet.refine_loop import run_refine_loop
from ..data.frame_extractor import FrameExtractor, frames_to_report
from ..data.io import load_mask, load_volume
from ..data.manifest_schema import ManifestRecord, load_manifest
from ..eval.compute_budget import ComputeUnitCosts, format_budget_report, match_b_enc_for_total_flops
from ..eval.metrics_grounding import compute_generation_grounding, tokens_to_mask, union_lesion_masks
from ..pcg.llama2_pcg import Llama2PCG, Llama2PCGConfig
from ..pcg.schema_version import SCHEMA_VERSION
from ..types import Generation, Issue, Token
from ..utils.artifact import build_artifact_meta, try_manifest_revision
from ..verifier import verify
from ..verifier.rules import RULE_SET_VERSION
from ..verifier.taxonomy import TAXONOMY_VERSION


def _resize_volume(vol: torch.Tensor, *, resize_shape: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(int(x) for x in vol.shape) == tuple(int(x) for x in resize_shape):
        return vol
    x = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    y = F.interpolate(x, size=tuple(int(x) for x in resize_shape), mode="trilinear", align_corners=False)
    return y[0, 0]


def _resize_mask(mask: np.ndarray, *, resize_shape: Tuple[int, int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 3:
        return mask
    if tuple(int(x) for x in mask.shape) == tuple(int(x) for x in resize_shape):
        return mask.astype(bool)
    x = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    src = tuple(int(s) for s in mask.shape)
    tgt = tuple(int(s) for s in resize_shape)
    if all(t <= s for t, s in zip(tgt, src)):
        y = F.adaptive_max_pool3d(x, output_size=tgt)
    else:
        y = F.interpolate(x, size=tgt, mode="nearest")
    return (y[0, 0].numpy() > 0.5)


def _issue_to_dict(iss: Issue) -> Dict[str, Any]:
    return {
        "frame_idx": int(iss.frame_idx),
        "issue_type": str(iss.issue_type),
        "severity": int(iss.severity),
        "rule_id": str(iss.rule_id),
        "message": str(iss.message),
        "evidence_trace": iss.evidence_trace or {},
    }


def _token_to_dict(tok: Token) -> Dict[str, Any]:
    return {
        "token_id": int(tok.token_id),
        "cell_id": str(tok.cell_id),
        "level": int(tok.level),
        "score": float(tok.score),
        "uncertainty": float(tok.uncertainty),
    }


def _select_positive_cited_ids(gen: Generation) -> List[int]:
    ids: set[int] = set()
    for idx, fr in enumerate(gen.frames):
        if str(fr.polarity) in ("present", "positive"):
            for tid in gen.citations.get(idx, []):
                ids.add(int(tid))
    return sorted(ids)


def _save_case_png(
    *,
    out_path: Path,
    volume: torch.Tensor,
    lesion_union: np.ndarray,
    cited_union: np.ndarray,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "E0163 plotting requires matplotlib. Install via `pip install matplotlib` "
            "or set E0163 to output arrays only."
        ) from e

    vol = volume.detach().cpu().float().numpy()
    vol = np.clip(vol, -1000.0, 1000.0) / 1000.0

    # Pick the slice with maximum lesion voxels; fall back to center slice.
    if isinstance(lesion_union, np.ndarray) and lesion_union.ndim == 3 and lesion_union.any():
        z = int(np.argmax(lesion_union.sum(axis=(1, 2))))
    else:
        z = int(vol.shape[0] // 2)

    img = vol[z]
    les = lesion_union[z] if lesion_union.ndim == 3 else np.zeros_like(img, dtype=bool)
    cite = cited_union[z] if cited_union.ndim == 3 else np.zeros_like(img, dtype=bool)

    fig = plt.figure(figsize=(6, 6), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap="gray", vmin=-1.0, vmax=1.0)
    if les.any():
        ax.imshow(np.ma.masked_where(~les, les), cmap="Reds", alpha=0.35)
    if cite.any():
        ax.imshow(np.ma.masked_where(~cite, cite), cmap="Blues", alpha=0.25)
    ax.set_title(title)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def _load_records(manifest_path: str, *, split: str) -> List[ManifestRecord]:
    recs = [r for r in load_manifest(manifest_path) if r.split == split]
    if not recs:
        raise RuntimeError(f"No manifest records for split={split!r} in {manifest_path!r}")
    return recs


def _pick_records(
    records: Sequence[ManifestRecord],
    *,
    sample_ids: Sequence[str],
    n_cases: int,
    seed: int,
) -> List[ManifestRecord]:
    if sample_ids:
        wanted = set(str(x) for x in sample_ids)
        picked = [r for r in records if r.scan_hash in wanted]
        missing = sorted(wanted.difference({r.scan_hash for r in picked}))
        if missing:
            raise RuntimeError(f"sample_ids not found in manifest split: {missing[:10]} (showing up to 10)")
        return picked

    n_cases = max(1, int(n_cases))
    rng = np.random.RandomState(int(seed))
    # Deterministic population order
    pool = sorted(list(records), key=lambda r: r.scan_hash)
    idx = rng.choice(len(pool), size=min(n_cases, len(pool)), replace=False)
    return [pool[int(i)] for i in idx.tolist()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate oral-ready qualitative case studies (case.json + case.png).")
    ap.add_argument("--manifest", type=str, required=True, help="Manifest jsonl path (ReXGroundingCT-100g).")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--n-cases", type=int, default=5)
    ap.add_argument("--sample-ids", type=str, nargs="*", default=[], help="Optional scan_hash list to select cases.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--budget-mode", type=str, default="flops", choices=["flops", "tokens"])
    ap.add_argument("--flops-total", type=float, default=5_000_000.0)
    ap.add_argument("--budget-tokens", type=int, default=256)
    ap.add_argument("--costs-json", type=str, default="outputs/compute_costs.json")
    ap.add_argument("--b-gen", type=int, default=128)
    ap.add_argument("--n-verify", type=int, default=1)

    ap.add_argument("--pcg-refresh-period", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--topk-citations", type=int, default=3)
    ap.add_argument(
        "--require-full-budget",
        action="store_true",
        help="Spend b_enc_target when possible (may increase steps significantly).",
    )

    ap.add_argument("--llama2-path", type=str, default="/data/models/Llama-2-7b-chat-hf")
    ap.add_argument("--llama2-quant", type=str, default="fp16", choices=["fp16", "8bit"])
    ap.add_argument("--encoder-device", type=str, default="cuda")

    ap.add_argument("--output-dir", type=str, default="outputs/E0163-cases")
    args = ap.parse_args()

    resize_shape = tuple(int(x) for x in args.resize_shape)
    out_root = Path(str(args.output_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    # Meta
    repo_root = Path(__file__).resolve().parents[2]
    data_revision, split_manifest_path = try_manifest_revision(str(args.manifest))
    meta = build_artifact_meta(
        repo_root=repo_root,
        seed=int(args.seed),
        config={
            "manifest": str(args.manifest),
            "split": str(args.split),
            "resize_shape": list(resize_shape),
            "n_cases": int(args.n_cases),
            "sample_ids": list(args.sample_ids),
            "budget_mode": str(args.budget_mode),
            "flops_total": float(args.flops_total),
            "budget_tokens": int(args.budget_tokens),
            "costs_json": str(args.costs_json),
            "b_gen": int(args.b_gen),
            "n_verify": int(args.n_verify),
            "pcg_refresh_period": int(args.pcg_refresh_period),
            "max_steps": int(args.max_steps),
            "topk_citations": int(args.topk_citations),
            "require_full_budget": bool(args.require_full_budget),
            "llama2_path": str(args.llama2_path),
            "llama2_quant": str(args.llama2_quant),
            "encoder_device": str(args.encoder_device),
            "output_dir": str(args.output_dir),
        },
        rule_set_version=RULE_SET_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
        data_revision=data_revision,
        split_manifest_path=split_manifest_path,
    )

    # Load records and pick cases.
    records = _load_records(str(args.manifest), split=str(args.split))
    cases = _pick_records(records, sample_ids=list(args.sample_ids), n_cases=int(args.n_cases), seed=int(args.seed))

    # Shared models (avoid repeated loads).
    costs = ComputeUnitCosts.from_json(str(args.costs_json)) if str(args.costs_json) else ComputeUnitCosts()
    encoder = SimpleCNN3D(in_channels=1, emb_dim=32).to(str(args.encoder_device)).eval()
    pcg = Llama2PCG(
        Llama2PCGConfig(
            model_path=str(args.llama2_path),
            device="cuda",
            quantization=str(args.llama2_quant),
            max_new_tokens=max(128, int(args.b_gen)),
            temperature=0.0,
            topk_citations=int(args.topk_citations),
        )
    )

    extractor = FrameExtractor()

    index: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "cases": [],
    }

    for i, r in enumerate(cases):
        case_seed = int(args.seed) + 10_000 + int(i)
        case_dir = out_root / f"case_{r.scan_hash}"
        case_dir.mkdir(parents=True, exist_ok=True)

        # Load + resize
        vol = load_volume(r.volume_path)
        vol = _resize_volume(vol, resize_shape=resize_shape)
        frames_gt = extractor.extract_frames(r.report_text)

        lesion_masks: Dict[int, Any] = {}
        mask_path = r.extra.get("mask_path")
        if mask_path:
            try:
                m = load_mask(str(mask_path))
                if isinstance(m, np.ndarray) and m.ndim == 4:
                    lesion_masks = {int(j): _resize_mask(m[j], resize_shape=resize_shape) for j in range(m.shape[0])}
                elif isinstance(m, np.ndarray) and m.ndim == 3:
                    lesion_masks = {0: _resize_mask(m, resize_shape=resize_shape)}
            except Exception:
                lesion_masks = {}

        # Budget selection
        if str(args.budget_mode) == "flops":
            b_enc = match_b_enc_for_total_flops(
                flops_total=float(args.flops_total),
                b_gen=int(args.b_gen),
                n_verify=int(args.n_verify),
                costs=costs,
                min_b_enc=1,
                max_b_enc=4096,
            )
            budget_info = {
                "mode": "flops",
                "flops_total_target": float(args.flops_total),
                "b_enc_target": int(b_enc),
            }
        else:
            b_enc = int(args.budget_tokens)
            budget_info = {
                "mode": "tokens",
                "b_enc_target": int(b_enc),
            }

        # Run refine loop (audit-friendly settings)
        result = run_refine_loop(
            volume=vol,
            budget_tokens=int(b_enc),
            steps=int(args.max_steps),
            generator_fn=lambda toks: pcg(toks),
            verifier_fn=lambda gen, toks: verify(gen, toks),
            emb_dim=32,
            seed=case_seed,
            encoder=encoder,
            require_full_budget=bool(args.require_full_budget),
            use_evidence_head=False,
            pcg_refresh_period=int(args.pcg_refresh_period),
        )

        gen = result.gen
        tokens = result.tokens
        issues = result.issues

        volume_shape = tuple(int(x) for x in vol.shape)
        grounding = compute_generation_grounding(gen, tokens, lesion_masks, volume_shape)

        lesion_union = union_lesion_masks(lesion_masks, volume_shape)
        cited_ids = _select_positive_cited_ids(gen)
        token_by_id = {int(t.token_id): t for t in tokens}
        cited_tokens = [token_by_id[int(tid)] for tid in cited_ids if int(tid) in token_by_id]
        cited_union = tokens_to_mask(cited_tokens, volume_shape)

        inter = int(np.logical_and(lesion_union, cited_union).sum())
        uni = int(np.logical_or(lesion_union, cited_union).sum())
        iou = float(inter / uni) if uni > 0 else 0.0

        # Save PNG
        png_path = case_dir / "case.png"
        title = f"{r.scan_hash} | iou={iou:.4f} | frames={len(gen.frames)} | issues={len(issues)}"
        _save_case_png(out_path=png_path, volume=vol, lesion_union=lesion_union, cited_union=cited_union, title=title)

        # Save JSON (no tensors)
        tokens_sorted = sorted(tokens, key=lambda t: (-float(t.score), int(t.token_id)))
        tokens_top = [_token_to_dict(t) for t in tokens_sorted[:50]]
        case_json: Dict[str, Any] = {
            "meta": meta.to_dict(),
            "record": {
                "scan_hash": r.scan_hash,
                "volume_path": r.volume_path,
                "mask_path": str(mask_path or ""),
                "split": r.split,
                "dataset": r.dataset,
            },
            "budget": budget_info,
            "report_text": r.report_text,
            "gt_frames": [asdict(f) for f in frames_gt],
            "gt_report_canonical": frames_to_report(frames_gt),
            "generation": {
                "frames": [asdict(f) for f in gen.frames],
                "citations": {str(k): [int(x) for x in v] for k, v in (gen.citations or {}).items()},
                "q": {str(k): float(v) for k, v in (gen.q or {}).items()},
                "refusal": {str(k): bool(v) for k, v in (gen.refusal or {}).items()},
                "text": str(gen.text or ""),
            },
            "issues": [_issue_to_dict(x) for x in issues],
            "trace": [asdict(t) for t in result.trace],
            "tokens_top": tokens_top,
            "grounding": {k: float(v) for k, v in grounding.items()},
            "masks_summary": {
                "lesion_union_voxels": int(lesion_union.astype(bool).sum()),
                "cited_union_voxels": int(cited_union.astype(bool).sum()),
                "intersection_voxels": int(inter),
                "union_voxels": int(uni),
                "iou_union": float(iou),
            },
            "paths": {
                "png": str(png_path),
            },
        }
        (case_dir / "case.json").write_text(json.dumps(case_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        index["cases"].append(
            {
                "scan_hash": r.scan_hash,
                "dir": str(case_dir),
                "png": str(png_path),
                "json": str(case_dir / "case.json"),
                "iou_union": float(iou),
                "num_frames": int(len(gen.frames)),
                "num_issues": int(len(issues)),
            }
        )

    (out_root / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(index['cases'])} cases -> {out_root}")


if __name__ == "__main__":
    # Reduce OpenMP oversubscription when loading many volumes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
