from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from provetok.bet.tokenize import TokenEncoder
from provetok.data.frame_extractor import FrameExtractor
from provetok.data.manifest_schema import ManifestRecord, load_manifest
from provetok.data.io import load_volume_and_affine
from provetok.grid.cells import Cell, root_cell
from provetok.pcg.llama2_pcg import Llama2PCGConfig, build_llama2_json_prompt


def _resize_volume(vol: torch.Tensor, *, resize_shape: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(int(x) for x in vol.shape) == tuple(int(x) for x in resize_shape):
        return vol
    x = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    y = F.interpolate(x, size=tuple(int(x) for x in resize_shape), mode="trilinear", align_corners=False)
    return y[0, 0]


def _init_cells(*, init_level: int, budget_tokens: int) -> List[Cell]:
    init_level = int(max(0, int(init_level)))
    while init_level > 0:
        n = 2 ** int(init_level)
        if int(n * n * n) <= int(budget_tokens):
            break
        init_level -= 1
    if init_level <= 0:
        return [root_cell()]
    n = 2 ** int(init_level)
    return [Cell(level=init_level, ix=ix, iy=iy, iz=iz) for ix in range(n) for iy in range(n) for iz in range(n)]


def _target_requires_citations(contract_mode: str) -> bool:
    mode = str(contract_mode or "").strip().lower()
    return mode in ("schema_citations", "full")


def _frame_to_min_dict(fr: Any) -> Dict[str, Any]:
    # Keep only the minimal keys requested by build_llama2_json_prompt.
    return {
        "finding": str(getattr(fr, "finding", "normal")),
        "polarity": str(getattr(fr, "polarity", "present")),
        "laterality": str(getattr(fr, "laterality", "unspecified")),
        "confidence": float(getattr(fr, "confidence", 0.5)),
    }


def _record_iter(manifest_path: str, *, split: str) -> List[ManifestRecord]:
    recs = [r for r in load_manifest(str(manifest_path)) if str(r.split) == str(split)]
    recs = sorted(recs, key=lambda r: str(r.scan_hash))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a tiny JSONL SFT dataset for LoRA smoke (pp.md §6.6).")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--max-records", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--init-level", type=int, default=1)
    ap.add_argument("--budget-tokens", type=int, default=256)
    ap.add_argument("--emb-dim", type=int, default=32)

    ap.add_argument("--max-tokens-in-prompt", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=1)
    ap.add_argument(
        "--contract-mode",
        type=str,
        default="schema_only",
        choices=["free_form", "schema_only", "schema_citations", "full"],
    )
    ap.add_argument("--topk-citations", type=int, default=3)

    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--out-meta", type=str, default="")

    args = ap.parse_args()

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    recs = _record_iter(args.manifest, split=str(args.split))
    if not recs:
        raise SystemExit("No records found for split.")

    max_n = int(args.max_records)
    if max_n > 0:
        recs = recs[:max_n]

    cfg = Llama2PCGConfig(
        model_path="",
        max_tokens_in_prompt=int(args.max_tokens_in_prompt),
        max_frames=int(args.max_frames),
        contract_mode=str(args.contract_mode),
        topk_citations=int(args.topk_citations),
    )
    extractor = FrameExtractor()

    rng = np.random.RandomState(int(args.seed))

    rows: List[Dict[str, Any]] = []

    for i, r in enumerate(recs):
        vol, affine_zyx = load_volume_and_affine(r.volume_path, seed=int(args.seed) + int(i))
        vol = _resize_volume(vol, resize_shape=tuple(int(x) for x in args.resize_shape))

        cells = _init_cells(init_level=int(args.init_level), budget_tokens=int(args.budget_tokens))
        enc = TokenEncoder(volume=vol, emb_dim=int(args.emb_dim), seed=int(args.seed), affine_zyx=affine_zyx)
        tokens = enc.encode(cells)

        prompt = build_llama2_json_prompt(tokens, cfg=cfg, max_tokens_in_prompt=int(args.max_tokens_in_prompt))

        frames = extractor.extract_frames(str(r.report_text or ""))
        frame_dicts = [_frame_to_min_dict(fr) for fr in frames][: max(0, int(args.max_frames))]
        if not frame_dicts:
            # Fallback: keep schema-valid minimal frame.
            frame_dicts = [{"finding": "opacity", "polarity": "present", "laterality": "unspecified", "confidence": 0.5}]

        # Minimal target JSON.
        tgt: Dict[str, Any] = {
            "frames": list(frame_dicts),
            "citations": {},
            "q": {"0": float(frame_dicts[0].get("confidence", 0.5)) if frame_dicts else 0.5},
            "refusal": {"0": False},
        }

        if _target_requires_citations(str(args.contract_mode)) and frame_dicts:
            ranked = sorted(tokens, key=lambda t: (-float(getattr(t, "score", 0.0)), int(getattr(t, "token_id", 0))))
            top = [int(getattr(t, "token_id", 0)) for t in ranked[: max(0, int(args.topk_citations))]]
            if top:
                tgt["citations"] = {"0": top}

        completion = json.dumps(tgt, ensure_ascii=False)

        rows.append(
            {
                "sample_id": str(r.scan_hash),
                "prompt": prompt,
                "completion": completion,
            }
        )

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.out_meta:
        meta = {
            "manifest": str(args.manifest),
            "split": str(args.split),
            "n_rows": int(len(rows)),
            "cfg": asdict(cfg),
        }
        Path(args.out_meta).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(str(out_jsonl))


if __name__ == "__main__":
    main()
