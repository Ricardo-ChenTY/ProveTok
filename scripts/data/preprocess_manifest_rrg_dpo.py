from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path when running as `python scripts/data/*.py ...`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.data.io import load_mask, load_volume_and_affine
from provetok.data.manifest_schema import ManifestRecord, get_record_mask_path, load_manifest, save_manifest_jsonl
from provetok.data.preprocess_rrg_dpo import RRGDPOPreprocessSpec, preprocess_rrg_dpo


def _human_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.2f} GB"


def _resolve_path(p: str, *, base_dir: Path) -> str:
    if not p:
        return p
    pp = Path(str(p))
    if pp.is_absolute():
        return str(pp)
    return str((base_dir / pp).resolve())


def _save_volume_npz(
    out_path: Path,
    *,
    volume: np.ndarray,
    affine_zyx: Optional[np.ndarray],
    meta: Dict[str, Any],
    compress: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kw: Dict[str, Any] = {
        "volume": np.asarray(volume),
        "orig_shape_dhw": np.asarray(meta.get("orig_shape_dhw", []), dtype=np.int32),
        "orig_spacing_zyx": np.asarray(meta.get("orig_spacing_zyx", []), dtype=np.float32),
        "target_shape_dhw": np.asarray(meta.get("target_shape_dhw", []), dtype=np.int32),
        "target_spacing_zyx": np.asarray(meta.get("target_spacing_zyx", []), dtype=np.float32),
    }
    if affine_zyx is not None:
        kw["affine_zyx"] = np.asarray(affine_zyx, dtype=np.float64)
    if compress:
        np.savez_compressed(str(out_path), **kw)
    else:
        np.savez(str(out_path), **kw)


def _save_mask_npz(out_path: Path, *, mask: np.ndarray, compress: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a single key so load_mask() works via z.files[0].
    kw = {"mask": np.asarray(mask)}
    if compress:
        np.savez_compressed(str(out_path), **kw)
    else:
        np.savez(str(out_path), **kw)


def _preprocess_mask(
    mask: np.ndarray,
    *,
    affine_zyx: np.ndarray,
    spec: RRGDPOPreprocessSpec,
) -> np.ndarray:
    if mask.ndim == 4:
        out = []
        for k in range(mask.shape[0]):
            mk = torch.from_numpy(mask[k].astype(np.float32))
            mk2, _, _ = preprocess_rrg_dpo(mk, affine_zyx=affine_zyx, spec=spec, is_mask=True)
            out.append((mk2.detach().cpu().numpy() > 0.5).astype(np.uint8))
        return np.stack(out, axis=0)

    mk = torch.from_numpy(mask.astype(np.float32))
    mk2, _, _ = preprocess_rrg_dpo(mk, affine_zyx=affine_zyx, spec=spec, is_mask=True)
    return (mk2.detach().cpu().numpy() > 0.5).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Preprocess a manifest with RRG-DPO comparable settings (pp.md §6.2):\n"
            "- resample to 0.75×0.75×1.5mm^3\n"
            "- center crop/pad to 480×480×240 voxels\n\n"
            "This writes new .npz volumes (and masks when present) with an embedded affine_zyx."
        )
    )
    ap.add_argument("--in-manifest", type=str, required=True, help="Input manifest.jsonl")
    ap.add_argument("--out-root", type=str, required=True, help="Output root directory")
    ap.add_argument("--out-manifest", type=str, default="", help="Output manifest.jsonl (default: <out-root>/manifest_rrg_dpo.jsonl)")
    ap.add_argument("--dataset-name", type=str, default="", help="Override dataset name in output manifest")
    ap.add_argument("--splits", type=str, nargs="*", default=None, help="Optional split allowlist (e.g., train val test)")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="Saved volume dtype")
    ap.add_argument("--no-masks", action="store_true", help="Do not preprocess masks even if present")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-compress", action="store_true", help="Disable np.savez_compressed")

    # Use xyz order for CLI ergonomics (matches pp.md text). Internally we convert to zyx/dhw.
    ap.add_argument(
        "--target-spacing-xyz",
        type=float,
        nargs=3,
        default=[0.75, 0.75, 1.5],
        help="Target spacing as (sx,sy,sz) in mm (default: 0.75 0.75 1.5)",
    )
    ap.add_argument(
        "--target-shape-xyz",
        type=int,
        nargs=3,
        default=[480, 480, 240],
        help="Target shape as (X,Y,Z) voxels (default: 480 480 240)",
    )

    args = ap.parse_args()

    in_manifest = Path(args.in_manifest).resolve()
    out_root = Path(args.out_root).resolve()
    out_manifest = Path(args.out_manifest).resolve() if args.out_manifest else (out_root / "manifest_rrg_dpo.jsonl")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    spacing_xyz = tuple(float(x) for x in args.target_spacing_xyz)
    shape_xyz = tuple(int(x) for x in args.target_shape_xyz)
    target_spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    target_shape_dhw = (int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]))
    spec = RRGDPOPreprocessSpec(target_spacing_zyx=target_spacing_zyx, target_shape_dhw=target_shape_dhw)

    recs = load_manifest(str(in_manifest))
    if args.splits:
        allow = {str(s) for s in args.splits}
        recs = [r for r in recs if str(r.split) in allow]

    if args.max_records and args.max_records > 0:
        recs = recs[: int(args.max_records)]

    vol_dir = out_root / "volumes_rrg_dpo"
    mask_dir = out_root / "masks_rrg_dpo"

    saved_dtype = np.float16 if str(args.dtype) == "float16" else np.float32
    compress = not bool(args.no_compress)

    total_out_bytes_est = 0
    out_records: List[ManifestRecord] = []
    num_skipped_existing = 0
    num_masks = 0

    for i, r in enumerate(recs):
        scan_hash = str(r.scan_hash)
        if not scan_hash:
            raise SystemExit(f"Record {i} has empty scan_hash")

        vol_out = vol_dir / f"{scan_hash}.npz"
        mask_out = mask_dir / f"{scan_hash}.npz"

        # Resolve input paths relative to the manifest dir when needed.
        in_vol_path = _resolve_path(str(r.volume_path), base_dir=in_manifest.parent)

        if vol_out.exists() and (not args.overwrite):
            num_skipped_existing += 1
            # Still update manifest to point to preprocessed outputs.
            dd = r.to_dict()
            dd["volume_path"] = str(vol_out)
            if (not args.no_masks) and get_record_mask_path(r):
                dd["mask_path"] = str(mask_out)
            if args.dataset_name:
                dd["dataset"] = str(args.dataset_name)
            dd["preprocess_rrg_dpo"] = True
            out_records.append(ManifestRecord.from_dict(dd))
            continue

        vol, affine_zyx = load_volume_and_affine(in_vol_path)
        if affine_zyx is None:
            raise SystemExit(f"Missing affine for {in_vol_path} (need NIfTI or .npz with affine_zyx)")

        vol2, aff2, meta = preprocess_rrg_dpo(vol, affine_zyx=affine_zyx, spec=spec, is_mask=False)
        vol_np = vol2.detach().cpu().numpy().astype(saved_dtype, copy=False)
        _save_volume_npz(vol_out, volume=vol_np, affine_zyx=aff2, meta=meta, compress=compress)

        mp = None if bool(args.no_masks) else get_record_mask_path(r)
        if mp:
            in_mask_path = _resolve_path(str(mp), base_dir=in_manifest.parent)
            m = load_mask(in_mask_path)
            if isinstance(m, np.ndarray):
                mask_np = _preprocess_mask(m.astype(bool), affine_zyx=affine_zyx, spec=spec)
                _save_mask_npz(mask_out, mask=mask_np, compress=compress)
                num_masks += 1

        dd = r.to_dict()
        dd["volume_path"] = str(vol_out)
        if mp:
            dd["mask_path"] = str(mask_out)
        if args.dataset_name:
            dd["dataset"] = str(args.dataset_name)
        dd["preprocess_rrg_dpo"] = True
        dd["preprocess_rrg_dpo_meta"] = meta
        dd["preprocess_rrg_dpo_in_volume_path"] = str(in_vol_path)
        out_records.append(ManifestRecord.from_dict(dd))

        # Rough estimate: target array bytes (uncompressed).
        total_out_bytes_est += int(np.prod(spec.target_shape_dhw)) * int(np.dtype(saved_dtype).itemsize)

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(recs)}] wrote {vol_out.name} (est_total={_human_gb(total_out_bytes_est)})")

    save_manifest_jsonl(out_records, str(out_manifest))
    meta_out = out_manifest.with_suffix(out_manifest.suffix + ".meta.json")
    meta_out.write_text(
        json.dumps(
            {
                "in_manifest": str(in_manifest),
                "out_root": str(out_root),
                "out_manifest": str(out_manifest),
                "num_in": len(recs),
                "num_out": len(out_records),
                "num_masks": int(num_masks),
                "skipped_existing": int(num_skipped_existing),
                "target_spacing_xyz": [float(x) for x in spacing_xyz],
                "target_shape_xyz": [int(x) for x in shape_xyz],
                "target_spacing_zyx": [float(x) for x in target_spacing_zyx],
                "target_shape_dhw": [int(x) for x in target_shape_dhw],
                "saved_dtype": str(args.dtype),
                "npz_compressed": bool(compress),
                "out_bytes_est_uncompressed": int(total_out_bytes_est),
                "out_gb_est_uncompressed": float(total_out_bytes_est) / 1e9,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_manifest": str(out_manifest),
                "meta": str(meta_out),
                "num_records": len(out_records),
                "num_masks": int(num_masks),
                "out_gb_est_uncompressed": float(total_out_bytes_est) / 1e9,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    # Reduce OpenMP oversubscription in multiprocess environments.
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
