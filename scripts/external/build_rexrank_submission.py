from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, generate_binary_structure, label

# Ensure repo root is on sys.path when running as `python scripts/*.py ...`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.models.saliency_cnn3d import load_saliency_cnn3d


def _infer_ct_rate_path(name: str) -> str:
    """Map a CT filename to the CT-RATE repo path (matches scripts/data/download_ct_rate_mini_from_rex.py)."""
    if not name.endswith(".nii.gz"):
        raise ValueError(f"Expected .nii.gz filename, got {name!r}")
    stem = name[: -len(".nii.gz")]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected name format: {name!r}")
    split = parts[0]
    pid = f"{parts[0]}_{parts[1]}"
    series = f"{parts[0]}_{parts[1]}_{parts[2]}"
    return f"dataset/{split}/{pid}/{series}/{name}"


def _load_rex_dataset_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and any(k in data for k in ("train", "val", "test")):
        rows: List[Dict[str, Any]] = []
        for k in ("train", "val", "test"):
            v = data.get(k)
            if isinstance(v, list):
                for r in v:
                    if isinstance(r, dict):
                        rr = dict(r)
                        rr["_split"] = str(k)
                        rows.append(rr)
        return rows
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        rows = []
        for r in data["data"]:
            if isinstance(r, dict):
                rr = dict(r)
                rr["_split"] = str(r.get("split", "unknown"))
                rows.append(rr)
        return rows
    if isinstance(data, list):
        rows = []
        for r in data:
            if isinstance(r, dict):
                rr = dict(r)
                rr["_split"] = str(r.get("split", "unknown"))
                rows.append(rr)
        return rows
    raise ValueError(f"Unexpected ReX dataset.json format: {type(data)}")


def _sorted_findings(findings: Dict[str, Any]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for k, v in (findings or {}).items():
        if not isinstance(v, str) or not v.strip():
            continue
        try:
            idx = int(k)
        except Exception:
            continue
        out.append((idx, v.strip()))
    out.sort(key=lambda x: x[0])
    return out


def _clip_and_scale_hu(vol: torch.Tensor, *, clip_hu: Tuple[float, float]) -> torch.Tensor:
    v = vol.float()
    lo, hi = float(clip_hu[0]), float(clip_hu[1])
    v = v.clamp(min=lo, max=hi)
    v = v / max(abs(lo), abs(hi), 1.0)
    return v


def _topk_binary_mask(prob: np.ndarray, *, ratio: float, min_voxels: int = 1) -> np.ndarray:
    if prob.size == 0:
        return np.zeros_like(prob, dtype=bool)
    r = float(max(0.0, min(1.0, ratio)))
    k = int(round(r * float(prob.size)))
    k = max(int(min_voxels), min(int(prob.size), k))
    flat = prob.reshape(-1)
    # Pick exactly top-k voxels.
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros_like(flat, dtype=bool)
    out[idx] = True
    return out.reshape(prob.shape)


def _filter_small_components(mask: np.ndarray, *, min_size: int, connectivity: int) -> np.ndarray:
    if not mask.any():
        return mask
    struct = generate_binary_structure(3, int(connectivity))
    lbl, n = label(mask.astype(bool), structure=struct)
    if n <= 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(lbl.reshape(-1))
    keep = [i for i in range(1, int(n) + 1) if int(sizes[i]) >= int(min_size)]
    if not keep:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(lbl, keep)


@dataclass(frozen=True)
class _Component:
    cid: int
    size: int
    center_zyx: Tuple[float, float, float]


def _components(mask: np.ndarray, *, connectivity: int, min_size: int) -> Tuple[np.ndarray, List[_Component]]:
    mask = _filter_small_components(mask, min_size=int(min_size), connectivity=int(connectivity))
    struct = generate_binary_structure(3, int(connectivity))
    lbl, n = label(mask.astype(bool), structure=struct)
    if n <= 0:
        return lbl, []
    sizes = np.bincount(lbl.reshape(-1))
    comp_ids = [i for i in range(1, int(n) + 1) if int(sizes[i]) >= int(min_size)]
    if not comp_ids:
        return lbl, []
    centers = center_of_mass(mask.astype(np.uint8), labels=lbl, index=comp_ids)
    comps = [
        _Component(cid=int(cid), size=int(sizes[int(cid)]), center_zyx=(float(c[0]), float(c[1]), float(c[2])))
        for cid, c in zip(comp_ids, centers)
    ]
    comps.sort(key=lambda x: (-int(x.size), int(x.cid)))
    return lbl, comps


_LOC_PATTERNS: List[Tuple[str, str]] = [
    (r"\\brul\\b|right upper lobe", "RUL"),
    (r"\\brml\\b|right middle lobe", "RML"),
    (r"\\brll\\b|right lower lobe", "RLL"),
    (r"\\blul\\b|left upper lobe", "LUL"),
    (r"\\blll\\b|left lower lobe", "LLL"),
    (r"lingula", "lingula"),
]


def _infer_laterality(text: str) -> str:
    t = str(text).lower()
    if "bilateral" in t or "both" in t:
        return "bilateral"
    if "left" in t:
        return "left"
    if "right" in t:
        return "right"
    return "unspecified"


def _infer_loc_bucket(text: str) -> str:
    t = str(text).lower()
    for pat, loc in _LOC_PATTERNS:
        if re.search(pat, t):
            return loc
    return "unspecified"


def _score_component_for_finding(
    comp: _Component,
    *,
    vol_shape_zyx: Tuple[int, int, int],
    laterality: str,
    loc: str,
    use_laterality: bool,
    use_superior_inferior: bool,
) -> float:
    dz, dy, dx = (float(vol_shape_zyx[0]), float(vol_shape_zyx[1]), float(vol_shape_zyx[2]))
    z, y, x = comp.center_zyx
    z_norm = float(z / max(dz - 1.0, 1.0))
    x_norm = float(x / max(dx - 1.0, 1.0))

    score = 0.0
    if use_laterality:
        lat = str(laterality)
        if lat == "left":
            score += 1.0 if x_norm < 0.5 else -1.0
        elif lat == "right":
            score += 1.0 if x_norm >= 0.5 else -1.0

    if use_superior_inferior:
        bucket = str(loc)
        if bucket in ("RUL", "LUL"):
            score += 0.6 if z_norm < 0.4 else -0.3
        elif bucket in ("RLL", "LLL"):
            score += 0.6 if z_norm > 0.6 else -0.3
        elif bucket in ("RML", "lingula"):
            score += 0.4 if (0.4 <= z_norm <= 0.6) else -0.2

    # Prefer larger components (weak prior).
    score += 0.1 * math.log1p(float(comp.size))
    return float(score)


def _assign_components_greedy(
    findings: List[Tuple[int, str]],
    *,
    comps: List[_Component],
    lbl: np.ndarray,
    vol_shape_zyx: Tuple[int, int, int],
    use_laterality: bool,
    use_superior_inferior: bool,
) -> Dict[int, np.ndarray]:
    if not findings:
        return {}
    if not comps:
        return {int(i): np.zeros(vol_shape_zyx, dtype=bool) for i, _ in findings}

    remaining = list(comps)
    out: Dict[int, np.ndarray] = {}
    for fi, sent in findings:
        lat = _infer_laterality(sent)
        loc = _infer_loc_bucket(sent)
        best = None
        best_score = None
        for c in remaining:
            s = _score_component_for_finding(
                c,
                vol_shape_zyx=vol_shape_zyx,
                laterality=lat,
                loc=loc,
                use_laterality=bool(use_laterality),
                use_superior_inferior=bool(use_superior_inferior),
            )
            if best_score is None or s > best_score:
                best_score = s
                best = c
        if best is None:
            out[int(fi)] = np.zeros(vol_shape_zyx, dtype=bool)
            continue
        out[int(fi)] = (lbl == int(best.cid))
        remaining = [c for c in remaining if int(c.cid) != int(best.cid)]
        if not remaining:
            remaining = list(comps)  # allow reuse once exhausted
    return out


def _save_4d_mask_nifti(
    out_path: Path,
    *,
    masks_zyx: Dict[int, np.ndarray],
    findings: List[Tuple[int, str]],
    affine: np.ndarray,
    header: nib.Nifti1Header,
) -> None:
    # ReXGroundingCT convention: (F, H, W, D) == (F, X, Y, Z)
    # Our masks are internal (Z, Y, X). Convert each channel to (X, Y, Z).
    f = len(findings)
    if f <= 0:
        raise ValueError("No findings; cannot save 4D mask")

    # Allocate in uint8 to minimize disk footprint and match eval expectations (>0).
    # Shape: (F, X, Y, Z)
    any_mask = next(iter(masks_zyx.values())) if masks_zyx else None
    if any_mask is None:
        raise ValueError("masks_zyx empty")
    z, y, x = any_mask.shape
    out = np.zeros((f, x, y, z), dtype=np.uint8)

    for j, (fi, _) in enumerate(findings):
        m = masks_zyx.get(int(fi))
        if m is None:
            continue
        if m.shape != (z, y, x):
            raise ValueError(f"Mask shape mismatch for finding={fi}: got {m.shape}, expected {(z,y,x)}")
        out[j] = np.transpose(m.astype(np.uint8), (2, 1, 0))  # (X,Y,Z)

    # Do NOT reuse the CT header: it can force float dtypes (bloats disk) and
    # may carry incompatible dimensional metadata. Keep the affine, but write a
    # clean uint8 segmentation volume.
    img = nib.Nifti1Image(out, affine=affine)
    img.set_data_dtype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


def _zip_dir(src_dir: Path, *, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(src_dir.glob("*.nii.gz")):
            zf.write(str(p), arcname=p.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a ReXrankCT submission zip from local CT volumes.")
    ap.add_argument("--rex-dataset-json", type=str, default="/data/provetok_datasets/rexgroundingct_raw/dataset.json")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--ct-rate-root", type=str, default="/data/tiasha/CT-RATE")
    ap.add_argument("--out-dir", type=str, default="outputs/E0190-rexrank-submission")
    ap.add_argument(
        "--saliency-weights",
        type=str,
        default="outputs/E0155-train_saliency_cnn3d_100g/saliency_cnn3d.pt",
        help="SaliencyCNN3D weights for union-mask prediction.",
    )
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--clip-hu", type=float, nargs=2, default=[-1000.0, 1000.0])
    ap.add_argument("--mask-ratio", type=float, default=0.005, help="Keep top-ratio voxels in the coarse 64^3 mask.")
    ap.add_argument("--min-size", type=int, default=50, help="Min component size in upsampled mask.")
    ap.add_argument("--connectivity", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument(
        "--assign",
        type=str,
        default="components_greedy",
        choices=["replicate", "components_greedy"],
        help="How to produce per-finding masks from the union saliency mask.",
    )
    ap.add_argument("--no-laterality", action="store_true", help="Disable laterality-based component assignment.")
    ap.add_argument("--no-superior-inferior", action="store_true", help="Disable upper/lower component assignment.")
    ap.add_argument(
        "--allow-missing-ct",
        action="store_true",
        help="Do not fail the run if some CT volumes are missing; still write partial predictions.",
    )
    ap.add_argument("--max-cases", type=int, default=0, help="Optional cap on number of cases (debug).")
    ap.add_argument("--zip", action="store_true", help="Also create a submission zip at <out-dir>/submission.zip")
    args = ap.parse_args()

    rex_json = Path(args.rex_dataset_json).resolve()
    ct_root = Path(args.ct_rate_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    pred_dir = out_dir / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not rex_json.exists():
        raise SystemExit(f"--rex-dataset-json not found: {rex_json}")
    if not ct_root.exists():
        raise SystemExit(f"--ct-rate-root not found: {ct_root}")

    rows = _load_rex_dataset_json(rex_json)
    want_split = str(args.split).strip().lower()
    rows = [r for r in rows if str(r.get("_split", "")).strip().lower() == want_split]
    if not rows:
        raise SystemExit(f"No rows found for split={want_split!r} in {rex_json}")
    if int(args.max_cases) and int(args.max_cases) > 0:
        rows = rows[: int(args.max_cases)]

    # Only keep rows we can actually submit: 3D volume with at least 1 finding.
    usable_rows = []
    for r in rows:
        name = str(r.get("name") or "")
        if not name.endswith(".nii.gz"):
            continue
        findings = _sorted_findings(r.get("findings") or {})
        if not findings:
            continue
        usable_rows.append(r)
    rows = usable_rows
    if not rows:
        raise SystemExit(f"No usable rows found for split={want_split!r} in {rex_json}")
    expected = len(rows)

    device = torch.device(str(args.device))
    model = load_saliency_cnn3d(str(args.saliency_weights), map_location="cpu").to(device).eval()
    resize_shape = tuple(int(x) for x in args.resize_shape)
    clip_hu = (float(args.clip_hu[0]), float(args.clip_hu[1]))

    started = time.time()
    n_ok = 0
    n_missing = 0
    for i, r in enumerate(rows, start=1):
        fname = Path(str(r.get("name") or "")).name
        findings = _sorted_findings(r.get("findings") or {})

        vol_path = ct_root / _infer_ct_rate_path(fname)
        if not vol_path.exists():
            # Fallback to a slow search for robustness.
            hits = list(ct_root.rglob(fname))
            if hits:
                vol_path = hits[0]
        if not vol_path.exists():
            n_missing += 1
            if (n_missing % 10) == 0:
                print(f"[rexrank] missing_ct={n_missing} last={fname}", flush=True)
            continue

        out_path = pred_dir / fname
        if out_path.exists():
            n_ok += 1
            continue

        # Load CT as provider-native (X,Y,Z), then convert to internal (Z,Y,X) for the saliency model.
        ct_img = nib.load(str(vol_path))
        ct_arr = np.asanyarray(ct_img.dataobj).astype(np.float32, copy=False)  # (X,Y,Z)
        vol_zyx = np.transpose(ct_arr, (2, 1, 0))  # (Z,Y,X)
        vol_t = torch.from_numpy(vol_zyx)
        vol_t = _clip_and_scale_hu(vol_t, clip_hu=clip_hu)

        # Resize on CPU to keep GPU memory stable.
        x_small = F.interpolate(
            vol_t.unsqueeze(0).unsqueeze(0),
            size=resize_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]  # (d,h,w)

        with torch.no_grad():
            prob_small = model.predict_proba(x_small.unsqueeze(0).unsqueeze(0).to(device=device))[0, 0].detach().cpu()
        prob_np = prob_small.numpy()
        mask_small = _topk_binary_mask(prob_np, ratio=float(args.mask_ratio), min_voxels=1)

        # Upsample the coarse binary mask back to original resolution (internal Z,Y,X).
        m_up = F.interpolate(
            torch.from_numpy(mask_small.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            size=tuple(int(x) for x in vol_t.shape),
            mode="nearest",
        )[0, 0].numpy() > 0.5

        lbl, comps = _components(m_up, connectivity=int(args.connectivity), min_size=int(args.min_size))
        if str(args.assign) == "replicate":
            per_finding = {int(fi): m_up for fi, _ in findings}
        else:
            per_finding = _assign_components_greedy(
                findings,
                comps=comps,
                lbl=lbl,
                vol_shape_zyx=tuple(int(x) for x in m_up.shape),
                use_laterality=(not bool(args.no_laterality)),
                use_superior_inferior=(not bool(args.no_superior_inferior)),
            )

        _save_4d_mask_nifti(
            out_path,
            masks_zyx=per_finding,
            findings=findings,
            affine=ct_img.affine,
            header=ct_img.header,
        )
        n_ok += 1
        if (n_ok % 10) == 0:
            elapsed = time.time() - started
            print(f"[rexrank] done={n_ok}/{expected} missing_ct={n_missing} elapsed={elapsed/60:.1f}min", flush=True)

    elapsed = time.time() - started
    print(json.dumps({"split": want_split, "ok": n_ok, "missing_ct": n_missing, "pred_dir": str(pred_dir), "elapsed_min": elapsed / 60.0}, indent=2))

    if (not bool(args.allow_missing_ct)) and n_missing:
        raise SystemExit(f"Missing CT volumes: missing_ct={n_missing} expected={expected} ct_rate_root={ct_root}")
    if (not bool(args.allow_missing_ct)) and (n_ok != expected):
        raise SystemExit(f"Incomplete submission: ok={n_ok} expected={expected} missing_ct={n_missing} out_dir={out_dir}")

    if bool(args.zip):
        zip_path = out_dir / "submission.zip"
        _zip_dir(pred_dir, zip_path=zip_path)
        print(json.dumps({"zip": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
