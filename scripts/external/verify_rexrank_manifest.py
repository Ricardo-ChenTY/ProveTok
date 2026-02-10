from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


GLOBAL_HIT_THR_DEFAULT = 0.1
DATA_DTYPE = np.uint8


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if isinstance(rec, dict):
            yield rec


def _sorted_findings(findings: Dict[str, Any], *, f_expected: Optional[int] = None) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for k, v in (findings or {}).items():
        if not isinstance(v, str):
            continue
        try:
            idx = int(k)
        except Exception:
            continue
        out.append((idx, v.strip()))
    out.sort(key=lambda x: x[0])
    if f_expected is not None and len(out) != int(f_expected):
        # Dataset json and mask channels should match; if not, fall back to dense indices.
        if len(out) < int(f_expected):
            used = {i for i, _ in out}
            for i in range(int(f_expected)):
                if i in used:
                    continue
                out.append((int(i), ""))
            out.sort(key=lambda x: x[0])
        else:
            out = out[: int(f_expected)]
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
    keep_ids = [i for i in range(1, int(n) + 1) if int(sizes[i]) >= int(min_size)]
    if not keep_ids:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(lbl, keep_ids)


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


def _compute_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _compute_dice(gt: np.ndarray, pred: np.ndarray, *, eps: float = 1e-6) -> float:
    inter = np.logical_and(gt, pred).sum()
    tot = int(gt.sum()) + int(pred.sum())
    if tot == 0:
        return 0.0
    return float((2 * inter + eps) / (tot + eps))


def _load_findings_map(dataset_json: Path) -> Dict[str, Dict[str, Any]]:
    d = _load_json(dataset_json)
    if not isinstance(d, dict) or not any(k in d for k in ("train", "val", "test")):
        raise ValueError(f"Unexpected dataset.json format: {dataset_json}")
    out: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        rows = d.get(split)
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            out[name] = r
    return out


def _predict_union_mask_zyx(
    vol_path: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
    resize_shape: Tuple[int, int, int],
    clip_hu: Tuple[float, float],
    mask_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Load provider-native volume: (X,Y,Z)
    ct_img = nib.load(str(vol_path))
    ct_arr = np.asanyarray(ct_img.dataobj).astype(np.float32, copy=False)
    vol_zyx = np.transpose(ct_arr, (2, 1, 0))  # (Z,Y,X)
    vol_t = torch.from_numpy(vol_zyx)
    vol_t = _clip_and_scale_hu(vol_t, clip_hu=clip_hu)

    # Resize on CPU for stability.
    x_small = F.interpolate(
        vol_t.unsqueeze(0).unsqueeze(0),
        size=resize_shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0]  # (d,h,w)

    with torch.no_grad():
        prob_small = model.predict_proba(x_small.unsqueeze(0).unsqueeze(0).to(device=device))[0, 0].detach().cpu()
    prob_np = prob_small.numpy()
    mask_small = _topk_binary_mask(prob_np, ratio=float(mask_ratio), min_voxels=1)

    # Upsample binary mask back to original internal (Z,Y,X).
    m_up = F.interpolate(
        torch.from_numpy(mask_small.astype(np.float32)).unsqueeze(0).unsqueeze(0),
        size=tuple(int(x) for x in vol_t.shape),
        mode="nearest",
    )[0, 0].numpy() > 0.5

    return m_up.astype(bool), ct_img.affine


def _save_pred_4d_nifti(pred_path: Path, *, per_finding_zyx: Dict[int, np.ndarray], f: int, affine: np.ndarray) -> None:
    # Convention: (F, X, Y, Z); internal masks are (Z,Y,X).
    any_mask = next(iter(per_finding_zyx.values())) if per_finding_zyx else None
    if any_mask is None:
        raise ValueError("per_finding_zyx empty")
    z, y, x = any_mask.shape
    out = np.zeros((int(f), int(x), int(y), int(z)), dtype=np.uint8)
    for fi, m in per_finding_zyx.items():
        if not (0 <= int(fi) < int(f)):
            continue
        if m.shape != (z, y, x):
            raise ValueError(f"Mask shape mismatch: got {m.shape}, expected {(z,y,x)}")
        out[int(fi)] = np.transpose(m.astype(np.uint8), (2, 1, 0))
    img = nib.Nifti1Image(out, affine=affine)
    img.set_data_dtype(np.uint8)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(pred_path))


def _eval_case(
    gt_mask_path: Path,
    *,
    pred_per_finding_zyx: Dict[int, np.ndarray],
    global_hit_thr: float,
) -> Dict[str, Any]:
    gt_img = nib.load(str(gt_mask_path))
    gt = np.asanyarray(gt_img.dataobj).astype(DATA_DTYPE, copy=False)  # (F,X,Y,Z)
    if gt.ndim != 4:
        raise ValueError(f"Expected 4D GT mask, got shape={gt.shape}")
    f, x, y, z = (int(gt.shape[0]), int(gt.shape[1]), int(gt.shape[2]), int(gt.shape[3]))

    # Evaluate in internal Z,Y,X to match our predicted masks.
    findings: List[Dict[str, Any]] = []
    for fi in range(int(f)):
        gt_xyz = gt[int(fi)]
        gt_zyx = np.transpose(gt_xyz, (2, 1, 0)) > 0
        pred_zyx = pred_per_finding_zyx.get(int(fi))
        if pred_zyx is None:
            pred_zyx = np.zeros((z, y, x), dtype=bool)
        if pred_zyx.shape != (z, y, x):
            raise ValueError(f"Pred shape mismatch: got {pred_zyx.shape}, expected {(z,y,x)}")

        inter = int(np.logical_and(gt_zyx, pred_zyx).sum())
        iou = _compute_iou(gt_zyx, pred_zyx)
        dice = _compute_dice(gt_zyx, pred_zyx)
        findings.append(
            {
                "finding_idx": int(fi),
                "gt_voxels": int(gt_zyx.sum()),
                "pred_voxels": int(pred_zyx.sum()),
                "intersection_voxels": int(inter),
                "iou": float(iou),
                "dice": float(dice),
                "hit_any_intersection": bool(inter > 0),
                "global_hit": bool(dice >= float(global_hit_thr)),
            }
        )

    dice_vals = [float(r["dice"]) for r in findings]
    iou_vals = [float(r["iou"]) for r in findings]
    global_hits = [bool(r["global_hit"]) for r in findings]
    any_hits = [bool(r["hit_any_intersection"]) for r in findings]
    return {
        "num_findings": int(f),
        "mean_dice": float(np.mean(dice_vals)) if dice_vals else 0.0,
        "mean_iou": float(np.mean(iou_vals)) if iou_vals else 0.0,
        "hit_rate_global_dice": float(np.mean(global_hits)) if global_hits else 0.0,
        "hit_rate_any_intersection": float(np.mean(any_hits)) if any_hits else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Verify the ReXrank-style submission pipeline on manifest subsets that include gold voxel masks.\n\n"
            "This is NOT the hidden-test evaluation. It runs on local manifests (e.g. rexgroundingct_100g/mini) where\n"
            "`mask_path` exists, and reports Dice/IoU/Hit on those cases to sanity-check the submission logic."
        )
    )
    ap.add_argument("--manifest", type=str, required=True, help="Manifest jsonl with volume_path + mask_path.")
    ap.add_argument("--rex-dataset-json", type=str, default="/data/provetok_datasets/rexgroundingct_raw/dataset.json")
    ap.add_argument("--splits", type=str, nargs="+", default=["val", "test"])
    ap.add_argument("--out-dir", type=str, default="outputs/E0192-rexrank-manifest-verify")
    ap.add_argument(
        "--saliency-weights",
        type=str,
        default="outputs/E0155-train_saliency_cnn3d_100g/saliency_cnn3d.pt",
        help="SaliencyCNN3D weights for union-mask prediction.",
    )
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--clip-hu", type=float, nargs=2, default=[-1000.0, 1000.0])
    ap.add_argument("--mask-ratio", type=float, default=0.005)
    ap.add_argument("--min-size", type=int, default=50)
    ap.add_argument("--connectivity", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--global-hit-thr", type=float, default=GLOBAL_HIT_THR_DEFAULT)
    ap.add_argument("--no-laterality", action="store_true")
    ap.add_argument("--no-superior-inferior", action="store_true")
    ap.add_argument("--max-cases", type=int, default=0, help="Optional cap per split (debug).")
    ap.add_argument("--save-preds", action="store_true", help="Also write predicted 4D masks under <out-dir>/pred/...")
    ap.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["components_greedy"],
        choices=["components_greedy", "replicate"],
        help="Prediction mapping from union mask to per-finding channels.",
    )
    args = ap.parse_args()

    started_at_utc = _now_utc()

    manifest = Path(args.manifest).resolve()
    rex_dataset_json = Path(args.rex_dataset_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "verify_rexrank_manifest.json"

    if not manifest.exists():
        raise SystemExit(f"--manifest not found: {manifest}")
    if not rex_dataset_json.exists():
        raise SystemExit(f"--rex-dataset-json not found: {rex_dataset_json}")

    findings_map = _load_findings_map(rex_dataset_json)

    want_splits = {str(s).strip() for s in args.splits}
    rows = [r for r in _iter_jsonl(manifest) if str(r.get("split") or "").strip() in want_splits]
    if not rows:
        raise SystemExit(f"No rows found for splits={sorted(want_splits)} in {manifest}")

    # Resume support
    processed: set[str] = set()
    existing_cases: List[Dict[str, Any]] = []
    if report_path.exists():
        try:
            prev = _load_json(report_path)
            if isinstance(prev, dict):
                for c in prev.get("cases") or []:
                    if isinstance(c, dict) and c.get("rex_name"):
                        processed.add(str(c["rex_name"]))
                existing_cases = [c for c in (prev.get("cases") or []) if isinstance(c, dict)]
        except Exception:
            processed = set()
            existing_cases = []

    device = torch.device(str(args.device))
    model = load_saliency_cnn3d(str(args.saliency_weights), map_location="cpu").to(device).eval()
    resize_shape = tuple(int(x) for x in args.resize_shape)
    clip_hu = (float(args.clip_hu[0]), float(args.clip_hu[1]))

    methods = [str(m) for m in args.methods]
    started = time.time()
    cases: List[Dict[str, Any]] = list(existing_cases)

    # Apply max-cases per split (after resume filtering)
    if int(args.max_cases) and int(args.max_cases) > 0:
        limited: List[Dict[str, Any]] = []
        per_split: Dict[str, int] = {}
        for r in rows:
            s = str(r.get("split") or "")
            if per_split.get(s, 0) >= int(args.max_cases):
                continue
            limited.append(r)
            per_split[s] = per_split.get(s, 0) + 1
        rows = limited

    n_total = len(rows)
    n_done = 0
    n_skip = 0
    n_err = 0

    for i, r in enumerate(rows, start=1):
        rex_name = str(r.get("rex_name") or "").strip()
        split = str(r.get("split") or "").strip()
        vol_path = Path(str(r.get("volume_path") or "")).resolve()
        mask_path = Path(str(r.get("mask_path") or "")).resolve()

        if not rex_name:
            n_err += 1
            continue
        if rex_name in processed:
            n_skip += 1
            continue
        if not vol_path.exists() or not mask_path.exists():
            cases.append(
                {
                    "rex_name": rex_name,
                    "split": split,
                    "ok": False,
                    "error": "missing_volume_or_mask",
                    "volume_path": str(vol_path),
                    "mask_path": str(mask_path),
                }
            )
            processed.add(rex_name)
            n_err += 1
            if (i % 10) == 0:
                _atomic_write_json(
                    report_path,
                    {
                        "meta": {"generated_at_utc": _now_utc()},
                        "cases": cases,
                    },
                )
            continue

        try:
            union_zyx, affine = _predict_union_mask_zyx(
                vol_path,
                model=model,
                device=device,
                resize_shape=resize_shape,
                clip_hu=clip_hu,
                mask_ratio=float(args.mask_ratio),
            )

            gt_img = nib.load(str(mask_path))
            gt = np.asanyarray(gt_img.dataobj)
            if gt.ndim != 4:
                raise ValueError(f"Expected 4D GT mask, got {gt.shape}")
            f = int(gt.shape[0])

            # Gather findings text if available.
            entry = findings_map.get(rex_name) or {}
            findings = _sorted_findings(entry.get("findings") or {}, f_expected=f)
            if not findings:
                findings = [(int(i), "") for i in range(int(f))]

            lbl, comps = _components(union_zyx, connectivity=int(args.connectivity), min_size=int(args.min_size))
            per_method: Dict[str, Dict[int, np.ndarray]] = {}
            for m in methods:
                if m == "replicate":
                    per_method[m] = {int(fi): union_zyx for fi, _ in findings}
                else:
                    per_method[m] = _assign_components_greedy(
                        findings,
                        comps=comps,
                        lbl=lbl,
                        vol_shape_zyx=tuple(int(x) for x in union_zyx.shape),
                        use_laterality=(not bool(args.no_laterality)),
                        use_superior_inferior=(not bool(args.no_superior_inferior)),
                    )

            metrics_by_method: Dict[str, Any] = {}
            for m, pred_pf in per_method.items():
                metrics_by_method[m] = _eval_case(mask_path, pred_per_finding_zyx=pred_pf, global_hit_thr=float(args.global_hit_thr))
                if bool(args.save_preds):
                    pred_path = out_dir / "pred" / m / rex_name
                    _save_pred_4d_nifti(pred_path, per_finding_zyx=pred_pf, f=f, affine=affine)

            cases.append(
                {
                    "rex_name": rex_name,
                    "split": split,
                    "ok": True,
                    "volume_path": str(vol_path),
                    "mask_path": str(mask_path),
                    "num_findings": int(f),
                    "metrics": metrics_by_method,
                }
            )
            processed.add(rex_name)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            cases.append(
                {
                    "rex_name": rex_name,
                    "split": split,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "volume_path": str(vol_path),
                    "mask_path": str(mask_path),
                }
            )
            processed.add(rex_name)
            n_err += 1

        if (n_done + n_err) % 10 == 0:
            elapsed = time.time() - started
            print(f"[verify_rexrank_manifest] done={n_done}/{n_total} err={n_err} skip={n_skip} elapsed={elapsed/60:.1f}min", flush=True)
            _atomic_write_json(
                report_path,
                {
                    "meta": {"generated_at_utc": _now_utc()},
                    "cases": cases,
                },
            )

    # Summaries
    by_split: Dict[str, Any] = {}
    for s in sorted(want_splits):
        s_cases = [c for c in cases if c.get("split") == s and c.get("ok") and isinstance(c.get("metrics"), dict)]
        by_split[s] = {}
        for m in methods:
            m_cases = [c for c in s_cases if m in (c.get("metrics") or {})]
            total_cases = len(m_cases)
            total_findings = sum(int((c.get("metrics") or {}).get(m, {}).get("num_findings", 0)) for c in m_cases)

            # Case-weighted (each CT scan contributes equally).
            mean_dice_case = float(np.mean([float((c.get("metrics") or {})[m]["mean_dice"]) for c in m_cases])) if m_cases else 0.0
            mean_iou_case = float(np.mean([float((c.get("metrics") or {})[m]["mean_iou"]) for c in m_cases])) if m_cases else 0.0
            hit_gd_case = float(np.mean([float((c.get("metrics") or {})[m]["hit_rate_global_dice"]) for c in m_cases])) if m_cases else 0.0
            hit_any_case = float(np.mean([float((c.get("metrics") or {})[m]["hit_rate_any_intersection"]) for c in m_cases])) if m_cases else 0.0

            # Finding-weighted (each finding channel contributes equally).
            denom = float(total_findings) if total_findings else 0.0
            mean_dice_find = (
                float(
                    sum(
                        float((c.get("metrics") or {})[m]["mean_dice"]) * float((c.get("metrics") or {})[m]["num_findings"])
                        for c in m_cases
                    )
                    / denom
                )
                if denom > 0
                else 0.0
            )
            mean_iou_find = (
                float(
                    sum(
                        float((c.get("metrics") or {})[m]["mean_iou"]) * float((c.get("metrics") or {})[m]["num_findings"])
                        for c in m_cases
                    )
                    / denom
                )
                if denom > 0
                else 0.0
            )
            hit_gd_find = (
                float(
                    sum(
                        float((c.get("metrics") or {})[m]["hit_rate_global_dice"])
                        * float((c.get("metrics") or {})[m]["num_findings"])
                        for c in m_cases
                    )
                    / denom
                )
                if denom > 0
                else 0.0
            )
            hit_any_find = (
                float(
                    sum(
                        float((c.get("metrics") or {})[m]["hit_rate_any_intersection"])
                        * float((c.get("metrics") or {})[m]["num_findings"])
                        for c in m_cases
                    )
                    / denom
                )
                if denom > 0
                else 0.0
            )
            by_split[s][m] = {
                "total_cases": int(total_cases),
                "total_findings": int(total_findings),
                "case_weighted": {
                    "mean_dice": float(mean_dice_case),
                    "mean_iou": float(mean_iou_case),
                    "hit_rate_global_dice": float(hit_gd_case),
                    "hit_rate_any_intersection": float(hit_any_case),
                },
                "finding_weighted": {
                    "mean_dice": float(mean_dice_find),
                    "mean_iou": float(mean_iou_find),
                    "hit_rate_global_dice": float(hit_gd_find),
                    "hit_rate_any_intersection": float(hit_any_find),
                },
            }

    elapsed = time.time() - started
    report = {
        "meta": {
            "generated_at_utc": _now_utc(),
            "started_at_utc": started_at_utc,
            "elapsed_min": float(elapsed / 60.0),
            "device": str(device),
            "saliency_weights": str(Path(args.saliency_weights).resolve()),
            "resize_shape": list(resize_shape),
            "clip_hu": list(clip_hu),
            "mask_ratio": float(args.mask_ratio),
            "min_size": int(args.min_size),
            "connectivity": int(args.connectivity),
            "global_hit_thr": float(args.global_hit_thr),
            "methods": methods,
            "save_preds": bool(args.save_preds),
        },
        "dataset": {
            "manifest": str(manifest),
            "rex_dataset_json": str(rex_dataset_json),
            "splits": sorted(want_splits),
        },
        "by_split": by_split,
        "cases": cases,
    }
    _atomic_write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "cases": len(cases), "elapsed_min": elapsed / 60.0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
