#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class MetricsKey:
    split: str
    method: str
    weighting: str  # "finding_weighted" or "case_weighted"
    metric: str  # e.g. "hit_rate_global_dice"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metric(d: Dict[str, Any], key: MetricsKey) -> float | None:
    by_split = d.get("by_split") or {}
    rec = ((by_split.get(key.split) or {}).get(key.method) or {}).get(key.weighting) or {}
    val = rec.get(key.metric)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _combo_out_dir(
    out_root: Path,
    *,
    mask_ratio: float,
    min_size: int,
    connectivity: int,
    use_laterality: bool,
    use_superior_inferior: bool,
) -> Path:
    lat = "lat1" if use_laterality else "lat0"
    si = "si1" if use_superior_inferior else "si0"
    # Keep the dir name stable and filesystem-friendly.
    r = str(mask_ratio).replace(".", "p")
    return out_root / f"r{r}_m{int(min_size)}_c{int(connectivity)}_{lat}_{si}"


def _build_verify_cmd(
    *,
    verify_script: Path,
    manifest: Path,
    splits: List[str],
    out_dir: Path,
    saliency_weights: Path,
    device: str,
    resize_shape: Tuple[int, int, int],
    clip_hu: Tuple[float, float],
    mask_ratio: float,
    min_size: int,
    connectivity: int,
    global_hit_thr: float,
    max_cases: int | None,
    methods: List[str],
    use_laterality: bool,
    use_superior_inferior: bool,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(verify_script),
        "--manifest",
        str(manifest),
        "--splits",
        *[str(s) for s in splits],
        "--out-dir",
        str(out_dir),
        "--saliency-weights",
        str(saliency_weights),
        "--device",
        str(device),
        "--resize-shape",
        str(int(resize_shape[0])),
        str(int(resize_shape[1])),
        str(int(resize_shape[2])),
        "--clip-hu",
        str(float(clip_hu[0])),
        str(float(clip_hu[1])),
        "--mask-ratio",
        str(float(mask_ratio)),
        "--min-size",
        str(int(min_size)),
        "--connectivity",
        str(int(connectivity)),
        "--global-hit-thr",
        str(float(global_hit_thr)),
        "--methods",
        *[str(m) for m in methods],
    ]
    if max_cases is not None and int(max_cases) > 0:
        cmd += ["--max-cases", str(int(max_cases))]
    if not use_laterality:
        cmd.append("--no-laterality")
    if not use_superior_inferior:
        cmd.append("--no-superior-inferior")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Grid-sweep ReXrank local verification parameters (mask_ratio/min_size/connectivity/heuristics)\n"
            "by repeatedly running scripts/external/verify_rexrank_manifest.py and aggregating its JSON outputs."
        )
    )
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--splits", type=str, nargs="+", default=["val"])
    ap.add_argument("--saliency-weights", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--resize-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--clip-hu", type=float, nargs=2, default=[-1000.0, 1000.0])
    ap.add_argument("--methods", type=str, nargs="+", default=["components_greedy"])
    ap.add_argument("--mask-ratios", type=float, nargs="+", required=True)
    ap.add_argument("--min-sizes", type=int, nargs="+", required=True)
    ap.add_argument("--connectivities", type=int, nargs="+", default=[2])
    ap.add_argument("--laterality", type=str, default="on", choices=["on", "off", "both"])
    ap.add_argument("--superior-inferior", type=str, default="on", choices=["on", "off", "both"])
    ap.add_argument("--global-hit-thr", type=float, default=0.1)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument(
        "--objective",
        type=str,
        default="hit_rate_global_dice",
        help="Metric to maximize (from finding_weighted or case_weighted).",
    )
    ap.add_argument("--weighting", type=str, default="finding_weighted", choices=["finding_weighted", "case_weighted"])
    ap.add_argument("--objective-split", type=str, default="val")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = Path(args.manifest).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    saliency_weights = Path(args.saliency_weights).resolve()

    verify_script = Path(__file__).resolve().parent / "verify_rexrank_manifest.py"
    if not verify_script.exists():
        raise FileNotFoundError(f"verify script not found: {verify_script}")

    max_cases = int(args.max_cases) if int(args.max_cases) > 0 else None
    resize_shape = (int(args.resize_shape[0]), int(args.resize_shape[1]), int(args.resize_shape[2]))
    clip_hu = (float(args.clip_hu[0]), float(args.clip_hu[1]))

    lat_grid = [True] if args.laterality == "on" else ([False] if args.laterality == "off" else [True, False])
    si_grid = [True] if args.superior_inferior == "on" else ([False] if args.superior_inferior == "off" else [True, False])

    objective_key = MetricsKey(
        split=str(args.objective_split),
        method=str(args.methods[0]),
        weighting=str(args.weighting),
        metric=str(args.objective),
    )

    combos = list(
        itertools.product(
            [float(x) for x in args.mask_ratios],
            [int(x) for x in args.min_sizes],
            [int(x) for x in args.connectivities],
            lat_grid,
            si_grid,
        )
    )
    print(f"[sweep] {len(combos)} combo(s) -> {out_root}", flush=True)

    rows: List[Dict[str, Any]] = []
    for mask_ratio, min_size, connectivity, use_lat, use_si in combos:
        out_dir = _combo_out_dir(
            out_root,
            mask_ratio=mask_ratio,
            min_size=min_size,
            connectivity=connectivity,
            use_laterality=bool(use_lat),
            use_superior_inferior=bool(use_si),
        )
        out_json = out_dir / "verify_rexrank_manifest.json"

        if not out_json.exists() and not bool(args.dry_run):
            cmd = _build_verify_cmd(
                verify_script=verify_script,
                manifest=manifest,
                splits=list(args.splits),
                out_dir=out_dir,
                saliency_weights=saliency_weights,
                device=str(args.device),
                resize_shape=resize_shape,
                clip_hu=clip_hu,
                mask_ratio=float(mask_ratio),
                min_size=int(min_size),
                connectivity=int(connectivity),
                global_hit_thr=float(args.global_hit_thr),
                max_cases=max_cases,
                methods=list(args.methods),
                use_laterality=bool(use_lat),
                use_superior_inferior=bool(use_si),
            )
            print(f"[sweep] run {out_dir.name}", flush=True)
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                rows.append(
                    {
                        "mask_ratio": float(mask_ratio),
                        "min_size": int(min_size),
                        "connectivity": int(connectivity),
                        "use_laterality": bool(use_lat),
                        "use_superior_inferior": bool(use_si),
                        "status": "failed",
                        "returncode": int(proc.returncode),
                        "out_dir": str(out_dir),
                    }
                )
                continue

        if not out_json.exists():
            rows.append(
                {
                    "mask_ratio": float(mask_ratio),
                    "min_size": int(min_size),
                    "connectivity": int(connectivity),
                    "use_laterality": bool(use_lat),
                    "use_superior_inferior": bool(use_si),
                    "status": "missing_output",
                    "out_dir": str(out_dir),
                }
            )
            continue

        d = _load_json(out_json)
        obj = _extract_metric(d, objective_key)

        row = {
            "mask_ratio": float(mask_ratio),
            "min_size": int(min_size),
            "connectivity": int(connectivity),
            "use_laterality": bool(use_lat),
            "use_superior_inferior": bool(use_si),
            "status": "ok",
            "objective": float(obj) if obj is not None else None,
            "out_dir": str(out_dir),
        }

        # Record a few commonly-read metrics for the first method on all requested splits.
        by_split = d.get("by_split") or {}
        for split in list(args.splits):
            srec = by_split.get(split) or {}
            for method in list(args.methods):
                wrec = ((srec.get(method) or {}).get(args.weighting) or {})
                if not isinstance(wrec, dict):
                    continue
                prefix = f"{split}:{method}:{args.weighting}:"
                for k in ["mean_dice", "mean_iou", "hit_rate_global_dice", "hit_rate_any_intersection"]:
                    v = wrec.get(k)
                    if isinstance(v, (int, float)):
                        row[prefix + k] = float(v)
        rows.append(row)

    # Sort best-first by objective.
    def key_fn(r: Dict[str, Any]) -> Tuple[int, float]:
        if r.get("status") != "ok" or r.get("objective") is None:
            return (1, float("-inf"))
        return (0, float(r["objective"]))

    rows_sorted = sorted(rows, key=key_fn, reverse=True)
    topk = rows_sorted[:10]
    print("\n[sweep] top configs:", flush=True)
    for r in topk:
        print(
            f"  obj={r.get('objective')} mask_ratio={r.get('mask_ratio')} min_size={r.get('min_size')} "
            f"conn={r.get('connectivity')} lat={r.get('use_laterality')} si={r.get('use_superior_inferior')} "
            f"status={r.get('status')} out={Path(str(r.get('out_dir'))).name}",
            flush=True,
        )

    out_summary = out_root / "sweep_summary.json"
    out_summary.write_text(json.dumps({"rows": rows_sorted, "objective": objective_key.__dict__}, indent=2) + "\n", encoding="utf-8")
    print(f"\n[sweep] wrote {out_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

