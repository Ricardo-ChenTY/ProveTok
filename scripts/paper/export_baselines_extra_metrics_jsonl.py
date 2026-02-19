#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.data.manifest_schema import load_manifest


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict):
                yield d


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def _load_baselines_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sample_ids_from_manifest(manifest_path: Path, *, split: str, max_samples: int) -> List[str]:
    recs = [r for r in load_manifest(str(manifest_path)) if str(r.split) == str(split)]
    recs = sorted(recs, key=lambda r: str(r.scan_hash))
    if int(max_samples) > 0:
        recs = recs[: int(max_samples)]
    return [str(r.scan_hash) for r in recs]


def _load_sample_ids_from_pairs(pairs_jsonl: Path, *, method: str) -> List[str]:
    # pairs.jsonl contains one row per (sample_id, method). We recover a stable
    # sample order by scanning for a single method.
    out: List[str] = []
    seen = set()
    for d in _read_jsonl(pairs_jsonl):
        if str(d.get("method") or "") != str(method):
            continue
        sid = str(d.get("sample_id") or d.get("scan_hash") or "").strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _select_metric_keys(raw: Dict[str, List[float]], *, include: Optional[Sequence[str]]) -> List[str]:
    if include:
        keys = [str(k) for k in include]
        missing = [k for k in keys if k not in raw]
        if missing:
            raise SystemExit(f"Requested --metrics keys missing in baselines raw: {missing}")
        return keys

    # Default: pp.md Table1 + proof/grounding essentials.
    default = [
        # Grounding / NLG-proxy
        "frame_f1",
        "iou",
        "dice",
        "hit_rate",
        "hit_at_8",
        "coverage_at_8",
        "laterality_grounding_acc",
        "combined",
        # Taxonomy
        "unsupported",
        "overclaim",
        # Proof rules
        "r1_no_citation",
        "r2_coarse_only",
        "r3_laterality_mismatch",
        "r4_bilateral_separation",
        "weighted_issue",
    ]
    return [k for k in default if k in raw]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Export per-sample metrics from a run_baselines baselines.json into an extra_metrics_jsonl "+
            "compatible with scripts/paper/compute_paper_metrics.py --extra-metrics-jsonl.\n\n"
            "This solves a common pp.md Table1 need: merging proof/grounding metrics (from baselines.json) "+
            "with text/LLM metrics (from pairs.jsonl + external tools like RadEval)."
        )
    )
    ap.add_argument("--baselines-json", type=str, required=True, help="Path to baselines.json produced by run_baselines")
    ap.add_argument("--out-jsonl", type=str, required=True, help="Output JSONL path")

    # For recovering sample_id order
    ap.add_argument("--manifest", type=str, default="", help="Manifest used by run_baselines (preferred for sample order)")
    ap.add_argument("--split", type=str, default="", help="Split name when using --manifest (default: read from baselines.json config)")
    ap.add_argument("--max-samples", type=int, default=-1, help="Max samples when using --manifest (default: read from baselines.json config)")

    ap.add_argument("--pairs-jsonl", type=str, default="", help="Fallback: pairs.jsonl (used only if manifest order check fails)")
    ap.add_argument("--pairs-method", type=str, default="fixed_grid", help="Method used to recover sample order from pairs.jsonl")

    ap.add_argument("--metrics", type=str, nargs="*", default=None, help="Optional allowlist of metric keys to export")

    args = ap.parse_args()

    bpath = Path(str(args.baselines_json)).resolve()
    out_path = Path(str(args.out_jsonl)).resolve()

    rep = _load_baselines_json(bpath)
    raw_all = rep.get("raw") or {}
    if not isinstance(raw_all, dict) or not raw_all:
        raise SystemExit(f"No raw metrics found in {bpath}")

    # Infer split + max_samples from the original config when possible.
    cfg = rep.get("config") or {}
    split = str(args.split or cfg.get("split") or "test")
    max_samples = int(args.max_samples) if int(args.max_samples) >= 0 else int(cfg.get("n_samples") or 0)

    # Recover sample ids (order matters to align arrays).
    sample_ids: List[str] = []
    manifest_path = Path(str(args.manifest)).resolve() if str(args.manifest).strip() else None
    if manifest_path is not None and manifest_path.exists():
        sample_ids = _load_sample_ids_from_manifest(manifest_path, split=split, max_samples=max_samples)

    # Sanity check against array lengths; fall back to pairs.jsonl if mismatch.
    any_method = next(iter(raw_all.keys()))
    any_raw = raw_all.get(any_method) or {}
    if not isinstance(any_raw, dict) or not any_raw:
        raise SystemExit(f"Malformed raw metrics for method={any_method!r} in {bpath}")

    # Determine N from any numeric list.
    n = None
    for v in any_raw.values():
        if isinstance(v, list):
            n = len(v)
            break
    if n is None:
        raise SystemExit(f"Could not infer per-sample length from raw metrics in {bpath}")

    if len(sample_ids) != int(n):
        pairs_path = Path(str(args.pairs_jsonl)).resolve() if str(args.pairs_jsonl).strip() else None
        if pairs_path is None or not pairs_path.exists():
            raise SystemExit(
                f"Sample order mismatch: got {len(sample_ids)} sample_ids but raw arrays have n={n}. "
                f"Provide --manifest or --pairs-jsonl to recover sample order."
            )
        sample_ids = _load_sample_ids_from_pairs(pairs_path, method=str(args.pairs_method))
        if len(sample_ids) != int(n):
            raise SystemExit(
                f"Failed to recover sample order from pairs.jsonl: got {len(sample_ids)} ids but raw arrays have n={n}."
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_rows = 0

    with out_path.open("w", encoding="utf-8") as f:
        for method in sorted(raw_all.keys()):
            mr = raw_all.get(method) or {}
            if not isinstance(mr, dict):
                continue

            keys = _select_metric_keys(mr, include=args.metrics)
            # Validate keys are list-like and aligned.
            for k in keys:
                v = mr.get(k)
                if not isinstance(v, list) or len(v) != int(n):
                    raise SystemExit(f"Metric {k!r} for method={method!r} is not a list of length n={n}")

            for i, sid in enumerate(sample_ids):
                metrics: Dict[str, Any] = {}
                for k in keys:
                    v = mr.get(k)[int(i)]  # type: ignore[index]
                    if _is_num(v):
                        metrics[str(k)] = float(v)
                f.write(json.dumps({"sample_id": str(sid), "method": str(method), "metrics": metrics}, ensure_ascii=False) + "\n")
                num_rows += 1

    print(f"wrote: {out_path} (rows={num_rows}, methods={len(raw_all)}, n={n})")


if __name__ == "__main__":
    main()
