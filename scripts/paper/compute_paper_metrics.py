#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
for _base in (ROOT / "ProveTok", ROOT):
    if _base.exists() and str(_base) not in sys.path:
        sys.path.insert(0, str(_base))

try:
    from provetok.eval.metrics_text import MissingTextMetricDependency, TextMetricConfig, compute_text_metrics
    from provetok.eval.stats import (
        bootstrap_mean_ci,
        holm_bonferroni,
        paired_bootstrap_mean_diff,
        paired_wilcoxon_signed_rank,
    )
except Exception:
    # Fallback for mixed-layout worktrees where package root is `ProveTok/*`.
    from ProveTok.eval.metrics_text import MissingTextMetricDependency, TextMetricConfig, compute_text_metrics
    from ProveTok.eval.stats import (
        bootstrap_mean_ci,
        holm_bonferroni,
        paired_bootstrap_mean_diff,
        paired_wilcoxon_signed_rank,
    )


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


@dataclass(frozen=True)
class PairRow:
    sample_id: str
    method: str
    pred_text: str
    ref_text: str


def _load_pairs(path: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    for d in _read_jsonl(path):
        sid = str(d.get("sample_id") or d.get("scan_hash") or "").strip()
        method = str(d.get("method") or "").strip()
        pred = str(d.get("pred_text") or d.get("pred") or "")
        ref = str(d.get("ref_text") or d.get("ref") or "")
        if not sid or not method:
            continue
        rows.append(PairRow(sample_id=sid, method=method, pred_text=pred, ref_text=ref))
    if not rows:
        raise SystemExit(f"No valid rows found in {path}")
    return rows


def _load_extra_metrics(path: Optional[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if path is None:
        return {}
    extra: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in _read_jsonl(path):
        sid = str(d.get("sample_id") or d.get("scan_hash") or "").strip()
        method = str(d.get("method") or "").strip()
        if not sid or not method:
            continue
        m = d.get("metrics")
        if isinstance(m, dict):
            metrics = dict(m)
        else:
            metrics = {k: v for k, v in d.items() if k not in ("sample_id", "scan_hash", "method")}
        extra[(sid, method)] = metrics
    return extra


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def _nanfloat(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if math.isnan(v) or math.isinf(v):
        return float("nan")
    return float(v)


def _nanmean(xs: Sequence[float]) -> float:
    arr = [float(x) for x in xs if _is_num(x)]
    if not arr:
        return 0.0
    return float(sum(arr) / float(len(arr)))


def _micro_prf(pred_b: Any, ref_b: Any) -> Tuple[float, float, float]:
    import numpy as np

    pred = np.asarray(pred_b).astype(bool)
    ref = np.asarray(ref_b).astype(bool)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} ref={ref.shape}")

    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, np.logical_not(ref)).sum())
    fn = int(np.logical_and(np.logical_not(pred), ref).sum())

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _collect_per_sample_metric(per_sample: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for r in per_sample:
        out.append(_nanfloat(r.get(key)))
    return out


def _aligned_pairs(
    method_rows: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    *,
    metric_key: str,
) -> Tuple[List[float], List[float]]:
    m_map: Dict[str, float] = {}
    for r in method_rows:
        sid = str(r.get("sample_id") or "")
        if not sid:
            continue
        m_map[sid] = _nanfloat(r.get(metric_key))

    b_map: Dict[str, float] = {}
    for r in baseline_rows:
        sid = str(r.get("sample_id") or "")
        if not sid:
            continue
        b_map[sid] = _nanfloat(r.get(metric_key))

    a: List[float] = []
    b: List[float] = []
    for sid, mv in m_map.items():
        bv = b_map.get(sid)
        if bv is None:
            continue
        if not _is_num(mv) or not _is_num(bv):
            continue
        a.append(float(mv))
        b.append(float(bv))
    return a, b


def _is_positive_frame_like(frame: Any) -> bool:
    pol = str(getattr(frame, "polarity", "")).strip().lower()
    if pol not in ("present", "positive"):
        return False
    finding = str(getattr(frame, "finding", "")).strip().lower()
    if finding in ("", "normal"):
        return False
    return True


def _positive_finding_counter(frames: Sequence[Any]) -> Counter[str]:
    c: Counter[str] = Counter()
    for fr in frames:
        if not _is_positive_frame_like(fr):
            continue
        finding = str(getattr(fr, "finding", "")).strip().lower()
        if finding:
            c[finding] += 1
    return c


def _compute_finding_proxy_metrics(pred_text: str, ref_text: str, *, extractor: Any) -> Dict[str, float]:
    """Compute lightweight R1 proxies from extracted finding frames.

    This is a text-only proxy metric computed from report strings:
    - finding_precision / finding_recall / finding_f1
    - abstention_rate := N_pred_pos / N_gold_pos  (proposal-defined ratio)
    """
    pred_frames = extractor.extract_frames(str(pred_text or ""))
    ref_frames = extractor.extract_frames(str(ref_text or ""))

    pred = _positive_finding_counter(pred_frames)
    ref = _positive_finding_counter(ref_frames)

    pred_total = int(sum(pred.values()))
    ref_total = int(sum(ref.values()))
    tp = int(sum(min(int(pred[k]), int(ref[k])) for k in set(pred.keys()) | set(ref.keys())))

    precision = float(tp / pred_total) if pred_total > 0 else (1.0 if ref_total == 0 else 0.0)
    recall = float(tp / ref_total) if ref_total > 0 else 1.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    abst = float(pred_total / ref_total) if ref_total > 0 else float("nan")

    return {
        "finding_precision": float(precision),
        "finding_recall": float(recall),
        "finding_f1": float(f1),
        "abstention_rate": float(abst),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute paper-grade metrics from a dumped (sample_id, method, pred_text, ref_text) JSONL.\n\n"
            "Designed to run on the output of: \n"
            "  python -m provetok.experiments.run_baselines --dump-text-pairs-jsonl <pairs.jsonl> ...\n\n"
            "Supports optional heavy metrics (CheXbert, RadGraph, RaTEScore) and merges\n"
            "external metrics (e.g., GREEN/RadCliQ) via --extra-metrics-jsonl.\n"
            "Also reports lightweight finding-level proxies (finding_precision/recall/f1, abstention_rate)."
        )
    )
    ap.add_argument("--text-pairs-jsonl", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="Output JSON path")
    ap.add_argument(
        "--extra-metrics-jsonl",
        type=str,
        default="",
        help=(
            "Optional JSONL with extra per-sample metrics to merge. Each line should contain: "
            "{sample_id, method, metrics:{...}} or {sample_id, method, <metric_key>:<val>, ...}. "
            "Use this for metrics not implemented in-repo (e.g., GREEN/RadCliQ)."
        ),
    )

    # Stats
    ap.add_argument("--baseline-method", type=str, default="", help="Optional baseline method for paired comparisons")
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument(
        "--holm-family",
        type=str,
        default="all",
        choices=["all", "per_metric"],
        help="How to define the Holm correction family over paired comparisons.",
    )

    # CheXbert
    ap.add_argument("--chexbert-model", type=str, default="", help="HF path/id for a CheXbert-style labeler")
    ap.add_argument("--chexbert-device", type=str, default="cpu")
    ap.add_argument("--chexbert-output-mode", type=str, default="auto", choices=["auto", "sigmoid", "chexbert4"])
    ap.add_argument("--chexbert-uncertain-positive", action="store_true")

    # RadGraph
    ap.add_argument(
        "--radgraph-model-type",
        type=str,
        default="",
        help="Enable RadGraph by selecting a model type (e.g., modern-radgraph-xl)",
    )
    ap.add_argument("--radgraph-cuda", type=int, default=None, help="RadGraph device: -1=cpu, 0=cuda:0, ... (default: auto)")
    ap.add_argument("--radgraph-batch-size", type=int, default=1)

    # RaTEScore
    ap.add_argument("--ratescore", action="store_true", help="Enable RaTEScore (downloads HF models; slow)")
    ap.add_argument("--ratescore-use-gpu", action="store_true")
    ap.add_argument("--ratescore-batch-size", type=int, default=1)
    ap.add_argument("--ratescore-affinity-matrix", type=str, default="long", choices=["long", "short"], help="RaTEScore affinity matrix")

    args = ap.parse_args()

    in_path = Path(str(args.text_pairs_jsonl)).resolve()
    out_path = Path(str(args.out)).resolve()
    extra_path = Path(str(args.extra_metrics_jsonl)).resolve() if str(args.extra_metrics_jsonl).strip() else None

    rows = _load_pairs(in_path)
    extra = _load_extra_metrics(extra_path)

    # Optional scorers
    chexbert_labeler = None
    if str(args.chexbert_model).strip():
        try:
            from provetok.eval.metrics_chexbert import CheXbertConfig, CheXbertLabeler

            chexbert_labeler = CheXbertLabeler(
                CheXbertConfig(
                    model_path=str(args.chexbert_model),
                    device=str(args.chexbert_device),
                    output_mode=str(args.chexbert_output_mode),
                    treat_uncertain_as_positive=bool(args.chexbert_uncertain_positive),
                )
            )
        except Exception as e:
            print(f"[warn] chexbert disabled: {type(e).__name__}: {e}")
            chexbert_labeler = None

    radgraph_scorer = None
    if str(args.radgraph_model_type).strip():
        try:
            from provetok.eval.metrics_radgraph import RadGraphConfig, RadGraphScorer

            radgraph_scorer = RadGraphScorer(
                RadGraphConfig(
                    model_type=str(args.radgraph_model_type),
                    cuda=args.radgraph_cuda,
                    batch_size=int(args.radgraph_batch_size),
                )
            )
        except Exception as e:
            print(f"[warn] radgraph disabled: {type(e).__name__}: {e}")
            radgraph_scorer = None

    ratescore_scorer = None
    if bool(args.ratescore):
        try:
            from provetok.eval.metrics_ratescore import RaTEScoreConfig, RaTEScoreScorer

            ratescore_scorer = RaTEScoreScorer(
                RaTEScoreConfig(
                    batch_size=int(args.ratescore_batch_size),
                    use_gpu=bool(args.ratescore_use_gpu),
                    affinity_matrix=str(args.ratescore_affinity_matrix),
                )
            )
        except Exception as e:
            print(f"[warn] ratescore disabled: {type(e).__name__}: {e}")
            ratescore_scorer = None

    finding_extractor = None
    finding_proxy_warned = False
    try:
        try:
            from provetok.data.frame_extractor import FrameExtractor
        except Exception:
            from ProveTok.data.frame_extractor import FrameExtractor

        finding_extractor = FrameExtractor()
    except Exception as e:
        print(f"[warn] finding proxy metrics disabled: {type(e).__name__}: {e}")
        finding_extractor = None

    # Group by method
    by_method: Dict[str, List[PairRow]] = {}
    for r in rows:
        by_method.setdefault(r.method, []).append(r)

    out: Dict[str, Any] = {
        "input": {
            "text_pairs_jsonl": str(in_path),
            "num_rows": int(len(rows)),
            "num_methods": int(len(by_method)),
            "extra_metrics_jsonl": str(extra_path) if extra_path is not None else "",
            "baseline_method": str(args.baseline_method or ""),
            "n_bootstrap": int(args.n_bootstrap),
            "ci": float(args.ci),
            "holm_family": str(args.holm_family),
        },
        "methods": {},
        "comparisons": {},
    }

    # Compute per method
    for method, mr in sorted(by_method.items(), key=lambda kv: kv[0]):
        preds = [x.pred_text for x in mr]
        refs = [x.ref_text for x in mr]

        per_sample: List[Dict[str, Any]] = []
        text_enabled = True

        for r in mr:
            row_out: Dict[str, Any] = {
                "sample_id": r.sample_id,
                "method": r.method,
            }

            try:
                tm = compute_text_metrics(r.pred_text, r.ref_text, cfg=TextMetricConfig(compute_meteor=True))
            except MissingTextMetricDependency as e:
                print(f"[warn] text metrics disabled: {e}")
                tm = {}
                text_enabled = False

            for k, v in tm.items():
                if _is_num(v):
                    row_out[str(k)] = float(v)

            if finding_extractor is not None:
                try:
                    fpm = _compute_finding_proxy_metrics(r.pred_text, r.ref_text, extractor=finding_extractor)
                    row_out.update({k: float(v) for k, v in fpm.items()})
                except Exception as e:
                    if not finding_proxy_warned:
                        print(f"[warn] finding proxy metrics failed (continuing): {type(e).__name__}: {e}")
                        finding_proxy_warned = True

            # Merge external per-sample metrics
            ext = extra.get((r.sample_id, r.method))
            if isinstance(ext, dict):
                for k, v in ext.items():
                    if k in row_out:
                        continue
                    row_out[str(k)] = v

            per_sample.append(row_out)

        # CheXbert (dataset-level micro PRF)
        chexbert_summary: Dict[str, float] = {}
        if chexbert_labeler is not None:
            try:
                pred_bin = chexbert_labeler.predict_binary(preds)
                ref_bin = chexbert_labeler.predict_binary(refs)
                p, r, f1 = _micro_prf(pred_bin, ref_bin)
                chexbert_summary = {
                    "chexbert_precision": float(p),
                    "chexbert_recall": float(r),
                    "chexbert_f1": float(f1),
                }
            except Exception as e:
                print(f"[warn] chexbert failed for method={method}: {type(e).__name__}: {e}")

        # RadGraph (batch)
        if radgraph_scorer is not None:
            try:
                from provetok.eval.metrics_radgraph import compute_radgraph_rewards

                per_rg, mean_rg = compute_radgraph_rewards(preds, refs, scorer=radgraph_scorer)
                for k, v in mean_rg.items():
                    chexbert_summary[str(k)] = float(v)
                for i in range(len(per_sample)):
                    for k, vs in per_rg.items():
                        if i < len(vs) and _is_num(vs[i]):
                            per_sample[i][str(k)] = float(vs[i])
            except Exception as e:
                print(f"[warn] radgraph failed for method={method}: {type(e).__name__}: {e}")

        # RaTEScore (batch)
        if ratescore_scorer is not None:
            try:
                scores = ratescore_scorer.score(preds, refs)
                if len(scores) == len(per_sample):
                    for i, s in enumerate(scores):
                        if _is_num(s):
                            per_sample[i]["ratescore"] = float(s)
            except Exception as e:
                print(f"[warn] ratescore failed for method={method}: {type(e).__name__}: {e}")

        # Discover numeric keys
        numeric_keys: List[str] = []
        if per_sample:
            keys = set()
            for r in per_sample:
                keys.update(str(k) for k in r.keys())
            keys.discard("sample_id")
            keys.discard("method")
            for k in sorted(keys):
                vals = _collect_per_sample_metric(per_sample, k)
                if any(_is_num(v) for v in vals):
                    numeric_keys.append(k)

        summary_mean: Dict[str, Any] = {
            "n": int(len(per_sample)),
            "text_metrics_enabled": bool(text_enabled),
        }
        summary_ci: Dict[str, Any] = {}

        for k in numeric_keys:
            vals = _collect_per_sample_metric(per_sample, k)
            mean = _nanmean(vals)
            summary_mean[str(k)] = float(mean)
            try:
                clean = [float(v) for v in vals if _is_num(v)]
                res = bootstrap_mean_ci(clean, n_boot=int(args.n_bootstrap), seed=0, ci=float(args.ci))
                summary_ci[str(k)] = {"mean": float(res.mean), "ci_low": float(res.ci_low), "ci_high": float(res.ci_high)}
            except Exception:
                summary_ci[str(k)] = {"mean": float(mean), "ci_low": float(mean), "ci_high": float(mean)}

        # Merge dataset-level summaries (CheXbert/RadGraph means)
        summary_mean.update(chexbert_summary)

        out["methods"][method] = {
            "summary": summary_mean,
            "summary_ci": summary_ci,
            "per_sample": per_sample,
        }

    # Paired comparisons to baseline
    baseline = str(args.baseline_method or "").strip()
    if baseline:
        if baseline not in out["methods"]:
            raise SystemExit(f"Unknown --baseline-method {baseline!r}. Available: {sorted(out['methods'].keys())}")

        base_rows: List[Dict[str, Any]] = list(out["methods"][baseline].get("per_sample") or [])

        # Collect all (method, metric) comparisons.
        raw_pvals: List[float] = []
        comp_keys: List[Tuple[str, str]] = []  # (method, metric)
        comps: Dict[str, Dict[str, Any]] = {}

        # Determine candidate metric keys from baseline per-sample.
        base_numeric = []
        if base_rows:
            keys = set()
            for r in base_rows:
                keys.update(str(k) for k in r.keys())
            keys.discard("sample_id")
            keys.discard("method")
            for k in sorted(keys):
                vals = _collect_per_sample_metric(base_rows, k)
                if any(_is_num(v) for v in vals):
                    base_numeric.append(k)

        for method in sorted(out["methods"].keys()):
            if method == baseline:
                continue
            method_rows: List[Dict[str, Any]] = list(out["methods"][method].get("per_sample") or [])
            method_comp: Dict[str, Any] = {}

            for k in base_numeric:
                a, b = _aligned_pairs(method_rows, base_rows, metric_key=str(k))
                if len(a) < 3:
                    continue

                # Effect size: paired mean diff + bootstrap CI and p-value.
                bs = paired_bootstrap_mean_diff(a, b, n_boot=int(args.n_bootstrap), seed=0, ci=float(args.ci))

                # Wilcoxon signed-rank (paired)
                try:
                    w = paired_wilcoxon_signed_rank(a, b)
                    p_w = float(w.get("p_value", 1.0))
                    w_out = {"statistic": float(w.get("statistic", 0.0)), "p_value": p_w, "n": int(w.get("n", len(a)))}
                except Exception as e:
                    p_w = 1.0
                    w_out = {"statistic": 0.0, "p_value": 1.0, "n": int(len(a)), "error": f"{type(e).__name__}: {e}"}

                rec = {
                    "n": int(len(a)),
                    "mean_diff": float(bs.mean_diff),
                    "ci_low": float(bs.ci_low),
                    "ci_high": float(bs.ci_high),
                    "p_bootstrap": float(bs.p_value),
                    "wilcoxon": w_out,
                    "p_holm": None,
                }
                method_comp[str(k)] = rec

                raw_pvals.append(p_w)
                comp_keys.append((method, str(k)))

            comps[method] = method_comp

        # Holm correction
        if raw_pvals:
            if str(args.holm_family) == "all":
                adj = holm_bonferroni(raw_pvals)
                for (method, k), p_adj in zip(comp_keys, adj):
                    if method in comps and k in comps[method]:
                        comps[method][k]["p_holm"] = float(p_adj)
            else:
                # per_metric
                by_metric: Dict[str, List[int]] = {}
                for idx, (_m, k) in enumerate(comp_keys):
                    by_metric.setdefault(k, []).append(idx)
                for k, idxs in by_metric.items():
                    pvals = [raw_pvals[i] for i in idxs]
                    adj = holm_bonferroni(pvals)
                    for j, i in enumerate(idxs):
                        m, kk = comp_keys[i]
                        if m in comps and kk in comps[m]:
                            comps[m][kk]["p_holm"] = float(adj[j])

        out["comparisons"] = {
            "baseline_method": baseline,
            "family": str(args.holm_family),
            "methods": comps,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
