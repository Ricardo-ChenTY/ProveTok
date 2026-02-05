from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


try:  # Optional deps (paper-grade text metrics)
    import sacrebleu  # type: ignore
except Exception:  # noqa: BLE001
    sacrebleu = None  # type: ignore

try:  # Optional deps
    from rouge_score import rouge_scorer  # type: ignore
except Exception:  # noqa: BLE001
    rouge_scorer = None  # type: ignore

try:  # Optional deps (very slow; keep opt-in)
    from bert_score import score as bert_score  # type: ignore
except Exception:  # noqa: BLE001
    bert_score = None  # type: ignore


class MissingTextMetricDependency(RuntimeError):
    pass


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _require(dep, name: str) -> None:
    if dep is None:
        raise MissingTextMetricDependency(
            f"Missing optional dependency for text metrics: {name}. "
            "Install via `pip install -r requirements.txt`."
        )


@dataclass(frozen=True)
class TextMetricConfig:
    """Text-metric computation config.

    Notes:
    - BLEU is returned in [0,1] (sacrebleu's 0-100 scaled down).
    - ROUGE is f-measure in [0,1].
    - BERTScore is optional and returned as F1 in [0,1].
    """

    compute_bleu: bool = True
    compute_rouge: bool = True
    compute_bertscore: bool = False

    # BLEU
    bleu_tokenize: str = "13a"
    bleu_smooth_method: str = "exp"

    # ROUGE
    rouge_types: Tuple[str, ...] = ("rouge1", "rouge2", "rougeL")
    rouge_use_stemmer: bool = True

    # BERTScore (opt-in; can trigger model downloads)
    bertscore_model_type: str = "distilroberta-base"
    bertscore_lang: Optional[str] = None
    bertscore_rescale_with_baseline: bool = True
    bertscore_device: str = "cpu"


def compute_text_metrics(
    pred: str,
    ref: str,
    *,
    cfg: TextMetricConfig = TextMetricConfig(),
) -> Dict[str, float]:
    """Compute per-sample text metrics for (pred, ref)."""
    pred_n = _normalize(pred)
    ref_n = _normalize(ref)

    out: Dict[str, float] = {}

    if cfg.compute_bleu:
        _require(sacrebleu, "sacrebleu")
        if not ref_n and not pred_n:
            out["bleu"] = 1.0
        elif not ref_n or not pred_n:
            out["bleu"] = 0.0
        else:
            bleu = sacrebleu.sentence_bleu(  # type: ignore[union-attr]
                pred_n,
                [ref_n],
                smooth_method=str(cfg.bleu_smooth_method),
                tokenize=str(cfg.bleu_tokenize),
            )
            out["bleu"] = float(bleu.score) / 100.0

    if cfg.compute_rouge:
        _require(rouge_scorer, "rouge-score")
        scorer = rouge_scorer.RougeScorer(list(cfg.rouge_types), use_stemmer=bool(cfg.rouge_use_stemmer))  # type: ignore[union-attr]
        if not ref_n and not pred_n:
            for k in cfg.rouge_types:
                out[k] = 1.0
        elif not ref_n or not pred_n:
            for k in cfg.rouge_types:
                out[k] = 0.0
        else:
            scores = scorer.score(ref_n, pred_n)
            for k in cfg.rouge_types:
                s = scores.get(k)
                out[k] = float(getattr(s, "fmeasure", 0.0)) if s is not None else 0.0

    return out


def compute_bertscore_f1_batch(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    model_type: str = "distilroberta-base",
    lang: Optional[str] = None,
    rescale_with_baseline: bool = True,
    device: str = "cpu",
) -> List[float]:
    """Compute BERTScore F1 per sample (batch).

    Warning: this may download large models on first use.
    """
    _require(bert_score, "bert-score")
    preds_n = [_normalize(p) for p in preds]
    refs_n = [_normalize(r) for r in refs]
    if len(preds_n) != len(refs_n):
        raise ValueError(f"preds/refs length mismatch: {len(preds_n)} vs {len(refs_n)}")
    if not preds_n:
        return []

    P, R, F1 = bert_score(  # type: ignore[misc]
        preds_n,
        refs_n,
        model_type=str(model_type),
        lang=str(lang) if lang is not None else None,
        rescale_with_baseline=bool(rescale_with_baseline),
        device=str(device),
        verbose=False,
    )
    return [float(x) for x in F1.detach().cpu().tolist()]


def compute_text_metrics_batch(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    cfg: TextMetricConfig = TextMetricConfig(),
) -> Dict[str, List[float]]:
    """Compute per-sample metrics for a batch.

    Returns:
        metric_name -> list of per-sample floats (same length as preds).
    """
    if len(preds) != len(refs):
        raise ValueError(f"preds/refs length mismatch: {len(preds)} vs {len(refs)}")

    out: Dict[str, List[float]] = {}
    if not preds:
        return out

    # BLEU/ROUGE per sample
    cfg_no_bert = dict(cfg.__dict__)
    cfg_no_bert["compute_bertscore"] = False
    cfg_no_bert_obj = TextMetricConfig(**cfg_no_bert)
    for p, r in zip(preds, refs):
        m = compute_text_metrics(p, r, cfg=cfg_no_bert_obj)
        for k, v in m.items():
            out.setdefault(str(k), []).append(float(v))

    # Optional BERTScore batch
    if cfg.compute_bertscore:
        f1 = compute_bertscore_f1_batch(
            preds,
            refs,
            model_type=str(cfg.bertscore_model_type),
            lang=cfg.bertscore_lang,
            rescale_with_baseline=bool(cfg.bertscore_rescale_with_baseline),
            device=str(cfg.bertscore_device),
        )
        out["bertscore_f1"] = [float(x) for x in f1]

    return out
