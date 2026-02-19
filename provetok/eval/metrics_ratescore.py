from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence



def _suppress_pyrush_logging() -> None:
    """Disable extremely verbose PyRuSH DEBUG logs.

    RaTEScore depends on medspacy/PyRuSH. PyRuSH uses loguru and can emit per-token
    DEBUG logs which can overwhelm stdout and drastically slow down evaluation.
    """

    try:
        from loguru import logger as _logger  # type: ignore

        _logger.disable("PyRuSH")
    except Exception:
        pass


try:  # Optional dependency
    from RaTEScore import RaTEScore as _RaTEScore  # type: ignore
except Exception:  # noqa: BLE001
    _RaTEScore = None  # type: ignore


class MissingRaTEScoreDependency(RuntimeError):
    pass


@dataclass(frozen=True)
class RaTEScoreConfig:
    """RaTEScore metric config.

    This wraps the `RaTEScore` package.

    Notes:
    - The scorer downloads transformer models on first use.
    - Default `use_gpu=False` keeps the metric runnable on CPU (slow).
    """

    bert_model: str = "Angelakeke/RaTE-NER-Deberta"
    eval_model: str = "FremyCompany/BioLORD-2023-C"
    batch_size: int = 1
    use_gpu: bool = False
    affinity_matrix: str = "long"  # long|short|path/to/json


class RaTEScoreScorer:
    def __init__(self, cfg: RaTEScoreConfig):
        if _RaTEScore is None:
            raise MissingRaTEScoreDependency(
                "Missing optional dependency for RaTEScore: RaTEScore. "
                "Install via `pip install RaTEScore` (and ensure transformers can download models)."
            )
        self.cfg = cfg
        _suppress_pyrush_logging()
        self._scorer = _RaTEScore(
            bert_model=str(cfg.bert_model),
            eval_model=str(cfg.eval_model),
            batch_size=int(max(1, int(cfg.batch_size))),
            use_gpu=bool(cfg.use_gpu),
            affinity_matrix=str(cfg.affinity_matrix),
        )

    def score(self, preds: Sequence[str], refs: Sequence[str]) -> List[float]:
        pred_list = [str(x or "") for x in preds]
        ref_list = [str(x or "") for x in refs]
        if len(pred_list) != len(ref_list):
            raise ValueError(f"preds/refs length mismatch: {len(pred_list)} vs {len(ref_list)}")
        scores = self._scorer.compute_score(pred_list, ref_list)
        return [float(x) for x in (scores or [])]
