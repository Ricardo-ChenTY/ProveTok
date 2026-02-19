from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


try:  # Optional dependency
    from radgraph import F1RadGraph  # type: ignore
except Exception:  # noqa: BLE001
    F1RadGraph = None  # type: ignore


class MissingRadGraphDependency(RuntimeError):
    pass


@dataclass(frozen=True)
class RadGraphConfig:
    """RadGraph metric config.

    This wraps the `radgraph` package's F1-RadGraph implementation.

    Notes:
    - `model_type` controls which model tarball is used (downloads from HF on first use).
    - We instantiate reward_level="all" to return the three common variants:
      (RG_E, RG_ER, RG_BAR_ER) ≈ (simple, partial, complete).
    """

    model_type: str = "modern-radgraph-xl"

    # Device selection follows radgraph semantics:
    # - None: auto (cuda:0 if available else cpu)
    # - -1: cpu
    # - >=0: cuda:{idx}
    cuda: Optional[int] = None

    batch_size: int = 1
    model_cache_dir: str = ""
    tokenizer_cache_dir: str = ""


class RadGraphScorer:
    def __init__(self, cfg: RadGraphConfig):
        if F1RadGraph is None:
            raise MissingRadGraphDependency(
                "Missing optional dependency for RadGraph metrics: radgraph. "
                "Install via `pip install radgraph` (and ensure network access for model download)."
            )
        self.cfg = cfg

        kwargs: Dict[str, object] = {}
        if cfg.cuda is not None:
            kwargs["cuda"] = int(cfg.cuda)
        kwargs["batch_size"] = int(max(1, int(cfg.batch_size)))
        if str(cfg.model_cache_dir).strip():
            kwargs["model_cache_dir"] = str(cfg.model_cache_dir)
        if str(cfg.tokenizer_cache_dir).strip():
            kwargs["tokenizer_cache_dir"] = str(cfg.tokenizer_cache_dir)

        # radgraph prints device info; keep experiments logs clean.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            self._scorer = F1RadGraph(reward_level="all", model_type=str(cfg.model_type), **kwargs)


def compute_radgraph_rewards(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    scorer: RadGraphScorer,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Compute RadGraph rewards for a batch.

    Returns:
        (per_sample, mean)

    Keys:
        - radgraph_e: entity-level F1 ("simple")
        - radgraph_er: entity+relation existence F1 ("partial"; commonly reported)
        - radgraph_bar_er: stricter relation match F1 ("complete")
        - radgraph_f1: alias of radgraph_er for convenience
    """
    pred_list = [str(x or "") for x in preds]
    ref_list = [str(x or "") for x in refs]
    if len(pred_list) != len(ref_list):
        raise ValueError(f"preds/refs length mismatch: {len(pred_list)} vs {len(ref_list)}")

    mean_reward, reward_list, _, _ = scorer._scorer(hyps=pred_list, refs=ref_list)
    if not (isinstance(reward_list, (list, tuple)) and len(reward_list) == 3):
        raise RuntimeError(f"Unexpected radgraph reward_list format: {type(reward_list)}")

    simple, partial, complete = reward_list
    per_sample = {
        "radgraph_e": [float(x) for x in simple],
        "radgraph_er": [float(x) for x in partial],
        "radgraph_bar_er": [float(x) for x in complete],
        "radgraph_f1": [float(x) for x in partial],
    }
    mean = {
        "radgraph_e": float(mean_reward[0]),
        "radgraph_er": float(mean_reward[1]),
        "radgraph_bar_er": float(mean_reward[2]),
        "radgraph_f1": float(mean_reward[1]),
    }
    return per_sample, mean


def compute_radgraph_f1(
    pred_text: str,
    ref_text: str,
    *,
    scorer: RadGraphScorer,
) -> Dict[str, float]:
    """Convenience wrapper for a single (pred, ref) pair."""
    per_sample, _ = compute_radgraph_rewards([pred_text], [ref_text], scorer=scorer)
    return {k: float(v[0]) for k, v in per_sample.items()}
