from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BootstrapMeanResult:
    mean: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class BootstrapQuantileResult:
    q: float
    value: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float


def paired_bootstrap_mean_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> PairedBootstrapResult:
    """Paired bootstrap for the mean difference (a - b).

    Returns:
        mean_diff, (ci_low, ci_high), and a two-sided p-value via bootstrap sign test.
    """
    a_arr = np.asarray(list(a), dtype=np.float64)
    b_arr = np.asarray(list(b), dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Shape mismatch: a={a_arr.shape}, b={b_arr.shape}")

    n = a_arr.shape[0]
    if n == 0:
        return PairedBootstrapResult(mean_diff=0.0, ci_low=0.0, ci_high=0.0, p_value=1.0)

    diffs = a_arr - b_arr
    mean_diff = float(diffs.mean())

    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot[i] = diffs[idx].mean()

    alpha = 1.0 - ci
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))

    # Two-sided p-value: proportion of bootstrap means crossing 0.
    p_pos = float(np.mean(boot >= 0.0))
    p_neg = float(np.mean(boot <= 0.0))
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))

    return PairedBootstrapResult(mean_diff=mean_diff, ci_low=lo, ci_high=hi, p_value=p_two)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapMeanResult:
    """Bootstrap CI for the mean of a 1D sample."""
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.shape[0])
    if n == 0:
        return BootstrapMeanResult(mean=0.0, ci_low=0.0, ci_high=0.0)

    mean = float(arr.mean())
    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot[i] = arr[idx].mean()

    alpha = 1.0 - ci
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return BootstrapMeanResult(mean=mean, ci_low=lo, ci_high=hi)


def bootstrap_quantile_ci(
    values: Sequence[float],
    *,
    q: float,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapQuantileResult:
    """Bootstrap CI for a quantile of a 1D sample.

    This is used for tail-latency (e.g., P95) in paper-grade artifacts where
    mean latency alone is insufficient.
    """
    q = float(q)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0,1], got q={q}")
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.shape[0])
    if n == 0:
        return BootstrapQuantileResult(q=q, value=0.0, ci_low=0.0, ci_high=0.0)

    value = float(np.quantile(arr, q))
    rng = np.random.RandomState(int(seed))
    boot = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.randint(0, n, size=n)
        boot[i] = float(np.quantile(arr[idx], q))

    alpha = 1.0 - float(ci)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return BootstrapQuantileResult(q=q, value=value, ci_low=lo, ci_high=hi)


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni correction (step-down) for multiple comparisons."""
    m = len(p_values)
    if m == 0:
        return []

    indexed = list(enumerate(float(p) for p in p_values))
    indexed.sort(key=lambda x: x[1])  # ascending p

    adjusted: Dict[int, float] = {}
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = (m - rank + 1) * p
        adj = min(1.0, adj)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

    return [adjusted[i] for i in range(m)]
