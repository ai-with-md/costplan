"""Small statistics helpers built on the standard library only."""

from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Callable, List, Optional, Sequence, Tuple

_NORMAL = NormalDist()


def check_level(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")


def z_two_sided(confidence: float) -> float:
    return _NORMAL.inv_cdf(1 - (1 - confidence) / 2)


def wilson(successes: int, n: int, confidence: float) -> Tuple[float, float]:
    z = z_two_sided(confidence)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def binom_cdf_table(n: int, p: float) -> List[float]:
    """[P(X <= k) for k = 0..n] for X ~ Binomial(n, p), computed in log space."""
    if p <= 0:
        return [1.0] * (n + 1)
    if p >= 1:
        return [0.0] * n + [1.0]
    log_p, log_q = math.log(p), math.log1p(-p)
    log_n = math.lgamma(n + 1)
    table, total = [], 0.0
    for i in range(n + 1):
        total += math.exp(log_n - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q)
        table.append(min(1.0, total))
    return table


def quantile(sorted_values: Sequence[float], p: float) -> float:
    """Linear interpolation quantile (the usual default in numpy and pandas)."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = p * (n - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def quantile_interval(
    sorted_values: Sequence[float], p: float, confidence: float
) -> Tuple[Optional[float], Optional[float]]:
    """Distribution-free interval for the p-th quantile from order statistics.

    Makes no assumption about the shape of the cost distribution. Either end
    is None when there are too few runs to bound it at this confidence.
    """
    n = len(sorted_values)
    cdf = binom_cdf_table(n, p)
    tail = (1 - confidence) / 2
    low = None
    for k in range(n, 0, -1):  # largest k with P(X_(k) > q) at most the tail
        if cdf[k - 1] <= tail:
            low = sorted_values[k - 1]
            break
    high = upper_quantile_bound(sorted_values, p, 1 - tail, cdf)
    return low, high


def upper_quantile_bound(
    sorted_values: Sequence[float], p: float, confidence: float, cdf: Optional[List[float]] = None
) -> Optional[float]:
    """Smallest order statistic that is above the p-th quantile with the given confidence."""
    n = len(sorted_values)
    cdf = cdf or binom_cdf_table(n, p)
    for k in range(1, n + 1):
        if cdf[k - 1] >= confidence:
            return sorted_values[k - 1]
    return None


def bootstrap_interval(
    columns: Sequence[Sequence],
    statistic: Callable[..., Optional[float]],
    confidence: float,
    resamples: int,
    seed: int,
) -> Tuple[float, float]:
    """Percentile bootstrap, resampling rows of the given columns together."""
    rng = random.Random(seed)
    n = len(columns[0])
    values: List[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        value = statistic(*[[col[i] for i in idx] for col in columns])
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    values.sort()
    tail = (1 - confidence) / 2
    return quantile(values, tail), quantile(values, 1 - tail)
