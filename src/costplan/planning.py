"""Plan what an agent costs from a handful of pilot runs.

Agent costs are lopsided: most runs are cheap, and a few loop, retry or read
long documents and cost many times more. That makes averages slow to settle
and makes "what cap should I set?" a question about the tail, not the middle.
The methods here make no assumption about the shape of the costs: quantiles
use order statistics, and intervals for averages and ratios use the bootstrap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import Iterable, List, Optional, Sequence, Tuple

from ._stats import (
    bootstrap_interval,
    check_level,
    quantile,
    quantile_interval,
    upper_quantile_bound,
    wilson,
    z_two_sided,
)


def run_cost(
    input_tokens: float,
    output_tokens: float,
    input_price: float,
    output_price: float,
    cached_tokens: float = 0,
    cached_price: Optional[float] = None,
) -> float:
    """Dollar cost of one call or one run from token counts.

    Prices are in dollars per million tokens, as providers list them.
    ``cached_tokens`` are the part of ``input_tokens`` served from the
    provider's prompt cache, billed at ``cached_price`` (defaults to the input
    price). costplan ships no price table, because prices change; take them
    from your provider's pricing page.

    >>> round(run_cost(12_000, 800, input_price=3.0, output_price=15.0), 4)
    0.048
    """
    for name, value in [("input_tokens", input_tokens), ("output_tokens", output_tokens), ("cached_tokens", cached_tokens)]:
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    if cached_tokens > input_tokens:
        raise ValueError("cached_tokens are part of input_tokens, so cannot exceed them")
    if cached_price is None:
        cached_price = input_price
    fresh = input_tokens - cached_tokens
    return (fresh * input_price + cached_tokens * cached_price + output_tokens * output_price) / 1_000_000


def _clean(costs: Iterable[float], successes: Optional[Iterable]) -> Tuple[List[float], Optional[List[bool]]]:
    values = [float(c) for c in costs]
    if not values:
        raise ValueError("need at least one run")
    if any(not math.isfinite(c) or c < 0 for c in values):
        raise ValueError("costs must be finite and not negative")
    flags = None
    if successes is not None:
        flags = [bool(s) for s in successes]
        if len(flags) != len(values):
            raise ValueError("successes needs one value per run, in the same order as costs")
    return values, flags


def _cost_per_success(costs: Sequence[float], flags: Sequence[bool]) -> Optional[float]:
    wins = sum(flags)
    return sum(costs) / wins if wins else None


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


@dataclass(frozen=True)
class CostSummary:
    n: int
    mean: float
    mean_interval: Tuple[float, float]
    median: float
    median_interval: Tuple[Optional[float], Optional[float]]
    p95: float
    p95_interval: Tuple[Optional[float], Optional[float]]
    max: float
    spread: float
    success_rate: Optional[float]
    success_interval: Optional[Tuple[float, float]]
    cost_per_success: Optional[float]
    cost_per_success_interval: Optional[Tuple[float, float]]
    confidence: float

    def summary(self) -> str:
        c = f"{self.confidence:.0%}"

        def rng(pair):
            lo, hi = pair
            return f"{_fmt(lo)} to {_fmt(hi)}" if lo is not None and hi is not None else "needs more runs"

        lines = [
            f"{self.n} runs.",
            f"  Average cost per run: {_fmt(self.mean)} ({c} interval {rng(self.mean_interval)})",
            f"  Typical run (median): {_fmt(self.median)} ({c} interval {rng(self.median_interval)})",
            f"  1 in 20 runs costs more than: {_fmt(self.p95)} ({c} interval {rng(self.p95_interval)})",
            f"  Most expensive run: {_fmt(self.max)}",
        ]
        if self.success_rate is not None:
            lo, hi = self.success_interval
            lines.append(f"  Success rate: {self.success_rate:.1%} ({c} interval {lo:.1%} to {hi:.1%})")
            if self.cost_per_success is None:
                lines.append("  Cost per successful task: n/a (no successful runs)")
            else:
                lines.append(
                    f"  Cost per successful task: {_fmt(self.cost_per_success)} "
                    f"({c} interval {rng(self.cost_per_success_interval)})"
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def summarize(
    costs: Iterable[float],
    successes: Optional[Iterable] = None,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> CostSummary:
    """What do these runs cost, and how sure can we be?

    Args:
        costs: dollar cost of each pilot run.
        successes: optional pass/fail for each run, to get the success rate
            and the cost per successful task.
        confidence: level for every interval.
        resamples, seed: bootstrap settings. The seed makes results repeatable.

    Intervals for the average and the cost per success use the percentile
    bootstrap, which is rough below about 30 runs. Quantile intervals make no
    assumption about the shape of the costs.
    """
    check_level(confidence, "confidence")
    values, flags = _clean(costs, successes)
    ordered = sorted(values)
    n = len(values)
    mean = fmean(values)
    spread = stdev(values) / mean if n > 1 and mean > 0 else 0.0
    mean_ci = bootstrap_interval([values], lambda c: fmean(c), confidence, resamples, seed) if n > 1 else (mean, mean)
    rate = rate_ci = cps = cps_ci = None
    if flags is not None:
        wins = sum(flags)
        rate = wins / n
        rate_ci = wilson(wins, n, confidence)
        cps = _cost_per_success(values, flags)
        if cps is not None and n > 1:
            cps_ci = bootstrap_interval([values, flags], _cost_per_success, confidence, resamples, seed)
    return CostSummary(
        n=n,
        mean=mean,
        mean_interval=mean_ci,
        median=quantile(ordered, 0.5),
        median_interval=quantile_interval(ordered, 0.5, confidence),
        p95=quantile(ordered, 0.95),
        p95_interval=quantile_interval(ordered, 0.95, confidence),
        max=ordered[-1],
        spread=spread,
        success_rate=rate,
        success_interval=rate_ci,
        cost_per_success=cps,
        cost_per_success_interval=cps_ci,
        confidence=confidence,
    )


# ---------------------------------------------------------------- budget caps


def runs_needed_for_cap(stop_rate: float = 0.02, confidence: float = 0.95) -> int:
    """Pilot runs needed before a cap can promise to stop at most ``stop_rate`` of normal runs.

    With this many runs, setting the cap at the most expensive pilot run
    keeps the share of normal runs that hit the cap at or below
    ``stop_rate``, with the given confidence, whatever the cost distribution.

    >>> runs_needed_for_cap(0.02, 0.95)
    149
    """
    check_level(stop_rate, "stop_rate")
    check_level(confidence, "confidence")
    return math.ceil(math.log(1 - confidence) / math.log(1 - stop_rate) - 1e-9)


def guaranteed_stop_rate(n_runs: int, confidence: float = 0.95) -> float:
    """The best promise a cap at the most expensive of ``n_runs`` pilot runs can make.

    >>> round(guaranteed_stop_rate(30), 3)
    0.095
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    check_level(confidence, "confidence")
    return 1 - (1 - confidence) ** (1 / n_runs)


def _pct_label(p: float) -> str:
    value = round(p * 100, 1)
    return f"{value:g}th"


@dataclass(frozen=True)
class CapAdvice:
    n: int
    stop_rate: float
    confidence: float
    estimate: float
    safe_cap: Optional[float]
    runs_needed: int
    best_promise: float
    max_cost: float

    def summary(self) -> str:
        head = (
            f"Goal: a cap that stops at most {self.stop_rate:.1%} of normal runs "
            f"({self.confidence:.0%} confidence), from {self.n} pilot runs."
        )
        if self.safe_cap is not None:
            return (
                f"{head}\n  Recommended cap: {_fmt(self.safe_cap)} per run.\n"
                f"  Best single guess for the {_pct_label(1 - self.stop_rate)} percentile: {_fmt(self.estimate)}. "
                "The recommended cap is set higher so the promise holds even if the pilot was lucky."
            )
        return (
            f"{head}\n  Not enough runs to promise that yet: it takes {self.runs_needed}.\n"
            f"  With {self.n} runs, a cap at the most expensive run ({_fmt(self.max_cost)}) can only promise to stop "
            f"at most {self.best_promise:.1%} of runs.\n"
            f"  Best single guess for the {_pct_label(1 - self.stop_rate)} percentile: {_fmt(self.estimate)} "
            "(a guess, not a promise)."
        )

    def __str__(self) -> str:
        return self.summary()


def recommend_cap(costs: Iterable[float], stop_rate: float = 0.02, confidence: float = 0.95) -> CapAdvice:
    """What per-run budget cap to set, so it rarely stops a normal run.

    The cap is an upper confidence bound on the ``1 - stop_rate`` quantile of
    run cost, from order statistics, so it makes no assumption about the cost
    distribution. If there are too few runs to make that promise, the advice
    says how many are needed and what the current pilot can promise instead.
    """
    check_level(stop_rate, "stop_rate")
    check_level(confidence, "confidence")
    values, _ = _clean(costs, None)
    ordered = sorted(values)
    n = len(ordered)
    return CapAdvice(
        n=n,
        stop_rate=stop_rate,
        confidence=confidence,
        estimate=quantile(ordered, 1 - stop_rate),
        safe_cap=upper_quantile_bound(ordered, 1 - stop_rate, confidence),
        runs_needed=runs_needed_for_cap(stop_rate, confidence),
        best_promise=guaranteed_stop_rate(n, confidence),
        max_cost=ordered[-1],
    )


# --------------------------------------------------------- pilot size, spend


def pilot_runs_for_average(spread: float, margin: float = 0.10, confidence: float = 0.95) -> int:
    """Pilot runs needed to pin the average cost to within ``margin`` (relative).

    ``spread`` is the coefficient of variation: standard deviation divided by
    the average. Agent costs often have a spread of 1 or more, because of the
    occasional runaway run. Get it from a small pilot with ``summarize``.

    >>> pilot_runs_for_average(spread=1.2, margin=0.10)
    554
    """
    if spread < 0:
        raise ValueError("spread must not be negative")
    check_level(margin, "margin")
    check_level(confidence, "confidence")
    z = z_two_sided(confidence)
    return max(2, math.ceil((z * spread / margin) ** 2 - 1e-9))


@dataclass(frozen=True)
class SpendProjection:
    runs: int
    expected: float
    low: float
    high: float
    confidence: float

    def summary(self) -> str:
        return (
            f"{self.runs:,} runs: about {_fmt(self.expected)} "
            f"({self.confidence:.0%} range {_fmt(self.low)} to {_fmt(self.high)})."
        )

    def __str__(self) -> str:
        return self.summary()


def project_spend(costs: Iterable[float], runs: int, confidence: float = 0.95) -> SpendProjection:
    """Total spend for ``runs`` future runs, given pilot run costs.

    The range includes both run to run variation and the uncertainty in the
    average learned from the pilot. For large volumes the second one
    dominates, so a bigger pilot narrows the range more than anything else.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    check_level(confidence, "confidence")
    values, _ = _clean(costs, None)
    n = len(values)
    if n < 2:
        raise ValueError("need at least two pilot runs to project spend")
    mean, sd = fmean(values), stdev(values)
    se_total = math.sqrt(runs * runs * sd * sd / n + runs * sd * sd)
    half = z_two_sided(confidence) * se_total
    expected = mean * runs
    return SpendProjection(runs, expected, max(0.0, expected - half), expected + half, confidence)


# --------------------------------------------------------------- comparison


@dataclass(frozen=True)
class CostComparison:
    a: str
    b: str
    a_cost_per_success: Optional[float]
    b_cost_per_success: Optional[float]
    ratio: Optional[float]
    ratio_interval: Optional[Tuple[float, float]]
    confidence: float
    paired: bool = False

    def summary(self) -> str:
        if self.ratio is None:
            return f"Cannot compare: {self.a} or {self.b} has no successful runs."
        lo, hi = self.ratio_interval
        if hi < 1:
            verdict = f"{self.b} is reliably cheaper per success."
        elif lo > 1:
            verdict = f"{self.b} is reliably more expensive per success."
        else:
            verdict = "The difference could be noise."
        return (
            f"Cost per successful task: {self.a} {_fmt(self.a_cost_per_success)}, {self.b} {_fmt(self.b_cost_per_success)}.\n"
            f"  {self.b} costs {self.ratio:.2f} times as much as {self.a} per success "
            f"({self.confidence:.0%} interval {lo:.2f} to {hi:.2f}, "
            f"{'same tasks, compared task by task' if self.paired else 'separate pilots'}). {verdict}"
        )

    def __str__(self) -> str:
        return self.summary()


def compare_cost_per_success(
    a_costs: Iterable[float],
    a_successes: Iterable,
    b_costs: Iterable[float],
    b_successes: Iterable,
    names: Tuple[str, str] = ("A", "B"),
    paired: bool = False,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> CostComparison:
    """Which setup is cheaper per successful task?

    A cheaper model that fails more often can cost more per task that gets
    done. The interval is a bootstrap of the ratio of costs per success.

    Args:
        paired: True when both setups ran the same tasks, in the same order.
            Hard tasks are expensive for both, so comparing task by task
            removes that variation and gives a much tighter interval from the
            same number of runs. False when the two pilots ran different tasks.
    """
    import random

    check_level(confidence, "confidence")
    ac, af = _clean(a_costs, a_successes)
    bc, bf = _clean(b_costs, b_successes)
    if paired and len(ac) != len(bc):
        raise ValueError("a paired comparison needs the same tasks for both setups, so the same number of runs")
    a_cps, b_cps = _cost_per_success(ac, af), _cost_per_success(bc, bf)
    if a_cps is None or b_cps is None:
        return CostComparison(names[0], names[1], a_cps, b_cps, None, None, confidence, paired)
    rng = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        ia = [rng.randrange(len(ac)) for _ in ac]
        ib = ia if paired else [rng.randrange(len(bc)) for _ in bc]
        ra = _cost_per_success([ac[i] for i in ia], [af[i] for i in ia])
        rb = _cost_per_success([bc[i] for i in ib], [bf[i] for i in ib])
        if ra and rb is not None:
            ratios.append(rb / ra)
    ratios.sort()
    tail = (1 - confidence) / 2
    interval = (quantile(ratios, tail), quantile(ratios, 1 - tail))
    return CostComparison(names[0], names[1], a_cps, b_cps, b_cps / a_cps, interval, confidence, paired)
