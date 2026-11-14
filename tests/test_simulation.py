"""Check costplan's promises against simulated agent runs.

Run costs are drawn from a lognormal distribution, which is lopsided like real
agent costs, so the true quantiles and averages are known exactly. Seeds are
fixed so the tests are repeatable.
"""

import math
import random
from statistics import NormalDist

import pytest

import costplan as cp

MU, SIGMA = math.log(0.05), 0.8  # typical run 5 cents, long right tail
TRUE_MEAN = math.exp(MU + SIGMA ** 2 / 2)
SIMS = 400


def true_quantile(p):
    return math.exp(MU + SIGMA * NormalDist().inv_cdf(p))


def true_exceed_rate(cap):
    return 1 - NormalDist().cdf((math.log(cap) - MU) / SIGMA)


def draw(rng, n):
    return [rng.lognormvariate(MU, SIGMA) for _ in range(n)]


def test_median_interval_covers_the_truth():
    rng = random.Random(1)
    truth, hits = true_quantile(0.5), 0
    for _ in range(SIMS):
        s = cp.summarize(draw(rng, 60), resamples=50)
        lo, hi = s.median_interval
        hits += lo <= truth <= hi
    assert hits / SIMS >= 0.93


def test_recommended_cap_keeps_its_promise():
    rng = random.Random(2)
    kept = 0
    for _ in range(SIMS):
        advice = cp.recommend_cap(draw(rng, 300), stop_rate=0.02, confidence=0.95)
        kept += true_exceed_rate(advice.safe_cap) <= 0.02
    assert kept / SIMS >= 0.93


def test_cap_at_the_max_of_the_needed_runs_keeps_its_promise():
    rng = random.Random(3)
    n = cp.runs_needed_for_cap(0.02, 0.95)
    kept = sum(true_exceed_rate(max(draw(rng, n))) <= 0.02 for _ in range(SIMS))
    assert kept / SIMS >= 0.93


def test_guaranteed_stop_rate_holds_for_a_small_pilot():
    rng = random.Random(4)
    promise = cp.guaranteed_stop_rate(30, 0.95)
    kept = sum(true_exceed_rate(max(draw(rng, 30))) <= promise for _ in range(SIMS))
    assert kept / SIMS >= 0.93


def test_average_cost_interval_covers_the_truth():
    rng = random.Random(5)
    hits = 0
    for _ in range(SIMS):
        lo, hi = cp.summarize(draw(rng, 80), resamples=400).mean_interval
        hits += lo <= TRUE_MEAN <= hi
    # The percentile bootstrap runs a little short on lopsided costs.
    assert hits / SIMS >= 0.90


def test_cost_per_success_interval_covers_the_truth():
    rng = random.Random(6)
    success_rate, hits = 0.8, 0
    truth = TRUE_MEAN / success_rate
    for _ in range(SIMS):
        costs = draw(rng, 100)
        wins = [rng.random() < success_rate for _ in costs]
        lo, hi = cp.summarize(costs, wins, resamples=400).cost_per_success_interval
        hits += lo <= truth <= hi
    assert hits / SIMS >= 0.90


def test_pilot_size_pins_the_average():
    rng = random.Random(7)
    spread = math.sqrt(math.exp(SIGMA ** 2) - 1)
    n = cp.pilot_runs_for_average(spread, margin=0.10)
    close = sum(abs(sum(draw(rng, n)) / n / TRUE_MEAN - 1) <= 0.10 for _ in range(SIMS))
    assert close / SIMS >= 0.92


def test_spend_projection_covers_the_real_bill():
    rng = random.Random(8)
    hits = 0
    for _ in range(SIMS):
        projection = cp.project_spend(draw(rng, 100), runs=2000)
        bill = sum(draw(rng, 2000))
        hits += projection.low <= bill <= projection.high
    assert hits / SIMS >= 0.90


def test_cheaper_model_is_found_cheaper_per_success():
    rng = random.Random(9)
    found = 0
    for _ in range(100):
        a = draw(rng, 80)
        a_ok = [rng.random() < 0.85 for _ in a]
        b = [c * 0.3 for c in draw(rng, 80)]  # 70% cheaper per run
        b_ok = [rng.random() < 0.6 for _ in b]  # but fails more often
        result = cp.compare_cost_per_success(a, a_ok, b, b_ok, resamples=300)
        found += result.ratio_interval[1] < 1
    assert found >= 80


# Paired comparison: both setups run the same tasks, and hard tasks cost more for both.


def paired_pilot(rng, n, ratio=0.6, p_a=0.9, p_b=0.75, noise=0.25):
    def wobble():
        return rng.lognormvariate(-noise ** 2 / 2, noise)  # average 1

    hardness = [rng.lognormvariate(-0.9 ** 2 / 2, 0.9) for _ in range(n)]
    a = [0.10 * h * wobble() for h in hardness]
    b = [0.10 * ratio * h * wobble() for h in hardness]
    a_ok = [rng.random() < p_a for _ in range(n)]
    b_ok = [rng.random() < p_b for _ in range(n)]
    return a, a_ok, b, b_ok


def test_paired_interval_covers_the_true_ratio():
    rng = random.Random(10)
    truth = 0.6 * 0.9 / 0.75
    hits = 0
    for _ in range(300):
        a, a_ok, b, b_ok = paired_pilot(rng, 80)
        lo, hi = cp.compare_cost_per_success(a, a_ok, b, b_ok, paired=True, resamples=400).ratio_interval
        hits += lo <= truth <= hi
    assert hits / 300 >= 0.90


def test_pairing_gives_much_tighter_intervals():
    rng = random.Random(11)
    paired_width = separate_width = 0.0
    for _ in range(60):
        a, a_ok, b, b_ok = paired_pilot(rng, 60)
        lo, hi = cp.compare_cost_per_success(a, a_ok, b, b_ok, paired=True, resamples=400).ratio_interval
        paired_width += hi - lo
        lo, hi = cp.compare_cost_per_success(a, a_ok, b, b_ok, paired=False, resamples=400).ratio_interval
        separate_width += hi - lo
    assert paired_width < 0.7 * separate_width
