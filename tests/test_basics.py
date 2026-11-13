import pytest

import costplan as cp
from costplan._stats import binom_cdf_table, quantile, quantile_interval, wilson


def test_run_cost():
    assert cp.run_cost(1_000_000, 0, 3.0, 15.0) == pytest.approx(3.0)
    assert cp.run_cost(0, 1_000_000, 3.0, 15.0) == pytest.approx(15.0)
    # Half the input from the cache at a tenth of the price.
    assert cp.run_cost(1_000_000, 0, 3.0, 15.0, cached_tokens=500_000, cached_price=0.3) == pytest.approx(1.65)


@pytest.mark.parametrize(
    "kwargs",
    [dict(input_tokens=-1), dict(output_tokens=-5), dict(cached_tokens=2_000_000)],
)
def test_run_cost_rejects_bad_counts(kwargs):
    args = dict(input_tokens=1_000_000, output_tokens=0, input_price=3.0, output_price=15.0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        cp.run_cost(**args)


def test_binomial_table_matches_known_values():
    table = binom_cdf_table(10, 0.5)
    assert table[0] == pytest.approx(1 / 1024)
    assert table[5] == pytest.approx(638 / 1024)
    assert table[-1] == pytest.approx(1.0)


def test_quantile_matches_numpy_default():
    assert quantile([1, 2, 3, 4], 0.5) == 2.5
    assert quantile([1, 2, 3, 4, 5], 0.9) == pytest.approx(4.6)


def test_quantile_interval_needs_enough_runs():
    low, high = quantile_interval(sorted(range(1, 11)), 0.95, 0.95)
    assert high is None  # 10 runs cannot bound the 95th percentile from above
    low, high = quantile_interval(sorted(range(1, 101)), 0.5, 0.95)
    assert low < 50 < high


def test_wilson_known_values():
    low, high = wilson(8, 10, 0.95)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)


def test_runs_needed_and_guarantee_are_inverses():
    n = cp.runs_needed_for_cap(0.02, 0.95)
    assert n == 149
    assert cp.guaranteed_stop_rate(n, 0.95) <= 0.02
    assert cp.guaranteed_stop_rate(n - 1, 0.95) > 0.02
    assert cp.runs_needed_for_cap(0.05, 0.95) == 59


def test_pilot_runs_for_average():
    assert cp.pilot_runs_for_average(1.2, 0.10) == 554
    assert cp.pilot_runs_for_average(0.5, 0.10) < cp.pilot_runs_for_average(1.0, 0.10)


def test_summary_without_successes():
    s = cp.summarize([0.1, 0.2, 0.3, 0.4])
    assert s.n == 4
    assert s.mean == pytest.approx(0.25)
    assert s.success_rate is None
    assert "Success rate" not in s.summary()


def test_summary_with_successes():
    s = cp.summarize([0.1, 0.2, 0.3, 0.4], [True, False, True, True])
    assert s.success_rate == 0.75
    assert s.cost_per_success == pytest.approx(1.0 / 3)
    assert "Cost per successful task" in s.summary()


def test_summary_with_no_successes():
    s = cp.summarize([0.1, 0.2], [False, False])
    assert s.cost_per_success is None
    assert "no successful runs" in s.summary()


def test_cap_advice_when_pilot_is_small():
    advice = cp.recommend_cap([0.1] * 20 + [0.5] * 5)
    assert advice.safe_cap is None
    assert "it takes 149" in advice.summary()


def test_cap_advice_when_pilot_is_big_enough():
    costs = [i / 1000 for i in range(1, 501)]
    advice = cp.recommend_cap(costs, stop_rate=0.05)
    assert advice.safe_cap is not None
    assert advice.safe_cap >= advice.estimate
    assert "Recommended cap" in advice.summary()


def test_project_spend_scales_with_runs():
    costs = [0.05, 0.08, 0.1, 0.12, 0.4]
    small, big = cp.project_spend(costs, 100), cp.project_spend(costs, 10_000)
    assert big.expected == pytest.approx(small.expected * 100)
    assert big.low >= 0


def test_compare_cost_per_success():
    a = cp.compare_cost_per_success([1.0] * 50, [True] * 50, [0.5] * 50, [True] * 25 + [False] * 25, names=("big", "small"))
    assert a.ratio == pytest.approx(1.0)
    assert "small" in a.summary()
    none = cp.compare_cost_per_success([1.0], [False], [1.0], [True])
    assert none.ratio is None and "Cannot compare" in none.summary()


@pytest.mark.parametrize(
    "call",
    [
        lambda: cp.summarize([]),
        lambda: cp.summarize([0.1, -0.2]),
        lambda: cp.summarize([0.1, 0.2], [True]),
        lambda: cp.recommend_cap([0.1], stop_rate=0),
        lambda: cp.project_spend([0.1], 100),
        lambda: cp.project_spend([0.1, 0.2], 0),
        lambda: cp.guaranteed_stop_rate(0),
        lambda: cp.pilot_runs_for_average(-1),
    ],
)
def test_bad_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
