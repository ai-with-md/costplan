"""How much will my agent cost? Plan agent budgets from a handful of pilot runs."""

from .loading import LoadedRuns, load_runs
from .planning import (
    CapAdvice,
    CostComparison,
    CostSummary,
    SpendProjection,
    compare_cost_per_success,
    guaranteed_stop_rate,
    pilot_runs_for_average,
    project_spend,
    recommend_cap,
    run_cost,
    runs_needed_for_cap,
    summarize,
)

__all__ = [
    "load_runs",
    "LoadedRuns",
    "run_cost",
    "summarize",
    "recommend_cap",
    "runs_needed_for_cap",
    "guaranteed_stop_rate",
    "pilot_runs_for_average",
    "project_spend",
    "compare_cost_per_success",
    "CostSummary",
    "CapAdvice",
    "SpendProjection",
    "CostComparison",
]

__version__ = "0.1.0"
