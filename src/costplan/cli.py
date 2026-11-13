"""Command line: ``costplan summary``, ``costplan cap``, ``costplan project`` and more."""

from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Optional, Tuple

from . import (
    compare_cost_per_success,
    guaranteed_stop_rate,
    pilot_runs_for_average,
    project_spend,
    recommend_cap,
    run_cost,
    runs_needed_for_cap,
    summarize,
)

_TRUE = {"1", "true", "t", "yes", "y", "pass", "passed", "success", "ok"}
_FALSE = {"0", "false", "f", "no", "n", "fail", "failed", "failure", "error"}


def read_runs(path: str, args: argparse.Namespace) -> Tuple[List[float], Optional[List[bool]]]:
    """Read one row per run: a ``cost`` column, or token columns plus prices, and an optional ``success`` column."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no runs")
    columns = {c.strip().lower(): c for c in rows[0].keys() if c}
    costs: List[float] = []
    if args.cost_column.lower() in columns:
        key = columns[args.cost_column.lower()]
        for line, row in enumerate(rows, start=2):
            try:
                costs.append(float(row[key]))
            except (TypeError, ValueError):
                raise ValueError(f"{path}, line {line}: cannot read cost {row[key]!r}") from None
    elif "input_tokens" in columns and "output_tokens" in columns:
        if args.input_price is None or args.output_price is None:
            raise ValueError(
                f"{path} has token columns but no cost column: pass --input-price and --output-price "
                "(dollars per million tokens)"
            )
        cached = columns.get("cached_tokens")
        for row in rows:
            costs.append(
                run_cost(
                    float(row[columns["input_tokens"]]),
                    float(row[columns["output_tokens"]]),
                    args.input_price,
                    args.output_price,
                    float(row[cached]) if cached and row[cached] else 0,
                    args.cached_price,
                )
            )
    else:
        raise ValueError(f"{path} needs a '{args.cost_column}' column, or input_tokens and output_tokens columns")
    flags = None
    if args.success_column.lower() in columns:
        key = columns[args.success_column.lower()]
        flags = []
        for line, row in enumerate(rows, start=2):
            cell = (row[key] or "").strip().lower()
            if cell in _TRUE:
                flags.append(True)
            elif cell in _FALSE:
                flags.append(False)
            else:
                raise ValueError(f"{path}, line {line}: cannot read {row[key]!r} as success or failure")
    return costs, flags


def _cmd_summary(args: argparse.Namespace) -> str:
    costs, flags = read_runs(args.file, args)
    return summarize(costs, flags, args.confidence).summary()


def _cmd_cap(args: argparse.Namespace) -> str:
    costs, _ = read_runs(args.file, args)
    return recommend_cap(costs, args.stop_rate, args.confidence).summary()


def _cmd_project(args: argparse.Namespace) -> str:
    costs, _ = read_runs(args.file, args)
    return project_spend(costs, args.runs, args.confidence).summary()


def _cmd_compare(args: argparse.Namespace) -> str:
    a_costs, a_flags = read_runs(args.file_a, args)
    b_costs, b_flags = read_runs(args.file_b, args)
    if a_flags is None or b_flags is None:
        raise ValueError(f"both files need a '{args.success_column}' column to compare cost per success")
    names = tuple(args.names) if args.names else (args.file_a, args.file_b)
    return compare_cost_per_success(a_costs, a_flags, b_costs, b_flags, names, args.confidence).summary()


def _cmd_pilot(args: argparse.Namespace) -> str:
    lines = []
    n = runs_needed_for_cap(args.stop_rate, args.confidence)
    lines.append(
        f"To set a cap that stops at most {args.stop_rate:.1%} of normal runs ({args.confidence:.0%} confidence): "
        f"{n} pilot runs."
    )
    for size in (20, 50, 100):
        if size < n:
            lines.append(f"  With {size} runs, the best promise is {guaranteed_stop_rate(size, args.confidence):.1%}.")
    if args.spread is not None:
        m = pilot_runs_for_average(args.spread, args.margin, args.confidence)
        lines.append(f"To pin the average cost to within {args.margin:.0%} (spread {args.spread:g}): {m} pilot runs.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="costplan",
        description="How much will my agent cost? Plan budgets, caps and spend from pilot runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def file_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cost-column", default="cost", help="column with each run's dollar cost (default: cost)")
        p.add_argument("--success-column", default="success", help="column with pass/fail (default: success)")
        p.add_argument("--input-price", type=float, help="dollars per million input tokens, if the file has tokens not costs")
        p.add_argument("--output-price", type=float, help="dollars per million output tokens")
        p.add_argument("--cached-price", type=float, help="dollars per million cached input tokens")
        p.add_argument("--confidence", type=float, default=0.95)

    p = sub.add_parser("summary", help="average, typical and expensive runs, success rate, cost per success")
    p.add_argument("file", help="CSV with one row per pilot run")
    file_options(p)
    p.set_defaults(run=_cmd_summary)

    p = sub.add_parser("cap", help="what per-run budget cap to set")
    p.add_argument("file")
    p.add_argument("--stop-rate", type=float, default=0.02, help="largest share of normal runs the cap may stop")
    file_options(p)
    p.set_defaults(run=_cmd_cap)

    p = sub.add_parser("project", help="total spend for a number of future runs")
    p.add_argument("file")
    p.add_argument("--runs", type=int, required=True, help="number of future runs, e.g. a month of traffic")
    file_options(p)
    p.set_defaults(run=_cmd_project)

    p = sub.add_parser("compare", help="which setup is cheaper per successful task")
    p.add_argument("file_a")
    p.add_argument("file_b")
    p.add_argument("--names", nargs=2, metavar=("A", "B"))
    file_options(p)
    p.set_defaults(run=_cmd_compare)

    p = sub.add_parser("pilot", help="how many pilot runs to collect")
    p.add_argument("--stop-rate", type=float, default=0.02)
    p.add_argument("--spread", type=float, help="cost spread (standard deviation / average), from a small pilot")
    p.add_argument("--margin", type=float, default=0.10, help="relative precision wanted for the average")
    p.add_argument("--confidence", type=float, default=0.95)
    p.set_defaults(run=_cmd_pilot)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(args.run(args))
    except (ValueError, OSError) as exc:
        print(f"costplan: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
