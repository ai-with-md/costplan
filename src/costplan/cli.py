"""Command line: ``costplan summary``, ``costplan cap``, ``costplan project`` and more."""

from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Optional, Tuple

from . import (
    LoadedRuns,
    compare_cost_per_success,
    load_runs,
    guaranteed_stop_rate,
    pilot_runs_for_average,
    project_spend,
    recommend_cap,
    runs_needed_for_cap,
    summarize,
)

def read_runs(path: str, args: argparse.Namespace) -> Tuple[List[float], Optional[List[bool]]]:
    loaded = _load(path, args)
    return loaded.costs, loaded.successes


def _load(path: str, args: argparse.Namespace, group_by: Optional[str] = None) -> LoadedRuns:
    loaded = load_runs(
        path,
        cost=args.cost_column,
        input_tokens=args.input_column,
        output_tokens=args.output_column,
        success=args.success_column,
        group_by=group_by if group_by is not None else args.group_by,
        input_price=args.input_price,
        output_price=args.output_price,
        cached_price=args.cached_price,
    )
    if args.verbose:
        print(loaded.report(), file=sys.stderr)
    return loaded


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
    key = args.task_column
    if args.paired and key and not args.group_by:
        # Group by the task column so the two files can be lined up task by task.
        a, b = _load(args.file_a, args, key), _load(args.file_b, args, key)
    else:
        a, b = _load(args.file_a, args), _load(args.file_b, args)
    if a.successes is None or b.successes is None:
        raise ValueError("both files need a success, status or error column to compare cost per success")
    a_costs, a_flags, b_costs, b_flags = a.costs, a.successes, b.costs, b.successes
    if args.paired and a.run_ids is not None and b.run_ids is not None:
        b_index = {run: i for i, run in enumerate(b.run_ids)}
        shared = [i for i, run in enumerate(a.run_ids) if run in b_index]
        if len(shared) < 2:
            raise ValueError("the two files share fewer than two task ids")
        a_costs = [a.costs[i] for i in shared]
        a_flags = [a.successes[i] for i in shared]
        b_costs = [b.costs[b_index[a.run_ids[i]]] for i in shared]
        b_flags = [b.successes[b_index[a.run_ids[i]]] for i in shared]
    names = tuple(args.names) if args.names else (args.file_a, args.file_b)
    return compare_cost_per_success(
        a_costs, a_flags, b_costs, b_flags, names, paired=args.paired, confidence=args.confidence
    ).summary()


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
        p.add_argument("--cost-column", help="column with each row's dollar cost (found automatically if left out)")
        p.add_argument("--input-column", help="column with input tokens (found automatically if left out)")
        p.add_argument("--output-column", help="column with output tokens (found automatically if left out)")
        p.add_argument("--success-column", help="column with pass/fail (found automatically if left out)")
        p.add_argument("--group-by", help="column naming the run each row belongs to, for exports with one row per model call")
        p.add_argument("-v", "--verbose", action="store_true", help="say which columns were used")
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
    p.add_argument("--paired", action="store_true", help="both files ran the same tasks; compare task by task")
    p.add_argument("--task-column", help="column with a task id, to line up the two files when --paired")
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
