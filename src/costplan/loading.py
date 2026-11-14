"""Read run costs straight from tracing and usage exports.

Exports come in two shapes:

* **One row per run** (a trace, a session, a task), with the run's total
  cost already added up. Load it as it is.
* **One row per model call**, with a column naming the run each call belongs
  to. Pass ``group_by`` with that column, and the calls are added up into runs.

Do not group an export that already has run totals, or the same cost is
counted twice. Some tools export both a run and its calls in one file; filter
it to one or the other first.

Column names are found automatically from common spellings (for example
``cost``, ``total_cost``, ``totalCost``, ``calculatedTotalCost``,
``prompt_tokens``, ``input_tokens``, ``usage.input``, ``trace_id``,
``traceId``). Pass the column names yourself whenever the guess is wrong.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .planning import run_cost

COST_NAMES = ["cost", "totalcost", "calculatedtotalcost", "costusd", "usd", "totalcostusd"]
INPUT_NAMES = ["inputtokens", "prompttokens", "usageinput", "usageinputtokens", "usageprompttokens"]
OUTPUT_NAMES = ["outputtokens", "completiontokens", "usageoutput", "usageoutputtokens", "usagecompletiontokens"]
CACHED_NAMES = ["cachedtokens", "cachereadinputtokens", "cachereadtokens"]
SUCCESS_NAMES = ["success", "succeeded", "passed", "pass", "issuccess", "ok"]
STATUS_NAMES = ["status"]
ERROR_NAMES = ["error"]

_TRUE = {"1", "true", "t", "yes", "y", "pass", "passed", "success", "succeeded", "ok", "completed", "done"}
_FALSE = {"0", "false", "f", "no", "n", "fail", "failed", "failure", "error", "errored", "cancelled", "timeout"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _flatten(record: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in record.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, full))
        else:
            out[full] = value
    return out


def _read_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [dict(row) for row in csv.DictReader(text.splitlines())]
    if suffix in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("data", "runs", "traces", "observations", "items", "results"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a list of records, or an object with a 'data' list")
        records = data
    else:
        raise ValueError(f"{path}: use a .csv, .json or .jsonl file")
    if not all(isinstance(r, dict) for r in records):
        raise ValueError(f"{path}: every record must be an object")
    return [_flatten(r) for r in records]


def _find(columns: Iterable[str], names: List[str]) -> Optional[str]:
    """First column whose full name, or last dotted part, matches one of the spellings."""
    columns = list(columns)
    for wanted in names:
        for col in columns:
            if _norm(col) == wanted:
                return col
    for wanted in names:
        for col in columns:
            if "." in col and _norm(col.rsplit(".", 1)[-1]) == wanted:
                return col
    return None


def _number(value: Any, what: str, where: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lstrip("$").replace(",", "")
    if text == "" or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{where}: cannot read {what} {value!r} as a number") from None


def _outcome(row: Dict[str, Any], success_col, status_col, error_col, where: str) -> Optional[bool]:
    if success_col is not None:
        cell = str(row.get(success_col, "")).strip().lower()
        if cell in _TRUE:
            return True
        if cell in _FALSE:
            return False
        if cell in {"", "none", "null"}:
            return None
        raise ValueError(f"{where}: cannot read {row.get(success_col)!r} as success or failure")
    if status_col is not None:
        cell = str(row.get(status_col, "")).strip().lower()
        if cell in _TRUE:
            return True
        if cell in _FALSE:
            return False
        return None
    if error_col is not None:
        value = row.get(error_col)
        return value is None or str(value).strip().lower() in {"", "none", "null", "false"}
    return None


@dataclass
class LoadedRuns:
    """Run costs and outcomes read from an export, plus what was used to read them."""

    costs: List[float]
    successes: Optional[List[bool]]
    run_ids: Optional[List[str]]
    rows: int
    columns_used: Dict[str, str] = field(default_factory=dict)
    rows_without_cost: int = 0

    def report(self) -> str:
        used = ", ".join(f"{k} from '{v}'" for k, v in self.columns_used.items())
        lines = [f"Read {self.rows} rows into {len(self.costs)} runs ({used})."]
        if self.rows_without_cost:
            lines.append(
                f"{self.rows_without_cost} rows had no cost and count as $0. If your tool could not price "
                "those calls, pass token columns and prices instead."
            )
        if self.successes is None:
            lines.append("No success column found, so cost per successful task is not available.")
        return "\n".join(lines)


def load_runs(
    path,
    *,
    cost: Optional[str] = None,
    input_tokens: Optional[str] = None,
    output_tokens: Optional[str] = None,
    cached_tokens: Optional[str] = None,
    success: Optional[str] = None,
    group_by: Optional[str] = None,
    input_price: Optional[float] = None,
    output_price: Optional[float] = None,
    cached_price: Optional[float] = None,
) -> LoadedRuns:
    """Read run costs from a CSV, JSON or JSON Lines export.

    Args:
        path: the export file.
        cost, input_tokens, output_tokens, cached_tokens, success: column
            names, if the automatic guess is wrong. Nested JSON fields are
            named with dots, for example ``usage.prompt_tokens``.
        group_by: column naming the run each row belongs to, for exports with
            one row per model call. Costs are added up per run, and a run
            fails if any of its rows failed.
        input_price, output_price, cached_price: dollars per million tokens,
            used when there is no cost column, or for rows with no cost.

    Without a success column, ``status`` (success or error) or ``error``
    (empty means success) are used if present.
    """
    path = Path(path)
    records = _read_records(path)
    if not records:
        raise ValueError(f"{path} has no rows")
    columns = list({k: None for r in records for k in r}.keys())

    def pick(explicit, names, label):
        if explicit is None:
            return _find(columns, names)
        if explicit not in columns:
            raise ValueError(f"{path} has no {label} column named {explicit!r}")
        return explicit

    cost_col = pick(cost, COST_NAMES, "cost")
    in_col = pick(input_tokens, INPUT_NAMES, "input token")
    out_col = pick(output_tokens, OUTPUT_NAMES, "output token")
    cached_col = pick(cached_tokens, CACHED_NAMES, "cached token")
    success_col = pick(success, SUCCESS_NAMES, "success")
    status_col = _find(columns, STATUS_NAMES) if success_col is None else None
    error_col = _find(columns, ERROR_NAMES) if success_col is None and status_col is None else None
    if group_by is not None and group_by not in columns:
        raise ValueError(f"{path} has no column named {group_by!r} to group by")
    have_prices = input_price is not None and output_price is not None
    if cost_col is None and not (in_col and out_col and have_prices):
        if in_col and out_col:
            raise ValueError(
                f"{path} has token columns but no cost column: give prices in dollars per million tokens "
                "(--input-price and --output-price, or input_price and output_price in Python)"
            )
        raise ValueError(
            f"{path}: found no cost column and no token columns. Columns are: {', '.join(columns[:15])}"
        )

    per_row: List[Tuple[Optional[str], float, Optional[bool]]] = []
    missing = priced_by_tokens = 0
    for i, row in enumerate(records, start=1):
        where = f"{path}, row {i}"
        value = _number(row.get(cost_col), "cost", where) if cost_col else None
        if value is None and in_col and out_col and have_prices:
            tokens_in = _number(row.get(in_col), "input tokens", where) or 0.0
            tokens_out = _number(row.get(out_col), "output tokens", where) or 0.0
            tokens_cached = (_number(row.get(cached_col), "cached tokens", where) or 0.0) if cached_col else 0.0
            value = run_cost(tokens_in, tokens_out, input_price, output_price, min(tokens_cached, tokens_in), cached_price)
            priced_by_tokens += 1
        if value is None:
            missing += 1
            value = 0.0
        if value < 0:
            raise ValueError(f"{where}: cost must not be negative")
        run = str(row.get(group_by)) if group_by else None
        per_row.append((run, value, _outcome(row, success_col, status_col, error_col, where)))

    if group_by:
        totals: Dict[str, float] = {}
        outcomes: Dict[str, Optional[bool]] = {}
        for run, value, ok in per_row:
            totals[run] = totals.get(run, 0.0) + value
            if ok is not None:
                outcomes[run] = (outcomes.get(run, True) and ok)
        run_ids = list(totals)
        costs = [totals[r] for r in run_ids]
        flags = [outcomes[r] for r in run_ids] if len(outcomes) == len(run_ids) else None
    else:
        run_ids = None
        costs = [value for _, value, _ in per_row]
        flags_raw = [ok for _, _, ok in per_row]
        flags = flags_raw if all(ok is not None for ok in flags_raw) else None

    used: Dict[str, str] = {}
    if cost_col:
        used["cost"] = cost_col
    if priced_by_tokens:
        used["tokens"] = f"{in_col} and {out_col}"
    for label, col in (("success", success_col), ("status", status_col), ("error", error_col)):
        if col and flags is not None:
            used[label] = col
    if group_by:
        used["runs"] = group_by
    return LoadedRuns(costs, flags, run_ids, len(records), used, missing)
