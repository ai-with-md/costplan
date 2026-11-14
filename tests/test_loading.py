"""Reading exports. The files here are shaped like real exports, using field
names from the tools' documentation: LangSmith runs (total_cost as text,
status and error), Langfuse observations (traceId, calculatedTotalCost,
nested usage), and provider usage logs (usage.prompt_tokens)."""

import json

import pytest

import costplan as cp
from costplan.cli import main


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def test_langsmith_shaped_runs(tmp_path):
    runs = [
        {"id": "r1", "trace_id": "r1", "total_cost": "0.0123", "prompt_tokens": 3000, "completion_tokens": 200, "status": "success", "error": None},
        {"id": "r2", "trace_id": "r2", "total_cost": "0.0410", "prompt_tokens": 9000, "completion_tokens": 900, "status": "error", "error": "Timeout"},
        {"id": "r3", "trace_id": "r3", "total_cost": "0.0080", "prompt_tokens": 2000, "completion_tokens": 100, "status": "success", "error": None},
    ]
    loaded = cp.load_runs(write(tmp_path, "runs.json", runs))
    assert loaded.costs == pytest.approx([0.0123, 0.041, 0.008])
    assert loaded.successes == [True, False, True]
    assert loaded.columns_used["cost"] == "total_cost"
    assert loaded.columns_used["status"] == "status"


def test_langfuse_shaped_observations_grouped_into_traces(tmp_path):
    data = {
        "data": [
            {"id": "o1", "traceId": "t1", "type": "GENERATION", "calculatedTotalCost": "0.002", "usage": {"input": 1000, "output": 50}},
            {"id": "o2", "traceId": "t1", "type": "GENERATION", "calculatedTotalCost": "0.003", "usage": {"input": 1500, "output": 80}},
            {"id": "o3", "traceId": "t2", "type": "GENERATION", "calculatedTotalCost": "0.010", "usage": {"input": 5000, "output": 300}},
        ]
    }
    loaded = cp.load_runs(write(tmp_path, "observations.json", data), group_by="traceId")
    assert loaded.run_ids == ["t1", "t2"]
    assert loaded.costs == pytest.approx([0.005, 0.010])
    assert loaded.successes is None
    assert "Read 3 rows into 2 runs" in loaded.report()


def test_unpriced_calls_are_priced_from_tokens(tmp_path):
    data = [
        {"traceId": "t1", "calculatedTotalCost": None, "usage": {"input": 1_000_000, "output": 0}},
        {"traceId": "t1", "calculatedTotalCost": "0.5", "usage": {"input": 10, "output": 10}},
    ]
    path = write(tmp_path, "obs.json", data)
    without_prices = cp.load_runs(path, group_by="traceId")
    assert without_prices.costs == pytest.approx([0.5])
    assert without_prices.rows_without_cost == 1
    with_prices = cp.load_runs(path, group_by="traceId", input_price=3.0, output_price=15.0)
    assert with_prices.costs == pytest.approx([3.5])
    assert with_prices.rows_without_cost == 0


def test_provider_usage_log_with_cached_tokens(tmp_path):
    lines = [
        {"run_id": "a", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "prompt_tokens_details": {"cached_tokens": 500_000}}, "passed": True},
        {"run_id": "a", "usage": {"prompt_tokens": 0, "completion_tokens": 100_000, "prompt_tokens_details": {"cached_tokens": 0}}, "passed": True},
        {"run_id": "b", "usage": {"prompt_tokens": 200_000, "completion_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0}}, "passed": False},
    ]
    path = write(tmp_path, "usage.jsonl", "\n".join(json.dumps(x) for x in lines))
    loaded = cp.load_runs(path, group_by="run_id", input_price=3.0, output_price=15.0, cached_price=0.3)
    # Run a: 500k fresh at $3 + 500k cached at $0.30 + 100k output at $15.
    assert loaded.costs == pytest.approx([1.5 + 0.15 + 1.5, 0.6])
    assert loaded.successes == [True, False]


def test_a_run_fails_if_any_of_its_calls_failed(tmp_path):
    csv_text = "trace,cost,status\nt1,0.1,success\nt1,0.2,error\nt2,0.3,success\n"
    loaded = cp.load_runs(write(tmp_path, "calls.csv", csv_text), group_by="trace")
    assert loaded.successes == [False, True]


def test_csv_with_dollar_signs_and_spaced_headers(tmp_path):
    loaded = cp.load_runs(write(tmp_path, "runs.csv", "Run,Total Cost,Success\n1,$0.12,yes\n2,\"$1,204.50\",no\n"))
    assert loaded.costs == pytest.approx([0.12, 1204.5])
    assert loaded.successes == [True, False]


def test_error_column_means_failure_when_filled(tmp_path):
    loaded = cp.load_runs(write(tmp_path, "runs.csv", "cost,error\n0.1,\n0.2,rate limited\n"))
    assert loaded.successes == [True, False]


def test_explicit_column_names(tmp_path):
    path = write(tmp_path, "runs.csv", "spend,good\n0.1,1\n0.2,0\n")
    loaded = cp.load_runs(path, cost="spend", success="good")
    assert loaded.costs == pytest.approx([0.1, 0.2])
    assert loaded.successes == [True, False]


@pytest.mark.parametrize(
    "name, content, kwargs, message",
    [
        ("runs.csv", "a,b\n1,2\n", {}, "no cost column"),
        ("runs.csv", "input_tokens,output_tokens\n10,10\n", {}, "--input-price"),
        ("runs.csv", "cost\n0.1\n", {"cost": "price"}, "no cost column named"),
        ("runs.csv", "cost\n0.1\n", {"group_by": "trace"}, "to group by"),
        ("runs.csv", "cost\nabc\n", {}, "cannot read cost"),
        ("runs.txt", "cost\n0.1\n", {}, ".csv, .json or .jsonl"),
    ],
)
def test_clear_errors(tmp_path, name, content, kwargs, message):
    with pytest.raises(ValueError, match=message):
        cp.load_runs(write(tmp_path, name, content), **kwargs)


def test_cli_group_by_and_verbose(tmp_path, capsys):
    path = write(tmp_path, "calls.csv", "trace_id,cost,status\nt1,0.1,success\nt1,0.2,success\nt2,0.4,error\nt3,0.3,success\n")
    assert main(["summary", str(path), "--group-by", "trace_id", "-v"]) == 0
    captured = capsys.readouterr()
    assert "Read 4 rows into 3 runs" in captured.err
    assert "3 runs." in captured.out


def test_cli_paired_lines_up_tasks_by_id(tmp_path, capsys):
    a = write(tmp_path, "a.csv", "task_id,cost,success\n1,0.10,1\n2,0.50,1\n3,0.20,0\n4,0.15,1\n")
    b = write(tmp_path, "b.csv", "task_id,cost,success\n4,0.09,1\n3,0.12,1\n2,0.30,0\n1,0.06,1\n")
    assert main(["compare", str(a), str(b), "--paired", "--task-column", "task_id"]) == 0
    assert "compared task by task" in capsys.readouterr().out


def test_cli_paired_needs_equal_counts(tmp_path, capsys):
    a = write(tmp_path, "a.csv", "cost,success\n0.1,1\n0.2,1\n0.3,0\n")
    b = write(tmp_path, "b.csv", "cost,success\n0.1,1\n0.2,0\n")
    assert main(["compare", str(a), str(b), "--paired"]) == 2
    assert "same number of runs" in capsys.readouterr().err
