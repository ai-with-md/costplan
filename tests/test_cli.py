from costplan.cli import main

EXAMPLE = "examples/pilot_runs.csv"


def test_every_command_runs(capsys):
    for command in [
        ["summary", EXAMPLE],
        ["cap", EXAMPLE, "--stop-rate", "0.05"],
        ["project", EXAMPLE, "--runs", "100000"],
        ["compare", EXAMPLE, "examples/cheaper_model_runs.csv", "--names", "large", "small"],
        ["pilot", "--spread", "1.0"],
    ]:
        assert main(command) == 0, command
    assert "149 pilot runs" in capsys.readouterr().out


def test_reads_token_columns_with_prices(tmp_path, capsys):
    path = tmp_path / "runs.csv"
    path.write_text("input_tokens,output_tokens,success\n1000000,0,1\n2000000,0,0\n")
    assert main(["summary", str(path), "--input-price", "3", "--output-price", "15"]) == 0
    assert "$4.50" in capsys.readouterr().out


def test_token_columns_without_prices_explain_the_fix(tmp_path, capsys):
    path = tmp_path / "runs.csv"
    path.write_text("input_tokens,output_tokens\n1000,10\n")
    assert main(["summary", str(path)]) == 2
    assert "--input-price" in capsys.readouterr().err


def test_bad_success_cell(tmp_path, capsys):
    path = tmp_path / "runs.csv"
    path.write_text("cost,success\n0.1,maybe\n")
    assert main(["summary", str(path)]) == 2
    assert "maybe" in capsys.readouterr().err
