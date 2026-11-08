"""`agent-adapter eval cases-from-runs RUNS_DIR --out cases.jsonl`."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from agent_adapter.cli.main import main
from agent_adapter.evals import load_cases
from tests.evals.test_from_runs import chat_run, recorded


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    folder = recorded(tmp_path / "runs", chat_run("good"))
    recorded(folder, chat_run("bad", ok=False))
    return recorded(folder, chat_run("quiet"), capture=False)


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    return main(["eval", *argv], out), out.getvalue()


def test_writes_cases_and_lists_skipped_runs(runs: Path, tmp_path: Path) -> None:
    out = tmp_path / "cases.jsonl"
    code, text = run("cases-from-runs", str(runs), "--out", str(out))
    assert code == 0
    assert "wrote 2 case(s)" in text and "SKIPPED quiet: recorded without capture_content" in text
    assert sorted(c.id for c in load_cases(out)) == ["run-bad", "run-good"]
    code, text = run("cases-from-runs", str(runs), "--out", str(out))
    assert code == 0 and "wrote 0 case(s)" in text and "EXISTS run-good" in text


def test_filters_and_expected_from(runs: Path, tmp_path: Path) -> None:
    out = tmp_path / "failed.jsonl"
    argv = ("cases-from-runs", str(runs), "--out", str(out), "--status", "failed")
    code, _ = run(*argv, "--expected-from", "none")
    assert code == 0
    (case,) = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert case["id"] == "run-bad" and case["expected"] is None
    code, text = run("cases-from-runs", str(runs), "--out", str(out), "--tag", "nothing")
    assert code == 1 and "wrote 0" in text
    code, text = run("cases-from-runs", str(runs), "--out", str(out), "--run", "missing")
    assert code == 1 and "SKIPPED missing: no such run" in text


def test_usage_errors_exit_2(
    runs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("cases-from-runs", str(runs))[0] == 2
    assert "needs --out" in capsys.readouterr().err
    assert run("cases-from-runs", str(tmp_path / "none"), "--out", "x.jsonl")[0] == 2
    assert "no such runs folder" in capsys.readouterr().err
    assert run("app:answer", "cases.jsonl", "--out", "x.jsonl")[0] == 2
    assert "belong to `eval cases-from-runs`" in capsys.readouterr().err
