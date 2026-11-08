"""Run the Evals view example offline: two saved reports the viewer lists and compares."""

import runpy
from pathlib import Path

import pytest

from agent_adapter.viewer import RunFolder
from agent_adapter.viewer._evals import EvalFolder

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "evals_viewer.py"


def test_the_example_saves_reports_whose_case_runs_are_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    found = runpy.run_path(str(EXAMPLE), run_name="example")["main"](tmp_path)
    assert found["regressed"] == ["concert", "museum"]
    assert found["scores"][0] > found["scores"][1]
    assert "agent-adapter ui" in capsys.readouterr().out
    folder = EvalFolder(tmp_path / "evals")
    rows = sorted(folder.summaries(), key=lambda row: row["mean_score"])
    worse, good = (row["key"] for row in rows)
    comparison = folder.compare(good, worse)
    assert comparison is not None and comparison["regressed"]
    report = folder.report(good)
    assert report is not None
    runs = RunFolder(tmp_path / "runs")
    runs.refresh()
    recorded = {summary["run_id"] for summary in runs.summaries()}
    assert {case["run_id"] for case in report["cases"]} <= recorded, "Open run finds every case"
