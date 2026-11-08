"""Run the evals quickstart example offline and check what it reports."""

import runpy
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "evals_quickstart.py"


def test_the_quickstart_passes_then_catches_a_regression(capsys: pytest.CaptureFixture[str]):
    found = runpy.run_path(str(EXAMPLE), run_name="example")["main"]()
    out = capsys.readouterr().out
    assert found["passed"] == found["cases"] == 3
    assert found["cost_usd"] > 0
    assert found["regressions"] == ["packed", "refund", "shipped"]
    assert "# Eval: shop-support" in out and "JUnit: 3 test cases" in out
