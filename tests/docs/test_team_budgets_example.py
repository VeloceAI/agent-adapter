"""Run the team budgets how-to example offline and check what it prints."""

import runpy
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "howto_team_budgets.py"


def test_team_budgets_share_spend_across_workers(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "ana answered",
        "ana answered",
        "bo answered",
        "bo answered",
        "ana refused by the team budget",
        "cy answered",
        "user ana 0.4 0.1",
        "team support 0.8 day",
    ]
