"""Run `examples/cost_own_models.py` offline and check what it prints."""

import logging
import runpy
from pathlib import Path

import pytest

import agent_adapter.cost.prices as prices_module

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "cost_own_models.py"


def test_own_models_example_prints_its_costs_without_warnings(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    prices_module._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="agent_adapter"):
        runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "llama* 0.0",
        "0.0044",
        "0.0017",
        "gpt-4.1 *",
        "empty+mine None",
        "0.00275",
    ]
    assert caplog.text == ""
