from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from agent_adapter.voice.talker import Talker

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "voxcat_open_models.py"


def test_the_offline_example_runs_a_call_with_a_talker(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "caller: where is my order",
        "talker cues while the agent thinks: 3",
        "agent: It ships today. (3600 samples at 24000 Hz)",
    ]


def test_the_stand_in_talker_fits_the_port() -> None:
    names = runpy.run_path(str(EXAMPLE))
    assert isinstance(names["NodTalker"](), Talker)
