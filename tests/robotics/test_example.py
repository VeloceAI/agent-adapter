import runpy
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "robot_gate.py"


def test_how_to_example_prints_what_the_page_shows(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "done: move at (0.000, -0.200, 0.200)"
    assert lines[1] == "asked about move_to {'x': 0.0, 'y': -0.2, 'z': 0.2, 'speed': 0.6}"
    assert lines[2] == "denied by the reviewer"
    assert lines[3] == "blocked: path enters keep-out zone operator"
    assert lines[4].startswith("e-stop (operator pressed stop): stopped at (")
    assert lines[5] == "latched: operator pressed stop"
