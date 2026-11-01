"""The AG-UI bridge example runs offline and prints what the docs show."""

import runpy
from pathlib import Path

import pytest

pytest.importorskip("ag_ui")

pytestmark = pytest.mark.contract

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


def test_bridge_example_pauses_then_resumes(capsys):
    runpy.run_path(str(EXAMPLES / "ag_ui_bridge.py"), run_name="__main__")
    first, check, second, result = capsys.readouterr().out.splitlines()
    assert first.startswith("['RUN_STARTED', 'TOOL_CALL_START'")
    assert first.endswith("'TOOL_CALL_END', 'CUSTOM', 'RUN_FINISHED']")
    assert check == "tool_call True"
    assert second == (
        "['RUN_STARTED', 'TOOL_CALL_RESULT', 'TEXT_MESSAGE_START', "
        "'TEXT_MESSAGE_CONTENT', 'TEXT_MESSAGE_END', 'RUN_FINISHED']"
    )
    assert result == "refunded 120.0 on o-7"
