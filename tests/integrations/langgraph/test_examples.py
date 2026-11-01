"""The LangGraph example runs and type-checks."""

import runpy
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

pytestmark = pytest.mark.contract

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "langgraph_flow.py"


def test_langgraph_example_pauses_resumes_and_compiles_a_flow(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "asks send_statement {'period': '2026-09'}",
        "['18.0', 'sent 2026-09']",
        "paused node:order.place {'rates': [18.0, 18.0]}",
        "done 2 {'rate': 1, 'place': 1} placed with 2 rates",
    ]


def test_langgraph_example_type_checks_strictly():
    import mypy.api

    limit = sys.getrecursionlimit()
    try:
        out, err, status = mypy.api.run(["--strict", "--no-error-summary", str(EXAMPLE)])
    finally:
        sys.setrecursionlimit(limit)  # mypy raises it for the whole process
    assert status == 0, out + err
