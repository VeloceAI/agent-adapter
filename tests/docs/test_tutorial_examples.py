"""The tutorial examples run offline and print what the tutorials show."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _run(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_research_agent_retrieves_in_parallel_pauses_resumes_and_reports_cost(
    capsys: pytest.CaptureFixture[str],
):
    assert _run("tutorial_research.py", capsys) == [
        "['web: EV battery #1', 'web: EV battery #2', 'paper: EV battery survey',"
        " 'news (us): EV battery rules change']",
        "pending []",
        "publish_report ['body', 'title']",
        "stop Done: published 'EV battery (draft)' (112 chars) ['EV battery (draft)']",
        "$0.0084 {'tool': 7, 'llm': 3}",
        '{"search_web": 0.0, "search_papers": 0.0, "search_news": 0.0, "tool_loop": 0.0084,'
        ' "publish_report": 0.0}',
    ]


def test_migrate_langgraph_before_after_one_use_and_compiled_flow(
    capsys: pytest.CaptureFixture[str],
):
    pytest.importorskip("langgraph")
    pytest.importorskip("langgraph.prebuilt")
    assert _run("tutorial_langgraph.py", capsys) == [
        "before: ['18.0', 'sent 2026-09']",
        "asks send_statement {'period': '2026-09'}",
        "refused bare",
        "refused forged",
        "after: ['18.0', 'sent 2026-09']",
        "one_use: sent 2026-10",
        "paused send_statement {'period': '2026-11'}",
        "done 18.0 sent 2026-11",
    ]
