"""Runs the WP12 integration examples; each is skipped when its framework is not installed."""

import pytest

from tests.test_examples import _run


def test_openai_agents_example_runs_tools_and_applies_the_edited_approval(capsys):
    pytest.importorskip("agents")
    lines = _run("openai_agents_tools.py", capsys).splitlines()
    assert lines == ["['18.0', 'refunded 50.0 on A1']", "done"]


def test_pydantic_ai_example_defers_the_large_refund_and_applies_the_edit(capsys):
    pytest.importorskip("pydantic_ai")
    lines = _run("pydantic_ai_tools.py", capsys).splitlines()
    assert lines == ["['refunded 20.0 on A1', 'refunded 100.0 on B2']", "done"]


def test_llamaindex_example_calls_async_sync_and_reports_bad_args(capsys):
    pytest.importorskip("llama_index.core")
    lines = _run("llamaindex_tools.py", capsys).splitlines()
    assert lines == ["shipping_rate", "18.0 5.0", "tool-args-invalid"]


def test_crewai_example_runs_both_paths(capsys):
    pytest.importorskip("crewai")
    lines = _run("crewai_tools.py", capsys).splitlines()
    assert lines == [
        "shipping_rate Shipping rate for an item, in percent of the order value (by region).",
        "18.0 5.0",
    ]
