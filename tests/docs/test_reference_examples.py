"""Snippets on the concept pages that the examples' own `main()` does not print."""

import runpy
from pathlib import Path

from agent_adapter import ToolContext

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_strict_mode_loose_tools_keep_optional_fields_optional() -> None:
    ns = runpy.run_path(str(EXAMPLES / "strict_mode.py"))
    function = ns["loose_tools"]()[0]["function"]
    assert function["strict"] is False
    assert function["parameters"]["required"] == ["query"]
    assert function["parameters"]["properties"]["top_k"]["default"] == 8


async def test_registry_views_call_runs_through_executor_and_alone() -> None:
    ns = runpy.run_path(str(EXAMPLES / "registry_views.py"))
    hits, alone, later = await ns["call_search"](ToolContext())
    assert hits == alone == ["returns"] * 8
    assert later == ["returns", "returns"]
