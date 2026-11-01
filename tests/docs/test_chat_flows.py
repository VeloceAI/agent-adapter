"""The help-desk chat example: one flow per pattern, `--check` runs offline, the page matches it."""

from __future__ import annotations

import asyncio
import re
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_adapter import AutoDeny, Executor, FlowRunner, Ledger
from agent_adapter.errors import FlowError
from agent_adapter.viewer import ChatMessage, ChatSession

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "chat_flows.py"
PAGE = ROOT / "docs" / "how-to" / "chat-flows.md"
FLOWS = ("ask_flow", "compare_flow", "booking_flow", "park_flow", "day_flow")


@pytest.fixture(scope="module")
def module() -> dict[str, Any]:
    return runpy.run_path(str(EXAMPLE), run_name="chat_flows_test")


def _graph(module: dict[str, Any], name: str) -> dict[str, Any]:
    return dict(module[name].graph())


def test_check_runs_every_question_offline(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(EXAMPLE), "--check"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    for flow in ("ask:", "compare_prices:", "booking:", "park_search:", "day_plan:"):
        assert flow in out, flow
    assert "closed on Mondays" in out
    assert "| Harbour Gallery | 9.00 EUR |" in out
    assert "Booked: **2 tickets**, City Museum, Saturday" in out
    assert "**Riverside Park** has a playground" in out
    assert "**City Museum** then **Science Hall**" in out


def test_each_question_picks_its_pattern(module: dict[str, Any]) -> None:
    picked = [module["pick_flow"](q) for q in module["QUESTIONS"]]
    assert picked == [module[name] for name in FLOWS]
    assert module["pick_flow"]("hello") is module["ask_flow"], "anything else: the router"


def test_the_flows_cover_router_fan_out_approval_loop_and_all_together(
    module: dict[str, Any],
) -> None:
    ask = _graph(module, "ask_flow")
    assert {e["target"] for e in ask["edges"] if e["source"] == "classify"} == {
        "hours",
        "prices",
        "help",
    }
    compare = _graph(module, "compare_flow")
    kinds = {n["id"]: n["kind"] for n in compare["nodes"]}
    assert (kinds["prices"], kinds["combine"]) == ("map", "join")
    assert module["registry"].get("book_tickets").confirm is True
    assert [n["tool"] for n in _graph(module, "booking_flow")["nodes"] if n.get("tool")] == [
        "ticket_price",
        "book_tickets",
    ]
    assert next(e for e in _graph(module, "park_flow")["edges"] if e["loop"])["target"] == "search"
    day = _graph(module, "day_flow")
    assert {n["kind"] for n in day["nodes"]} >= {"call", "map", "join"}
    assert any(e["loop"] for e in day["edges"]) and any(e["label"] == "rain" for e in day["edges"])


def test_a_denied_booking_fails_the_turn_and_books_nothing(module: dict[str, Any]) -> None:
    executor = Executor(
        module["registry"], ledger=Ledger(prices=module["PRICES"]), approval=AutoDeny()
    )
    session = ChatSession("ses_1", (), {}, executor, run_id="run-1")

    async def ask() -> str:
        return "".join([c async for c in module["chat"](ChatMessage("Book two tickets"), session)])

    with pytest.raises(FlowError, match=r"approval-denied.*book_tickets"):
        asyncio.run(ask())
    assert "questions" not in session.state


def test_the_reply_node_needs_the_writer_model(module: dict[str, Any]) -> None:
    runner = FlowRunner(Executor(module["registry"]))
    with pytest.raises(FlowError, match="FlowRunner"):
        asyncio.run(runner.run(module["ask_flow"], {"question": "Hi there"}))


def test_the_page_shows_each_flow_and_includes_its_code(module: dict[str, Any]) -> None:
    text = PAGE.read_text(encoding="utf-8")
    blocks = [b.strip() for b in re.findall(r"```mermaid\n(.*?)```", text, re.S)]
    for name in FLOWS:
        assert module[name].to_mermaid().strip() in blocks, name
    source = EXAMPLE.read_text(encoding="utf-8")
    for section in re.findall(r'--8<-- "examples/chat_flows\.py:([a-z_]+)"', text):
        assert f"# --8<-- [start:{section}]" in source and f"# --8<-- [end:{section}]" in source
    assert len(re.findall(r'--8<-- "examples/chat_flows\.py:', text)) >= 6
