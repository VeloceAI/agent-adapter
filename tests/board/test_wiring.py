"""Each run's board reaches executors, agent tools, tool loops and flow nodes."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_adapter import Registry, ToolCall, ToolContext
from agent_adapter.board import (
    BOARD_KEY,
    Board,
    BoardSettings,
    SQLiteBoardStore,
    board_of,
    board_tools,
)
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import Message
from agent_adapter.exec import Executor, LoopAgent, agent_tool, run_tool_loop
from agent_adapter.flow import Flow, FlowRunner
from tests.fakes.models import ScriptedModel, calls, text
from tests.fakes.sinks import RecordingSink

ASK = [Message(role="user", content="plan a trip")]


def test_every_run_gets_its_own_board_from_the_executor_store():
    executor = Executor(Registry())
    one, two = executor.context(), executor.context()
    assert isinstance(one.extras[BOARD_KEY], Board)
    assert one.extras[BOARD_KEY].scope != two.extras[BOARD_KEY].scope
    assert one.extras[BOARD_KEY].store is two.extras[BOARD_KEY].store
    assert executor.replace(namespace="x").board.store is executor.board.store
    mine = Board(scope="mine")
    assert executor.context(extras={BOARD_KEY: mine}).extras[BOARD_KEY] is mine


def test_thread_scope_shares_a_board_across_runs(tmp_path):
    settings = BoardSettings(store=SQLiteBoardStore(tmp_path / "b.db"), scope="thread")
    executor = Executor(Registry(), board=settings)
    first = executor.context(thread_id="t1").extras[BOARD_KEY]
    second = executor.context(thread_id="t1").extras[BOARD_KEY]
    alone = executor.context().extras[BOARD_KEY]
    assert first.scope == second.scope == "thread:t1" and alone.scope.startswith("run:")
    assert isinstance(Executor(Registry(), board=settings.store).board, BoardSettings)
    with pytest.raises(ToolUsageError):
        Executor(Registry(), board="memory")  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError):
        BoardSettings(scope="global")  # type: ignore[arg-type]


async def test_agents_post_their_status_and_see_each_other():
    async def scout(task: str, ctx: ToolContext) -> str:
        await board_of(ctx).post("finding", "Oslo hotels: 3 under 100 EUR", key="hotels")
        return "found hotels"

    writer_model = ScriptedModel([text("trip plan")])
    writer = LoopAgent(writer_model, Registry(), instructions="Write.", board="auto")
    registry = Registry()
    executor = Executor(registry)
    registry.add(agent_tool(scout, name="scout", description="Finds hotels."))
    registry.add(agent_tool(writer, name="writer", description="Writes.", executor=executor))
    ctx = executor.context()
    await executor.run("scout", {"task": "find hotels in Oslo"}, ctx)
    assert await executor.run("writer", {"task": "write the plan"}, ctx) == "trip plan"

    entries = await board_of(ctx).read()
    assert [(e.by, e.kind) for e in entries] == [
        ("scout", "status"),
        ("scout", "finding"),
        ("scout", "status"),
        ("writer", "status"),
        ("writer", "status"),
    ]
    assert entries[0].text == "started: find hotels in Oslo"
    assert entries[2].text == "finished (ok)"
    sent = writer_model.requests[0].messages
    assert [m.role for m in sent] == ["system", "user", "user"]
    note = sent[2].content
    assert "<shared-board>" in note and "Oslo hotels" in note and "finished (ok)" in note
    assert "writer" not in note  # an agent never sees its own entries


async def test_the_note_carries_only_news_and_is_kept_in_the_history():
    registry = Registry()

    @registry.tool
    async def look(ctx: ToolContext) -> str:
        """Look around while another agent posts."""
        await board_of(ctx).post("finding", "the sea is calm", by="other")
        return "looked"

    executor = Executor(registry)
    ctx = executor.context()
    await board_of(ctx).post("finding", "the sky is blue", by="other")
    look_call = calls(ToolCall(name="look", arguments={}, id="l1"))
    model = ScriptedModel([look_call, look_call, text("ok")])
    result = await run_tool_loop(model, registry, ASK, executor=executor, ctx=ctx, board="auto")
    first, second, third = (r.messages for r in model.requests)
    assert "the sky is blue" in first[-1].content and "(up to #1)" in first[-1].content
    assert "the sea is calm" in second[-1].content and "sky" not in second[-1].content
    assert "the sea is calm" in third[-1].content  # the second look posted it again
    notes = [m for m in result.messages if "<shared-board>" in m.content]
    assert len(notes) == 3 and all(m.role == "user" for m in notes)

    quiet = ScriptedModel([look_call, text("ok")])
    mine = executor.context()
    await run_tool_loop(quiet, Registry(), ASK, executor=executor, ctx=mine, board="auto")
    assert len(quiet.requests[0].messages) == 1  # an empty board adds nothing
    with pytest.raises(ToolUsageError, match="board"):
        await run_tool_loop(quiet, registry, ASK, board="on")  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError, match="board_chars"):
        await run_tool_loop(quiet, registry, ASK, board="auto", board_chars=50)


async def test_board_tools_run_through_the_executor():
    registry = Registry()
    registry.add_many(board_tools())
    sink = RecordingSink()
    executor = Executor(registry, sink=sink)
    model = ScriptedModel(
        [
            calls(
                ToolCall(name="board_claim", arguments={"key": "search:oslo"}, id="c1"),
                ToolCall(
                    name="board_post",
                    arguments={"kind": "finding", "text": "3 hotels", "key": "oslo"},
                    id="c2",
                ),
            ),
            calls(ToolCall(name="board_read", arguments={"kind": "finding"}, id="c3")),
            text("done"),
        ]
    )
    result = await run_tool_loop(model, registry, ASK, executor=executor)
    tool_text = {m.tool_call_id: m.content for m in result.messages if m.role == "tool"}
    assert tool_text["c1"] == "claimed 'search:oslo': it is yours"
    assert tool_text["c2"] == "posted as #2"
    assert "3 hotels" in tool_text["c3"] and "<shared-board>" in tool_text["c3"]
    assert {e.data["tool"] for e in sink.of_type("tool.finished")} == {
        "board_claim",
        "board_post",
        "board_read",
    }
    assert [e.data["by"] for e in sink.of_type("board.posted")] == ["main"]
    assert all(not spec.confirm for spec in registry)
    with pytest.raises(ToolUsageError):
        board_tools(["write"])


async def test_a_losing_claim_names_the_holder():
    ctx = Executor(Registry()).context()
    await board_of(ctx).claim("search:oslo", by="scout")
    registry = Registry()
    registry.add_many(board_tools(["claim"]))
    answer = await Executor(registry).run("board_claim", {"key": "search:oslo"}, ctx)
    assert answer.startswith("scout already does 'search:oslo'")


class Trip(BaseModel):
    city: str = "Oslo"
    note: str = ""
    seen: list[str] = []


async def test_flow_nodes_post_their_status_and_read_earlier_nodes():
    async def look(state: Trip, ctx: ToolContext) -> dict[str, str]:
        await board_of(ctx).post("handoff", f"{state.city} is sunny")
        return {"note": "looked"}

    async def plan(state: Trip, ctx: ToolContext) -> dict[str, list[str]]:
        return {"seen": [e.text for e in await board_of(ctx).read(kind="handoff")]}

    flow = Flow("trip", state=Trip).node("look", look).node("plan", plan)
    flow = flow.edge("look", "plan").end("plan")
    executor = Executor(Registry())
    ctx = executor.context()
    result = await FlowRunner(executor).run(flow, {}, ctx)
    assert result.state.seen == ["Oslo is sunny"]
    entries = await board_of(ctx).read()
    assert [(e.by, e.kind, e.text) for e in entries] == [
        ("look", "handoff", "Oslo is sunny"),
        ("look", "status", "finished; set note"),
        ("plan", "status", "finished; set seen"),
    ]
