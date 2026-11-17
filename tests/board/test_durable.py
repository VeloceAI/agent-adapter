"""A resumed or recovered run gets its board back, even in a fresh process."""

from __future__ import annotations

import contextlib
from typing import Any

from pydantic import BaseModel

from agent_adapter import Registry, ToolCall, ToolContext
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.board import board_of, board_tools
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor, recover_tool_loop, run_tool_loop
from agent_adapter.flow import Flow, FlowRunner
from tests.fakes.models import ScriptedModel, calls, text
from tests.flow.durable_helpers import KEY, Crash, CrashingStore

ASK = [Message(role="user", content="research")]


class Notes(BaseModel):
    seen: list[str] = []


def tools() -> Registry:
    registry = Registry()
    registry.add_many(board_tools())

    @registry.tool(confirm=True)
    def publish(text: str) -> str:
        """Publish a note (needs approval)."""
        return f"published {text}"

    return registry


def notes_flow(registry: Registry, executor_of: dict[str, Executor]) -> Flow[Notes]:
    async def scout(state: Notes, ctx: ToolContext) -> dict[str, Any]:
        await board_of(ctx).post("finding", "prices rose 4%", by="scout")
        return {}

    async def act(state: Notes, ctx: ToolContext) -> dict[str, Any]:
        await executor_of["now"].run("publish", {"text": "report"}, ctx)
        return {"seen": [e.text for e in await board_of(ctx).read(kind="finding")]}

    flow = Flow("notes", state=Notes).node("scout", scout).node("act", act)
    return flow.edge("scout", "act").end("act")


async def test_a_paused_flow_resumes_with_its_board_on_a_fresh_executor():
    registry = tools()
    holder: dict[str, Executor] = {}
    first = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    holder["now"] = first
    flow = notes_flow(registry, holder)
    paused = await FlowRunner(first, checkpoint_key=KEY).run(flow, {})
    assert paused.status == "paused"
    assert [row["text"] for row in paused.checkpoint._board][:1] == ["prices rose 4%"]

    second = Executor(registry)  # a new process: its in-memory board store is empty
    holder["now"] = second
    runner = FlowRunner(second, checkpoint_key=KEY)
    text_form = runner.dump_checkpoint(paused.checkpoint)
    loaded = runner.load_checkpoint(flow, text_form)
    done = await runner.resume(flow, loaded, ApprovalDecision(approved=True))
    assert done.status == "done" and done.state.seen == ["prices rose 4%"]


async def test_a_crashed_durable_flow_recovers_its_board():
    registry = tools()

    async def scout(state: Notes, ctx: ToolContext) -> dict[str, Any]:
        await board_of(ctx).post("finding", "found it", by="scout")
        return {}

    crashes = {"left": 1}

    async def read(state: Notes, ctx: ToolContext) -> dict[str, Any]:
        if crashes["left"]:
            crashes["left"] -= 1
            raise Crash("worker died")
        return {"seen": [e.text for e in await board_of(ctx).read(kind="finding")]}

    flow = Flow("crash", state=Notes).node("scout", scout).node("read", read)
    flow = flow.edge("scout", "read").end("read")
    store = CrashingStore()
    with contextlib.suppress(Crash):
        await FlowRunner(Executor(registry), checkpoints=store, checkpoint_key=KEY).run(
            flow, {}, Executor(registry).context(run_id="run-f")
        )
    runner = FlowRunner(Executor(registry), checkpoints=store, checkpoint_key=KEY)
    result = await runner.recover(flow, "run-f")
    assert result.state.seen == ["found it"]


async def test_a_crashed_durable_tool_loop_recovers_its_board():
    registry = tools()
    ran = {"boom": 0}

    @registry.tool(idempotent=True)
    def boom() -> str:
        """Fails once, like a worker dying."""
        ran["boom"] += 1
        if ran["boom"] == 1:
            raise Crash("worker died")
        return "fine"

    post = ToolCall(name="board_post", arguments={"kind": "finding", "text": "fact"}, id="p1")
    store = CrashingStore()
    model = ScriptedModel([calls(post), calls(ToolCall(name="boom", arguments={}, id="b1"))])
    with contextlib.suppress(Crash):
        await run_tool_loop(
            model, registry, ASK, checkpoints=store, checkpoint_key=KEY, run_id="run-l"
        )
    fresh = Executor(registry)  # nothing in memory: the board comes from the checkpoint
    ctx = fresh.context(run_id="run-l")
    again = ScriptedModel([text("done")])
    result = await recover_tool_loop(
        again, registry, "run-l", checkpoints=store, checkpoint_key=KEY, executor=fresh, ctx=ctx
    )
    assert result.finish == "stop"
    assert [e.text for e in await board_of(ctx).read(kind="finding")] == ["fact"]
