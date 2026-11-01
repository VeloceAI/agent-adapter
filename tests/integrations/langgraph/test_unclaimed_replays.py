"""An answer whose call already ran is never handed to a retry, whatever failed in between."""

import asyncio
from collections import Counter
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState
from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalAlreadyDecided, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.langgraph import (
    LangGraphApproval,
    compile_flow,
    one_use,
    run_compiled,
)
from agent_adapter.integrations.langgraph._spent import CLAIMS
from agent_adapter.integrations.langgraph._user_node import USER_NODE_MEMO
from tests.integrations.langgraph.support import config, one_node, resume

pytestmark = pytest.mark.contract

RAN: Counter[str] = Counter()
STARTED = asyncio.Event()
registry = Registry()
APPROVE = ApprovalDecision(approved=True)


class Opaque:
    """A result the replay journal cannot hold."""


@registry.tool(confirm=True)
async def wire(amount: int) -> str:
    """Wire money (needs approval)."""
    RAN[f"wire {amount}"] += 1
    return f"wired {amount}"


@registry.tool(confirm=True)
async def opaque() -> Any:
    """Return a live object (needs approval)."""
    RAN["opaque"] += 1
    return Opaque()


@registry.tool(confirm=True)
async def slow_wire(amount: int) -> str:
    """Wire money slowly (needs approval); the test cancels it mid-way."""
    RAN[f"wire {amount}"] += 1
    if not STARTED.is_set():  # the first run blocks until the test cancels it
        STARTED.set()
        await asyncio.sleep(30)
    return f"wired {amount}"


@pytest.fixture(autouse=True)
def _clean():
    global STARTED
    STARTED = asyncio.Event()  # an event binds to the loop of its first wait: one per test
    RAN.clear()
    CLAIMS.clear()
    USER_NODE_MEMO.clear()


async def _answer_both(app: Any, cfg: Any) -> Any:
    """Pause at the first ask, answer it, pause at the second; returns the second interrupt."""
    [a] = (await app.ainvoke({"messages": []}, cfg, version="v2")).interrupts
    [b] = (await app.ainvoke(resume(a, APPROVE), cfg, version="v2")).interrupts
    return b


async def test_a_one_use_rerun_failing_before_its_next_ask_spends_the_replayed_answer():
    executor = Executor(registry, approval=LangGraphApproval())
    reruns = Counter[str]()

    async def node(state: Any) -> dict[str, Any]:
        ctx = executor.context(run_id="user-run")
        await executor.run("wire", {"amount": 1}, ctx, call_id="w1")
        reruns["after w1"] += 1
        if reruns["after w1"] == 2:  # the re-run after the second answer
            raise ConnectionError("upstream down")
        await executor.run("wire", {"amount": 2}, ctx, call_id="w2")
        return {"messages": []}

    app, cfg = one_node(one_use(node), MessagesState), config()
    b = await _answer_both(app, cfg)
    with pytest.raises(ConnectionError):
        await app.ainvoke(resume(b, APPROVE), cfg)
    with pytest.raises(ApprovalAlreadyDecided):  # the host retries the failed task
        await app.ainvoke(None, cfg)
    assert RAN == {"wire 1": 1}


async def test_a_finished_result_the_journal_cannot_hold_is_not_run_again_on_retry():
    executor = Executor(registry, approval=LangGraphApproval())

    async def node(state: Any) -> dict[str, Any]:
        ctx = executor.context(run_id="user-run")
        await executor.run("opaque", {}, ctx, call_id="o1")
        await executor.run("wire", {"amount": 2}, ctx, call_id="w2")
        return {"messages": []}

    app, cfg = one_node(one_use(node), MessagesState), config()
    b = await _answer_both(app, cfg)
    with pytest.raises(ToolUsageError, match="could not be recorded"):
        await app.ainvoke(resume(b, APPROVE), cfg)
    with pytest.raises(ApprovalAlreadyDecided):  # the host retries the failed task
        await app.ainvoke(None, cfg)
    assert RAN == {"opaque": 1}


class Note(BaseModel):
    text: str = ""


def _slow_flow(executor: Executor) -> Any:
    from agent_adapter.flow import Flow

    async def code(state: Note, ctx: ToolContext) -> dict[str, Any]:
        await executor.run("slow_wire", {"amount": 5}, ctx)
        return {"text": "sent"}

    flow = Flow("slow", state=Note).node("go", code).end("go")
    return compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver())


async def _cancel_mid_tool(app: Any, cfg: Any, decision: Any) -> None:
    run = asyncio.ensure_future(run_compiled(app, decision, cfg))
    await asyncio.wait_for(STARTED.wait(), 5)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run


async def test_a_run_cancelled_while_the_approved_tool_runs_does_not_run_it_again():
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    app, cfg = _slow_flow(executor), config()
    await run_compiled(app, {}, cfg)
    await _cancel_mid_tool(app, cfg, APPROVE)
    with pytest.raises(ApprovalAlreadyDecided):  # the host retries the cancelled run
        await run_compiled(app, None, cfg)
    assert RAN == {"wire 5": 1}


async def test_a_cancelled_run_with_langgraph_approval_asks_again():
    executor = Executor(registry, approval=LangGraphApproval())
    app, cfg = _slow_flow(executor), config()
    await run_compiled(app, {}, cfg)
    await _cancel_mid_tool(app, cfg, APPROVE)
    again = await run_compiled(app, None, cfg)  # nothing was saved: the retry asks again
    assert again.status == "paused"
    assert again.pending is not None and again.pending.tool == "slow_wire"
    assert RAN == {"wire 5": 1}
