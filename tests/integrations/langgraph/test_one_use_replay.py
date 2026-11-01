"""A node wrapped by `one_use` replays its finished executor calls when LangGraph re-runs it."""

import asyncio
from collections import Counter
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import MessagesState
from pydantic import ValidationError

from agent_adapter.core.errors import ApprovalAlreadyDecided
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.integrations.langgraph import LangGraphApproval, one_use
from agent_adapter.integrations.langgraph._spent import CLAIMS
from agent_adapter.integrations.langgraph._user_node import USER_NODE_MEMO
from tests.integrations.langgraph.support import config, one_node, resume

pytestmark = pytest.mark.contract

WIRED: Counter[int] = Counter()
registry = Registry()
APPROVE = ApprovalDecision(approved=True)


@registry.tool(confirm=True)
async def wire(amount: int) -> str:
    """Wire money (needs approval)."""
    WIRED[amount] += 1
    return f"wired {amount}"


@registry.tool
async def look(amount: int) -> int:
    """Read a balance (no approval)."""
    WIRED[-amount] += 1
    return amount


@pytest.fixture(autouse=True)
def _clean():
    WIRED.clear()
    CLAIMS.clear()
    USER_NODE_MEMO.clear()


def _wires(executor: Executor, *, ask_first: bool = False) -> Any:
    port = executor.approval

    async def node(state: Any) -> dict[str, Any]:
        ctx = executor.context(run_id="user-run")
        if ask_first:  # an approval the node asks itself, before its tool calls
            request = ApprovalRequest(approval_id="apr_x", run_id="user-run", call_id="x", tool="x")
            assert (await port.request(request)).approved
        await executor.run("look", {"amount": 7}, ctx, call_id="look")
        await executor.run("wire", {"amount": 1}, ctx, call_id="w1")
        await executor.run("wire", {"amount": 2}, ctx, call_id="w2")
        await asyncio.sleep(0.03)  # a duplicate overlaps this
        return {"messages": []}

    return node


async def _paused_twice(app: Any, cfg: Any, first: int = 2) -> list[Any]:
    out = await app.ainvoke({"messages": []}, cfg, version="v2")
    asked = list(out.interrupts)
    for _ in range(first - 1):
        out = await app.ainvoke(resume(asked[-1], APPROVE), cfg, version="v2")
        asked.extend(out.interrupts)
    return asked


async def test_a_one_use_node_with_two_approved_calls_runs_each_once():
    executor = Executor(registry, approval=LangGraphApproval())
    app, cfg = one_node(one_use(_wires(executor)), MessagesState), config()
    a, b = await _paused_twice(app, cfg)
    assert WIRED == {1: 1, -7: 1}  # paused at the second wire
    paused = (await app.aget_state(cfg)).config
    command = resume(b, APPROVE)
    results = await asyncio.gather(
        app.ainvoke(command, cfg), app.ainvoke(command, cfg), return_exceptions=True
    )
    failed = [r for r in results if isinstance(r, BaseException)]
    assert [type(e) for e in failed] == [ApprovalAlreadyDecided]
    assert WIRED == {1: 1, 2: 1, -7: 1}  # every finished call replayed, approved or not
    assert len(USER_NODE_MEMO) == 0  # the finished task forgot its calls
    for replay in (resume(b, APPROVE), resume(a, APPROVE)):  # the paused checkpoint again
        with pytest.raises(ApprovalAlreadyDecided):
            await app.ainvoke(replay, paused)
    assert (WIRED[1], WIRED[2]) == (1, 1)


async def test_a_node_that_asks_itself_and_then_calls_tools_keeps_its_resume_positions():
    executor = Executor(registry, approval=LangGraphApproval())
    app, cfg = one_node(one_use(_wires(executor, ask_first=True)), MessagesState), config()
    *_, last = await _paused_twice(app, cfg, first=3)
    done = await app.ainvoke(resume(last, APPROVE), cfg, version="v2")
    assert not done.interrupts
    assert WIRED == {1: 1, 2: 1, -7: 1}


async def test_a_failed_one_use_node_forgets_its_calls():
    executor = Executor(registry, approval=LangGraphApproval())

    async def node(state: Any) -> dict[str, Any]:
        ctx = executor.context(run_id="user-run")
        await executor.run("wire", {"amount": 1}, ctx, call_id="w1")
        await executor.run("wire", {"amount": 2}, ctx, call_id="w2")
        raise RuntimeError("after the wires")

    app, cfg = one_node(one_use(node), MessagesState), config()
    [a] = (await app.ainvoke({"messages": []}, cfg, version="v2")).interrupts
    [b] = (await app.ainvoke(resume(a, APPROVE), cfg, version="v2")).interrupts
    assert len(USER_NODE_MEMO) == 1
    with pytest.raises(RuntimeError, match="after the wires"):
        await app.ainvoke(resume(b, APPROVE), cfg)
    assert len(USER_NODE_MEMO) == 0
    with pytest.raises(ApprovalAlreadyDecided):  # the retry: both answers were used
        await app.ainvoke(None, cfg)
    assert WIRED == {1: 1, 2: 1}


async def test_an_invalid_resume_value_keeps_the_finished_calls_for_the_next_answer():
    executor = Executor(registry, approval=LangGraphApproval())
    app, cfg = one_node(one_use(_wires(executor)), MessagesState), config()
    _, b = await _paused_twice(app, cfg)
    with pytest.raises(ValidationError):  # signed with another key: LangGraph's invalid resume
        await app.ainvoke(resume(b, APPROVE, key=b"another-key-of-32-bytes-length!!"), cfg)
    done = await app.ainvoke(resume(b, APPROVE), cfg, version="v2")
    assert not done.interrupts
    assert WIRED == {1: 1, 2: 1, -7: 1}
