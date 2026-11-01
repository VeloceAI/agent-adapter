"""`tool_node` approvals: one interrupt at a time, bound resumes, re-runs without double counts."""

import json
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage
from langgraph.types import Command
from pydantic import ValidationError

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalPending
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.cost import Ledger
from agent_adapter.exec import Executor
from agent_adapter.integrations.langgraph import LangGraphApproval, tool_node
from agent_adapter.integrations.langgraph.tools import TOOL_NODE_MEMO
from tests.integrations.langgraph.support import config, one_node, resume

pytestmark = pytest.mark.contract

RUNS: list[str] = []
registry = Registry()


@registry.tool(once_per_run=True)
async def lookup(item: str) -> str:
    """Look an item up (once per run)."""
    RUNS.append(f"lookup {item}")
    return f"found {item}"


@registry.tool(confirm=True)
async def delete(item: str) -> str:
    """Delete an item (needs approval)."""
    RUNS.append(f"delete {item}")
    return f"deleted {item}"


APPROVE = ApprovalDecision(approved=True)


def calls() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "delete", "args": {"item": "a"}, "id": "d1"},
            {"name": "lookup", "args": {"item": "x"}, "id": "l1"},
            {"name": "delete", "args": {"item": "b"}, "id": "d2"},
        ],
    )


def app_and_ledger() -> tuple[Any, Ledger]:
    ledger = Ledger()
    executor = Executor(registry, approval=LangGraphApproval(), ledger=ledger)
    return one_node(tool_node(registry, executor=executor)), ledger


@pytest.fixture(autouse=True)
def _clean():
    RUNS.clear()
    TOOL_NODE_MEMO.clear()


async def test_two_confirm_calls_interrupt_one_at_a_time_after_the_others():
    app, ledger = app_and_ledger()
    cfg = config("thread-a")
    out = await app.ainvoke({"messages": [calls()]}, cfg, version="v2")
    [first] = out.interrupts
    assert first.value["request"]["call_id"] == "d1"
    assert RUNS == ["lookup x"]  # the plain call ran first; nothing else asked yet
    out = await app.ainvoke(resume(first, APPROVE), cfg, version="v2")
    [second] = out.interrupts
    assert second.value["request"]["call_id"] == "d2"
    assert second.value["request"]["arguments"] == {"item": "b"}
    out = await app.ainvoke(resume(second, ApprovalDecision(approved=False)), cfg, version="v2")
    assert not out.interrupts
    tool_messages = out.value["messages"][1:]
    assert [m.tool_call_id for m in tool_messages] == ["d1", "l1", "d2"]
    assert [m.status for m in tool_messages] == ["success", "success", "error"]
    assert json.loads(tool_messages[2].content)["error"]["kind"] == "approval-denied"
    assert RUNS == ["lookup x", "delete a"]  # in-process re-runs replay finished calls
    assert sorted(e.call_id for e in ledger.entries("thread-a")) == ["d1", "d2", "l1"]
    assert len(TOOL_NODE_MEMO) == 0


async def test_a_re_run_without_the_memo_repeats_calls_but_never_counts_them_twice():
    app, ledger = app_and_ledger()
    cfg = config("thread-b")
    [first] = (await app.ainvoke({"messages": [calls()]}, cfg, version="v2")).interrupts
    TOOL_NODE_MEMO.clear()  # as after a process restart
    [second] = (await app.ainvoke(resume(first, APPROVE), cfg, version="v2")).interrupts
    TOOL_NODE_MEMO.clear()
    out = await app.ainvoke(resume(second, APPROVE), cfg, version="v2")
    assert not out.interrupts
    assert [m.status for m in out.value["messages"][1:]] == ["success"] * 3
    # LangGraph re-ran the node: the bodies ran again, the run guards and ledger did not count
    assert RUNS == ["lookup x", "lookup x", "delete a", "lookup x", "delete a", "delete b"]
    entries = ledger.entries("thread-b")
    assert sorted(e.call_id for e in entries) == ["d1", "d2", "l1"]  # once_per_run held too
    assert all(e.ok for e in entries)


async def test_edited_args_run_and_a_forged_resume_is_refused():
    app, _ = app_and_ledger()
    cfg = config()
    message = AIMessage(
        content="", tool_calls=[{"name": "delete", "args": {"item": "a"}, "id": "d1"}]
    )
    [first] = (await app.ainvoke({"messages": [message]}, cfg, version="v2")).interrupts
    forged = {"binding": "f" * 64, "decision": {"approved": True}}
    with pytest.raises(ValidationError):
        await app.ainvoke(Command(resume={first.id: forged}), cfg, version="v2")
    assert RUNS == []
    edit = ApprovalDecision(approved=True, edited_args={"item": "z"})
    out = await app.ainvoke(resume(first, edit), cfg, version="v2")
    assert out.value["messages"][-1].content == "deleted z"
    assert RUNS == ["delete z"]


async def test_without_an_executor_the_default_executor_asks_by_interrupt():
    app = one_node(tool_node(registry))
    cfg = config()
    message = AIMessage(
        content="", tool_calls=[{"name": "delete", "args": {"item": "q"}, "id": "d9"}]
    )
    [pending] = (await app.ainvoke({"messages": [message]}, cfg, version="v2")).interrupts
    assert pending.value["request"]["tool"] == "delete"
    out = await app.ainvoke(resume(pending, APPROVE), cfg, version="v2")
    assert out.value["messages"][-1].content == "deleted q"


async def test_a_deferred_port_pending_propagates_as_approval_pending():
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    app = one_node(tool_node(registry, executor=executor))
    message = AIMessage(
        content="", tool_calls=[{"name": "delete", "args": {"item": "q"}, "id": "d9"}]
    )
    with pytest.raises(ApprovalPending):
        await app.ainvoke({"messages": [message]}, config())
