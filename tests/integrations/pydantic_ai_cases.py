"""Pydantic AI bridge cases: deferred two-phase approval and gate details.

Collected through `test_pydantic_ai_contract.py` (so the contract session measures them).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import DeferredToolRequests

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalPending
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.pydantic_ai import pydantic_ai_tools, resolve_deferred
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations.pydantic_ai_support import agent_with, result_part, value_of
from tests.integrations.sample_tools import RecordingTools


class Tools(RecordingTools):
    """The shared recording tools plus `tag`, exported as Pydantic AI tools."""

    def __init__(self) -> None:
        super().__init__()

        @self.registry.tool
        async def tag(native: str, label: str = "x") -> str:
            """A parameter named like the bridge's own context argument."""
            return f"{native}:{label}"

    def export(self, **kwargs: Any) -> dict[str, Any]:
        return {tool.name: tool for tool in pydantic_ai_tools(self.registry, **kwargs)}


async def test_deferred_approval_one_decision_per_pass_finishes_without_reasking():
    tools, store = Tools(), MemoryApprovalStore()
    port = DeferredApproval(store)
    calls = [("wipe", {"path": "/a"}, "w1"), ("wipe", {"path": "/b"}, "w2")]
    agent = agent_with([tools.export(executor=Executor())["wipe"]], calls)
    first = await agent.run("go")
    requests = first.output
    assert isinstance(requests, DeferredToolRequests)
    with pytest.raises(ApprovalPending):  # pass 1: both records created
        await resolve_deferred(requests, port, run_id="run_x")
    one = await store.find("run_x", "w1")
    assert one is not None
    await store.decide(one.approval_id, ApprovalDecision(approved=True))
    with pytest.raises(ApprovalPending) as caught:  # pass 2: w2 still open
        await resolve_deferred(requests, port, run_id="run_x")
    assert caught.value.call_id == "w2"
    kept = await store.find("run_x", "w1")
    assert kept is not None and kept.status == "approved" and not kept.consumed
    two = await store.find("run_x", "w2")
    assert two is not None
    await store.decide(two.approval_id, ApprovalDecision(approved=True))
    results = await resolve_deferred(requests, port, run_id="run_x")  # pass 3
    assert len(await store.list()) == 2 and set(results.approvals) == {"w1", "w2"}
    await agent.run(message_history=first.all_messages(), deferred_tool_results=results)
    assert sorted(tools.ran, key=str) == [("wipe", {"path": "/a"}), ("wipe", {"path": "/b"})]


async def test_confirm_true_never_asks_approval_for_invalid_args():
    tools = Tools()
    wipe = tools.export(executor=Executor())["wipe"]
    result = await agent_with([wipe], [("wipe", {"path": 3}, "w1")]).run("go")
    assert isinstance(result.output, str)
    part = result_part(result, "w1")
    assert part.outcome == "retried" and "invalid arguments" in part.content
    assert tools.ran == []


async def test_a_callable_rule_that_says_no_is_checked_again_by_the_executor():
    tools, port = Tools(), ScriptedApproval([])  # the executor must not ask its port either
    pay = tools.export(executor=Executor(approval=port))["pay"]
    result = await agent_with([pay], [("pay", {"amount": 7}, "c1")]).run("go")
    assert value_of(result_part(result, "c1")) == "paid 7"
    assert tools.asked == [7, 7] and port.requests == []  # the gate, then the executor


async def test_a_parameter_named_native_reaches_the_tool():
    tag = Tools().export(executor=Executor())["tag"]
    result = await agent_with([tag], [("tag", {"native": "n", "label": "l"}, "c1")]).run("go")
    assert value_of(result_part(result, "c1")) == "n:l"


async def test_a_wrapped_deferred_approval_resolves():
    tools, store = Tools(), MemoryApprovalStore()
    port = RecordingApproval(DeferredApproval(store))
    calls = [("wipe", {"path": "/a"}, "w1"), ("wipe", {"path": "/b"}, "w2")]
    agent = agent_with([tools.export(executor=Executor())["wipe"]], calls)
    first = await agent.run("go")
    for call_id in ("w1", "w2"):  # one decision per pass
        with pytest.raises(ApprovalPending):
            await resolve_deferred(first.output, port, run_id="run_w")
        record = await store.find("run_w", call_id)
        assert record is not None
        await store.decide(record.approval_id, ApprovalDecision(approved=True))
    results = await resolve_deferred(first.output, port, run_id="run_w")  # third pass
    await agent.run(message_history=first.all_messages(), deferred_tool_results=results)
    assert len(tools.ran) == 2 and len(await store.list()) == 2
