"""Agents SDK bridge cases: deferred two-phase approval, edits, gate details.

Collected through `test_openai_agents_contract.py` (so the contract session measures them).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agents.exceptions import UserError

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import (
    ApprovalPending,
    IntegrationError,
    ToolFailed,
    ToolTimeout,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.openai_agents import (
    _TOKENS,
    _take,
    function_tool,
    function_tools,
    resolve_interruptions,
)
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations.openai_agents_support import agent_with, decoded, output_of, run
from tests.integrations.sample_tools import RecordingTools


class Tools(RecordingTools):
    """The shared recording tools, exported as OpenAI Agents `FunctionTool`s."""

    def export(self, **kwargs: Any) -> dict[str, Any]:
        return {tool.name: tool for tool in function_tools(self.registry, **kwargs)}


async def test_deferred_approval_one_decision_per_pass_finishes_without_reasking():
    tools, store = Tools(), MemoryApprovalStore()
    port = DeferredApproval(store)
    agent = agent_with(
        [tools.export(executor=Executor())["wipe"]],
        [("wipe", {"path": "/a"}, "w1"), ("wipe", {"path": "/b"}, "w2")],
    )
    state = (await run(agent)).to_state()
    with pytest.raises(ApprovalPending):  # pass 1: both records created
        await resolve_interruptions(state, port, run_id="run_x")
    first = await store.find("run_x", "w1")
    assert first is not None
    await store.decide(first.approval_id, ApprovalDecision(approved=True))
    with pytest.raises(ApprovalPending) as caught:  # pass 2: w2 still open
        await resolve_interruptions(state, port, run_id="run_x")
    assert caught.value.call_id == "w2"
    kept = await store.find("run_x", "w1")
    assert kept is not None and kept.status == "approved" and not kept.consumed
    second = await store.find("run_x", "w2")
    assert second is not None
    await store.decide(second.approval_id, ApprovalDecision(approved=True))
    await resolve_interruptions(state, port, run_id="run_x")  # pass 3: both consumed
    assert len(await store.list()) == 2
    assert await store.find("run_x", "w1") is None and await store.find("run_x", "w2") is None
    await run(agent, state)
    assert sorted(tools.ran, key=str) == [("wipe", {"path": "/a"}), ("wipe", {"path": "/b"})]


async def test_a_refused_edit_consumes_no_deferred_decision():
    import agents

    store = MemoryApprovalStore()
    port = DeferredApproval(store)

    @agents.function_tool(needs_approval=True)
    def native_wipe(path: str) -> str:
        """A plain SDK tool: an edit cannot reach it."""
        return path

    agent = agent_with([native_wipe], [("native_wipe", {"path": "/a"}, "w1")])
    state = (await run(agent)).to_state()
    with pytest.raises(ApprovalPending):
        await resolve_interruptions(state, port, run_id="run_y")
    record = await store.find("run_y", "w1")
    assert record is not None
    edit = ApprovalDecision(approved=True, edited_args={"path": "/"})
    await store.decide(record.approval_id, edit)
    with pytest.raises(ToolUsageError, match="native_wipe"):
        await resolve_interruptions(state, port, run_id="run_y")
    assert not (await store.get(record.approval_id)).consumed


async def test_an_edit_for_a_tool_this_bridge_did_not_export_is_refused():
    import agents

    ran: list[str] = []

    @agents.function_tool(needs_approval=True)
    def native_wipe(path: str) -> str:
        """A plain SDK tool."""
        ran.append(path)
        return path

    ours = Tools().export(executor=Executor())
    agent = agent_with([native_wipe, ours["wipe"]], [("native_wipe", {"path": "/a"}, "n1")])
    first = await run(agent)
    port = ScriptedApproval([ApprovalDecision(approved=True, edited_args={"path": "/tmp"})])
    _TOKENS.clear()  # the table is module-wide: start from empty
    with pytest.raises(ToolUsageError, match="native_wipe"):
        await resolve_interruptions(first, port)
    assert ran == [] and len(_TOKENS) == 0


async def test_confirm_true_never_asks_approval_for_invalid_args():
    tools = Tools()
    wipe = tools.export(executor=Executor())["wipe"]
    result = await run(agent_with([wipe], [("wipe", {"path": 3}, "w1")]))
    assert not result.interruptions
    assert decoded(output_of(result, "w1"))["error"]["kind"] == "tool-args-invalid"
    assert tools.ran == []


async def test_a_callable_rule_that_says_no_is_checked_again_by_the_executor():
    tools, port = Tools(), ScriptedApproval([])  # the executor must not ask its port either
    pay = tools.export(executor=Executor(approval=port))["pay"]
    result = await run(agent_with([pay], [("pay", {"amount": 7}, "c1")]))
    assert output_of(result, "c1") == "paid 7"
    assert tools.asked == [7, 7] and port.requests == []  # the gate, then the executor


def test_a_token_matches_only_its_exact_call():
    class Scoped:
        _agent_tool_state_scope_id = "scope_1"

    spec = Tools().registry.get("pay")
    _TOKENS.clear()  # the table is module-wide: start from empty
    _TOKENS.issue(
        "pay", {"amount": 500}, ApprovalDecision(approved=True), call_id="c1", run_id="scope_1"
    )
    assert _take(spec, {"amount": 5}, "c1", Scoped()) is None  # other arguments
    assert _take(spec, {"amount": 500}, "", Scoped()) is None  # no call id
    assert _take(spec, {"amount": 500}, "c1", object()) is None  # no run state scope
    assert _take(spec, {"amount": "x"}, "c1", Scoped()) is None  # invalid arguments
    found = _take(spec, {"amount": 500}, "c1", Scoped())
    assert found is not None and found.approved and len(_TOKENS) == 0


async def test_needs_approval_asks_the_rule_and_never_for_invalid_arguments():
    tools = Tools()
    pay = tools.export(executor=Executor())["pay"]
    for call_id in ("c1", "c2"):
        assert await pay.needs_approval(None, {"amount": 1}, call_id) is False
    assert await pay.needs_approval(None, {"amount": "x"}, "c3") is False  # invalid: not asked
    assert tools.asked == [1, 1]


async def test_resolve_passes_the_run_id_to_the_port():
    wipe = Tools().export(executor=Executor())["wipe"]
    first = await run(agent_with([wipe], [("wipe", {"path": "/a"}, "w1")]))
    port = ScriptedApproval([ApprovalDecision(approved=True)])
    await resolve_interruptions(first, port, run_id="run_7")
    assert port.requests[0].run_id == "run_7"


async def test_timeouts_and_failures_surface_as_sdk_user_errors_with_our_cause():
    registry = Registry()

    @registry.tool(timeout=0.01)
    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(1)
        return "x"

    @registry.tool
    async def boom() -> str:
        """Boom."""
        raise RuntimeError("nope")

    exported = {t.name: t for t in function_tools(registry, executor=Executor())}
    for name, cause in (("slow", ToolTimeout), ("boom", ToolFailed)):
        with pytest.raises(UserError) as caught:
            await run(agent_with([exported[name]], [(name, {}, "c1")]))
        assert isinstance(caught.value.__cause__, cause)


def test_a_sdk_without_the_failure_formatter_switch_is_an_integration_error(monkeypatch):
    import agents.tool

    monkeypatch.delattr(agents.tool, "set_function_tool_failure_error_function")
    with pytest.raises(IntegrationError, match="set_function_tool_failure_error_function"):
        function_tool(Tools().registry.get("wipe"))


# --- fix round 2: exact records, wrapped ports --------------------------------------------


async def test_a_record_replaced_between_phases_is_rejected_not_applied():
    tools, store = Tools(), MemoryApprovalStore()
    wipe = tools.export(executor=Executor())["wipe"]
    agent = agent_with([wipe], [("wipe", {"path": "/a"}, "w1")])
    state = (await run(agent)).to_state()
    with pytest.raises(ApprovalPending):
        await resolve_interruptions(state, DeferredApproval(store), run_id="run_r")
    vetted = await store.find("run_r", "w1")
    assert vetted is not None
    await store.decide(vetted.approval_id, ApprovalDecision(approved=True))

    class RacingPort(DeferredApproval):
        async def consume(self, record: Any, request: Any) -> ApprovalDecision:
            assert await store.consume(record.approval_id)  # a concurrent resume wins
            rival = await store.create(request.model_copy(update={"approval_id": "apr_rival"}))
            edit = ApprovalDecision(approved=True, edited_args={"path": "/edited-unvetted"})
            await store.decide(rival.approval_id, edit)
            return await super().consume(record, request)

    await resolve_interruptions(state, RacingPort(store), run_id="run_r")
    result = await run(agent, state)
    assert decoded(output_of(result, "w1"))["error"]["kind"] == "approval-denied"
    assert tools.ran == []


async def test_a_wrapped_deferred_approval_resolves():
    tools, store = Tools(), MemoryApprovalStore()
    port = RecordingApproval(DeferredApproval(store))
    wipe = tools.export(executor=Executor())["wipe"]
    agent = agent_with([wipe], [("wipe", {"path": "/a"}, "w1"), ("wipe", {"path": "/b"}, "w2")])
    state = (await run(agent)).to_state()
    for call_id in ("w1", "w2"):  # one decision per pass
        with pytest.raises(ApprovalPending):
            await resolve_interruptions(state, port, run_id="run_w")
        record = await store.find("run_w", call_id)
        assert record is not None
        await store.decide(record.approval_id, ApprovalDecision(approved=True))
    await resolve_interruptions(state, port, run_id="run_w")  # third pass
    await run(agent, state)
    assert len(tools.ran) == 2 and len(await store.list()) == 2
