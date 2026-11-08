import dataclasses
import logging

import pytest

from agent_adapter import (
    ApprovalDecision,
    ApprovalDenied,
    ApprovalPending,
    Registry,
    ToolCall,
    ToolContext,
)
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core import control
from agent_adapter.errors import RunLimitExceeded, ToolUsageError
from agent_adapter.exec import Executor, register_control_flow
from tests.exec.helpers import DeferredFake, check_events, make
from tests.fakes.approval import ScriptedApproval

YES = ApprovalDecision(approved=True)

registry = Registry()
sent: list[tuple[str, int]] = []


@registry.tool(confirm=True)
async def transfer(to: str, amount: int = 10) -> str:
    """Move money."""
    sent.append((to, amount))
    return f"{amount} to {to}"


@pytest.fixture(autouse=True)
def _reset():
    sent.clear()


async def test_confirm_asks_the_port_and_emits_approval_events():
    port = ScriptedApproval([YES])
    ex, sink, _, _ = make(registry, approval=port)
    result = await ex.run_call(
        ToolCall(name="transfer", arguments={"to": "bob"}, id="c1"), ex.context(run_id="r")
    )
    assert result.ok and sent == [("bob", 10)]
    [request] = port.requests
    assert (request.run_id, request.call_id, request.tool, request.kind) == (
        "r",
        "c1",
        "transfer",
        "tool",
    )
    assert request.arguments == {"to": "bob", "amount": 10}
    assert request.approval_id.startswith("apr_") and request.created_at > 0
    assert check_events(sink) == [
        "tool.started",
        "approval.requested",
        "approval.decided",
        "tool.finished",
    ]
    decided = sink.of_type("approval.decided")[0]
    assert decided.data == {
        "tool": "transfer",
        "approval_id": request.approval_id,
        "approved": True,
        "edited": False,
    }


async def test_denied_calls_raise_approval_denied_with_the_reason():
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="too much")])
    ex, sink, ledger, _ = make(registry, approval=port)
    with pytest.raises(ApprovalDenied, match="too much") as info:
        await ex.run("transfer", {"to": "eve"})
    assert info.value.reason == "too much" and sent == []
    assert sink.of_type("approval.decided")[0].data["approved"] is False
    assert ledger.entries()[0].error_kind == "approval-denied"


async def test_no_port_denies_with_one_warning(caplog):
    ex = Executor(registry)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.exec._approve"):
        result = await ex.run_call(ToolCall(name="transfer", arguments={"to": "SECRET-IBAN"}))
    assert result.error.kind == "approval-denied"
    assert result.exception.reason == "no approval port configured"
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    assert "SECRET-IBAN" not in caplog.text  # arguments of confirm tools are never logged


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_callable_confirm_rules_see_the_validated_call_and_context(asynchronous):
    seen = []

    def rule(call, ctx):
        seen.append((call.arguments, ctx.run_id))
        return call.arguments["amount"] > 100

    async def async_rule(call, ctx):
        return rule(call, ctx)

    reg = Registry()
    reg.add(
        dataclasses.replace(registry.get("transfer"), confirm=async_rule if asynchronous else rule)
    )
    port = ScriptedApproval([YES])
    ex = Executor(reg, approval=port)
    await ex.run("transfer", {"to": "a"}, ex.context(run_id="r"))
    assert port.requests == []
    await ex.run("transfer", {"to": "b", "amount": 500})
    assert len(port.requests) == 1 and seen[0] == ({"to": "a", "amount": 10}, "r")


async def test_a_failing_confirm_rule_fails_the_call_and_is_logged(caplog):
    def rule(call, ctx):
        raise RuntimeError("rule broke")

    reg = Registry()
    reg.add(dataclasses.replace(registry.get("transfer"), confirm=rule, retries=3))
    ex = Executor(reg, approval=ScriptedApproval([]))
    with caplog.at_level(logging.ERROR, logger="agent_adapter.approval.builtin"):
        result = await ex.run_call(ToolCall(name="transfer", arguments={"to": "SECRET-IBAN"}))
    assert isinstance(result.exception, RuntimeError) and result.attempts == 0
    assert "confirm rule for tool 'transfer' failed" in caplog.text and sent == []
    assert "SECRET-IBAN" not in caplog.text  # confirm-tool arguments are never logged


async def test_a_non_bool_confirm_rule_is_a_usage_error_and_the_tool_does_not_run():
    reg = Registry()
    reg.add(dataclasses.replace(registry.get("transfer"), confirm=lambda call, ctx: "yes"))
    ex = Executor(reg, approval=ScriptedApproval([YES]))
    result = await ex.run_call(ToolCall(name="transfer", arguments={"to": "a"}))
    assert isinstance(result.exception, ToolUsageError) and "must return a bool" in str(
        result.exception
    )
    assert sent == []


async def test_control_flow_from_a_confirm_rule_is_not_logged_as_an_error(caplog, monkeypatch):
    class Interrupt(Exception):
        pass

    def rule(call, ctx):
        raise Interrupt

    monkeypatch.setattr(control, "_registered", ())  # keep the global registry clean
    register_control_flow(Interrupt)
    reg = Registry()
    reg.add(dataclasses.replace(registry.get("transfer"), confirm=rule))
    ex = Executor(reg, approval=ScriptedApproval([]))
    log = caplog.at_level(logging.ERROR, logger="agent_adapter.approval.builtin")
    with log, pytest.raises(Interrupt):
        await ex.run_call(ToolCall(name="transfer", arguments={"to": "a"}))
    assert caplog.records == [] and sent == []


async def test_a_reused_call_id_with_other_arguments_fails_closed_and_the_tool_does_not_run():
    store = MemoryApprovalStore()
    ex = Executor(registry, approval=DeferredApproval(store))
    ctx = ex.context(run_id="run_reuse")
    first = await ex.run_call(ToolCall(name="transfer", arguments={"to": "a"}, id="c1"), ctx)
    assert first.status == "pending"
    await store.decide(first.error.details["approval_id"], YES)
    swapped = ToolCall(name="transfer", arguments={"to": "mallory", "amount": 900}, id="c1")
    result = await ex.run_call(swapped, ctx)
    assert isinstance(result.exception, ToolUsageError) and result.attempts == 0
    assert sent == []
    with pytest.raises(ToolUsageError, match="call id reused"):
        await ex.run("transfer", {"to": "mallory", "amount": 900}, ctx, call_id="c1")
    assert sent == []
    resumed = await ex.run_call(ToolCall(name="transfer", arguments={"to": "a"}, id="c1"), ctx)
    assert resumed.ok and sent == [("a", 10)]


async def test_edited_args_are_validated_again_and_used():
    port = ScriptedApproval(
        [ApprovalDecision(approved=True, edited_args={"to": "bob", "amount": 1})]
    )
    ex = Executor(registry, approval=port)
    assert await ex.run("transfer", {"to": "bob", "amount": 900}) == "1 to bob"


async def test_invalid_edited_args_fail_validation():
    port = ScriptedApproval([ApprovalDecision(approved=True, edited_args={"amount": "lots"})])
    ex = Executor(registry, approval=port)
    result = await ex.run_call(ToolCall(name="transfer", arguments={"to": "bob"}))
    assert result.error.kind == "tool-args-invalid" and sent == []


async def test_pre_approval_on_the_context_skips_the_port():
    port = ScriptedApproval([])
    ex, sink, _, _ = make(registry, approval=port)
    ctx = ToolContext(run_id="r", approval=ApprovalDecision(approved=True, edited_args={"to": "z"}))
    assert await ex.run("transfer", {"to": "a"}, ctx) == "10 to z"
    assert port.requests == [] and sink.of_type("approval.") == []


async def test_a_denying_pre_approval_denies():
    ctx = ToolContext(run_id="r", approval=ApprovalDecision(approved=False, reason="host said no"))
    with pytest.raises(ApprovalDenied, match="host said no"):
        await Executor(registry).run("transfer", {"to": "a"}, ctx)


async def test_nested_calls_do_not_inherit_the_parent_pre_approval():
    reg = Registry()
    ex = Executor(reg, approval=ScriptedApproval([ApprovalDecision(approved=False, reason="ask")]))
    reg.add(registry.get("transfer"))

    @reg.tool(parallel_safe=True)
    async def pay(ctx: ToolContext) -> str:
        """Calls transfer with its own context."""
        return await ex.run("transfer", {"to": "n"}, ctx)

    ctx = ToolContext(run_id="r", approval=YES)
    with pytest.raises(ApprovalDenied, match="ask"):
        await ex.run("pay", {}, ctx)


async def test_pending_approval_is_a_pending_result_without_ledger_entry():
    port = DeferredFake()
    ex, sink, ledger, _ = make(registry, approval=port)
    result = await ex.run_call(ToolCall(name="transfer", arguments={"to": "a"}), ex.context())
    assert result.status == "pending" and result.error.details == {"approval_id": "apr_pending"}
    assert check_events(sink) == ["tool.started", "approval.requested", "tool.pending"]
    assert sink.events[-1].data["approval_id"] == "apr_pending"
    assert ledger.entries() == []
    with pytest.raises(ApprovalPending):
        await ex.run("transfer", {"to": "a"})


async def test_guards_count_distinct_call_ids_across_an_approval_resume():
    reg = Registry()
    reg.add(dataclasses.replace(registry.get("transfer"), once_per_run=True))
    port = DeferredFake()
    ex, _, ledger, _ = make(reg, approval=port, limits=None)
    ctx = ex.context(run_id="run_b5")
    first = await ex.run_call(ToolCall(name="transfer", arguments={"to": "a"}, id="c1"), ctx)
    assert first.status == "pending"
    port.decide("run_b5", "c1", YES)
    resumed = await ex.run_call(ToolCall(name="transfer", arguments={"to": "a"}, id="c1"), ctx)
    assert resumed.ok and sent == [("a", 10)]
    again = await ex.run_call(ToolCall(name="transfer", arguments={"to": "a"}, id="c2"), ctx)
    assert isinstance(again.exception, RunLimitExceeded) and again.exception.limit == "once_per_run"
    assert [e.call_id for e in ledger.entries() if e.ok] == ["c1"]
