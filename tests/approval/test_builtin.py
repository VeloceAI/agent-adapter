import logging

import pytest

from agent_adapter.approval.builtin import (
    AutoApprove,
    AutoDeny,
    CallbackApproval,
    evaluate_confirm,
)
from agent_adapter.core import control
from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import register_control_flow
from agent_adapter.core.errors import ApprovalPending, ToolUsageError
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest, ToolCall

REQUEST = ApprovalRequest(
    approval_id="apr_1", run_id="run_1", call_id="call_1", tool="delete", arguments={"pw": "x"}
)
CALL = ToolCall(name="delete", arguments={"path": "/tmp"}, id="call_1")
CTX = ToolContext(run_id="run_1", call_id="call_1")


@pytest.mark.parametrize("rule", [True, False])
async def test_evaluate_confirm_bool_rule_is_returned(rule):
    assert await evaluate_confirm(rule, CALL, CTX) is rule


async def test_evaluate_confirm_sync_callable_gets_call_and_context():
    seen = []

    def rule(call, ctx):
        seen.append((call, ctx))
        return call.arguments["path"] == "/tmp"

    assert await evaluate_confirm(rule, CALL, CTX) is True
    assert seen == [(CALL, CTX)]


async def test_evaluate_confirm_async_callable_is_awaited():
    async def rule(call, ctx):
        return False

    assert await evaluate_confirm(rule, CALL, CTX) is False


async def test_evaluate_confirm_non_bool_result_is_a_usage_error():
    with pytest.raises(ToolUsageError, match="must return a bool"):
        await evaluate_confirm(lambda call, ctx: "yes", CALL, CTX)


async def test_evaluate_confirm_rule_error_is_logged_and_reraised(caplog):
    def rule(call, ctx):
        raise KeyError("boom")

    with caplog.at_level(logging.ERROR), pytest.raises(KeyError):
        await evaluate_confirm(rule, CALL, CTX)
    assert "confirm rule for tool 'delete' failed" in caplog.text
    assert "/tmp" not in caplog.text  # arguments of confirm tools are never logged


async def test_evaluate_confirm_rejects_a_non_rule():
    with pytest.raises(ToolUsageError, match="bool or a callable"):
        await evaluate_confirm(1, CALL, CTX)  # type: ignore[arg-type]


async def test_auto_approve_and_auto_deny():
    approve, deny = AutoApprove(), AutoDeny(reason="read-only mode")
    assert isinstance(approve, ApprovalPort) and isinstance(deny, ApprovalPort)
    yes = await approve.request(REQUEST)
    no = await deny.request(REQUEST)
    assert (yes.approved, yes.reason, yes.decided_by) == (True, "auto", "auto")
    assert (no.approved, no.reason, no.decided_by) == (False, "read-only mode", "auto")
    assert yes.decided_at > 0 and no.decided_at > 0
    assert (await AutoDeny().request(REQUEST)).reason == "denied by policy"


@pytest.mark.parametrize("answer", [True, False])
async def test_callback_bool_becomes_a_decision(answer):
    port = CallbackApproval(lambda req: answer)
    decision = await port.request(REQUEST)
    assert decision.approved is answer and decision.decided_by == "callback"


async def test_callback_decision_passes_through_and_async_is_awaited():
    expected = ApprovalDecision(approved=True, edited_args={"path": "/safe"}, decided_by="ana")

    async def ask(req):
        assert req is REQUEST
        return expected

    assert await CallbackApproval(ask).request(REQUEST) is expected


async def test_callback_bad_return_is_a_usage_error():
    with pytest.raises(ToolUsageError, match="ApprovalDecision or bool"):
        await CallbackApproval(lambda req: "ok").request(REQUEST)


async def test_callback_pending_propagates_without_error_log(caplog):
    def ask(req):
        raise ApprovalPending(req.approval_id, tool=req.tool)

    with caplog.at_level(logging.DEBUG), pytest.raises(ApprovalPending):
        await CallbackApproval(ask).request(REQUEST)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_callback_error_is_logged_without_arguments_and_reraised(caplog):
    def ask(req):
        raise RuntimeError("reviewer service down")

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        await CallbackApproval(ask).request(REQUEST)
    assert "approval callback failed for tool 'delete'" in caplog.text
    assert "'pw'" not in caplog.text


class GraphBubbleUp(Exception):
    """Same name as LangGraph's interrupt signal."""


async def test_graph_interrupts_propagate_without_error_log(caplog):
    def ask(req):
        raise GraphBubbleUp()

    def rule(call, ctx):
        raise GraphBubbleUp()

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(GraphBubbleUp):
            await CallbackApproval(ask).request(REQUEST)
        with pytest.raises(GraphBubbleUp):
            await evaluate_confirm(rule, CALL, CTX)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_registered_control_flow_types_propagate_without_error_log(caplog, monkeypatch):
    class HostInterrupt(Exception):
        pass

    def raise_it(*_):
        raise HostInterrupt()

    monkeypatch.setattr(control, "_registered", ())
    register_control_flow(HostInterrupt)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(HostInterrupt):
            await evaluate_confirm(raise_it, CALL, CTX)
        with pytest.raises(HostInterrupt):
            await CallbackApproval(raise_it).request(REQUEST)
    assert caplog.records == []
