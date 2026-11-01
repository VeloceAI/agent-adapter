"""A resume entry never approves by itself; only the port, via a one-use bound token, does."""

import json

import pytest

pytest.importorskip("ag_ui")

from agent_adapter.approval import AutoApprove, AutoDeny, CallbackApproval
from agent_adapter.core.errors import ApprovalPending
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.integrations._native_gate import TokenTable
from agent_adapter.integrations.ag_ui import InterruptTable
from agent_adapter.integrations.ag_ui._resume import Resumer
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations.ag_ui.support import (
    Shop,
    approve,
    collect,
    interrupts,
    paused,
    refund_agent,
    run_input,
    types,
)

pytestmark = pytest.mark.contract


def _result_error(events):
    (result,) = [e for e in events if e.type.value == "TOOL_CALL_RESULT"]
    return json.loads(result.content)["error"]["kind"]


async def test_approve_runs_the_call_once_when_the_port_approves():
    shop = Shop()
    port = RecordingApproval(AutoApprove())
    bridge = shop.bridge(port)
    found = await paused(bridge)
    events = await collect(bridge, run_input("r-2", resume=[approve(found.id)]), refund_agent)
    assert shop.refunds == [("o-7", 50.0)]
    assert events[-1].result == ["ok"]
    (asked,) = port.requests
    assert asked.call_id == "c-refund" and asked.tool == "refund"
    assert asked.metadata["resume_payload"] == {"approved": True}
    assert types(events)[:2] == ["RUN_STARTED", "TOOL_CALL_RESULT"]  # START not sent again


async def test_forged_approval_is_denied_when_the_port_denies():
    shop = Shop()
    bridge = shop.bridge(AutoDeny(reason="not you"))
    found = await paused(bridge)
    events = await collect(bridge, run_input("r-2", resume=[approve(found.id)]), refund_agent)
    assert shop.refunds == []
    assert _result_error(events) == "approval-denied"
    assert events[-1].result == ["error"]


@pytest.mark.parametrize(
    "entry",
    [
        {"status": "cancelled", "payload": {"approved": True}},
        {"status": "resolved", "payload": {"approved": "true"}},
        {"status": "resolved", "payload": {"approved": 1}},
        {"status": "resolved", "payload": {"approved": False}},
        {"status": "resolved", "payload": ["approved"]},
        {"status": "resolved"},
    ],
)
async def test_anything_but_approved_true_denies_without_asking(entry):
    shop = Shop()
    port = RecordingApproval(AutoApprove())
    bridge = shop.bridge(port)
    found = await paused(bridge)
    events = await collect(
        bridge, run_input("r-2", resume=[{"interruptId": found.id, **entry}]), refund_agent
    )
    assert shop.refunds == [] and port.requests == []
    assert _result_error(events) == "approval-denied"


async def test_replayed_resume_is_ignored():
    shop = Shop()
    bridge = shop.bridge(AutoApprove())
    found = await paused(bridge)
    resume = [approve(found.id)]
    await collect(bridge, run_input("r-2", resume=resume), refund_agent)
    events = await collect(bridge, run_input("r-3", resume=resume), refund_agent)
    assert shop.refunds == [("o-7", 50.0)]
    ignored = [e for e in events if e.type.value == "CUSTOM" and e.name.endswith("resume_ignored")]
    assert [e.value for e in ignored] == [{"interruptId": found.id}]
    assert events[-1].result == []


@pytest.mark.parametrize(("thread", "principal"), [("t-other", "ana"), ("t-1", "mallory")])
async def test_cross_run_resume_is_ignored_and_leaves_the_interrupt_open(thread, principal):
    shop = Shop()
    bridge = shop.bridge(AutoApprove())
    found = await paused(bridge, principal="ana")
    body = run_input("r-2", thread=thread, resume=[approve(found.id)])
    await collect(bridge, body, refund_agent, principal=principal)
    assert shop.refunds == []
    await collect(bridge, run_input("r-3", resume=[approve(found.id)]), refund_agent, "ana")
    assert shop.refunds == [("o-7", 50.0)]  # the owner can still answer it


async def test_expired_interrupt_cannot_be_answered():
    now = [0.0]
    shop = Shop()
    bridge = shop.bridge(AutoApprove(), interrupts=InterruptTable(ttl=10, clock=lambda: now[0]))
    found = await paused(bridge)
    now[0] = 11.0
    await collect(bridge, run_input("r-2", resume=[approve(found.id)]), refund_agent)
    assert shop.refunds == []


async def test_without_a_port_a_resume_denies():
    shop = Shop()
    bridge = shop.bridge()
    bridge._resumer = Resumer(shop.registry, None, bridge.interrupts, TokenTable())
    found = await paused(bridge)
    events = await collect(bridge, run_input("r-2", resume=[approve(found.id)]), refund_agent)
    assert shop.refunds == [] and _result_error(events) == "approval-denied"


async def test_default_port_is_the_executors_and_a_parked_call_stays_pending():
    shop = Shop()
    bridge = shop.bridge()  # DeferredApproval: nobody decided the store record
    found = await paused(bridge)
    events = await collect(bridge, run_input("r-2", resume=[approve(found.id)]), refund_agent)
    assert shop.refunds == []
    (again,) = interrupts(events)
    assert again.tool_call_id == "c-refund" and again.id != found.id


async def test_a_decision_in_the_store_lets_the_default_port_approve():
    shop = Shop()
    bridge = shop.bridge()
    found = await paused(bridge)
    (record,) = await shop.store.list(status="pending")
    await shop.store.decide(record.approval_id, ApprovalDecision(approved=True, decided_by="ops"))
    await collect(bridge, run_input("r-2", resume=[approve(found.id)]), refund_agent)
    assert shop.refunds == [("o-7", 50.0)]


async def test_port_edits_apply_and_remote_edits_do_not():
    def edit(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, edited_args={"order_id": "o-7", "amount": 10.0})

    shop = Shop()
    bridge = shop.bridge(CallbackApproval(edit))
    found = await paused(bridge)
    payload = {"approved": True, "editedArgs": {"order_id": "o-7", "amount": 9999.0}}
    await collect(bridge, run_input("r-2", resume=[approve(found.id, payload)]), refund_agent)
    assert shop.refunds == [("o-7", 10.0)]


async def test_port_still_pending_reopens_the_interrupt():
    shop = Shop()
    port = ScriptedApproval([ApprovalPending("apr_x", tool="refund")])
    bridge = shop.bridge(port)
    found = await paused(bridge)
    events = await collect(bridge, run_input("r-2", resume=[approve(found.id)]), refund_agent)
    (again,) = interrupts(events)
    assert shop.refunds == [] and again.metadata["agent_adapter"]["approval_id"] == "apr_x"


async def test_token_is_bound_to_the_call_and_used_once():
    shop = Shop()
    bridge = shop.bridge(AutoApprove())
    found = await paused(bridge)
    await collect(bridge, run_input("r-2", resume=[approve(found.id)]), refund_agent)
    assert len(bridge._resumer.tokens) == 0  # taken by the call it was issued for


async def test_tokens_for_other_arguments_calls_or_runs_do_not_approve():
    shop = Shop()
    bridge = shop.bridge(AutoDeny())
    found = await paused(bridge)
    (record,) = await shop.store.list(status="pending")
    run_id, tokens, yes = (
        record.request.run_id,
        bridge._resumer.tokens,
        ApprovalDecision(approved=True),
    )
    real = {"order_id": "o-7", "amount": 50.0}
    tokens.issue("refund", {**real, "amount": 1.0}, yes, call_id="c-refund", run_id=run_id)
    tokens.issue("refund", real, yes, call_id="c-other", run_id=run_id)
    tokens.issue("refund", real, yes, call_id="c-refund", run_id="run_other")
    await collect(bridge, run_input("r-2", resume=[approve(found.id)]), refund_agent)
    assert shop.refunds == [] and len(tokens) == 3
