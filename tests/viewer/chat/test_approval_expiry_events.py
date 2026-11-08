"""A chat approval nobody decides sends `approval.expired` on the waiting call's sink."""

from __future__ import annotations

import asyncio

from agent_adapter.core.context import ToolContext
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.runtime.current import use_context
from agent_adapter.tracing.sinks import EventLog
from agent_adapter.viewer.chat.approvals import ChatApprovals
from agent_adapter.viewer.chat.live import LiveTurn
from agent_adapter.viewer.chat.watch import TurnWatch


def desk(timeout: float) -> tuple[ChatApprovals, LiveTurn]:
    watch = TurnWatch()
    live = LiveTurn("turn_1", "ses_1", "run_1")
    watch.add(live)
    return ChatApprovals(watch, reviewer="local", timeout=timeout), live


def request(approval_id: str = "apr_1") -> ApprovalRequest:
    return ApprovalRequest(approval_id=approval_id, run_id="run_1", call_id="c1", tool="send")


async def test_a_timed_out_approval_is_announced() -> None:
    log = EventLog()
    approvals, _ = desk(0.05)
    with use_context(ToolContext(run_id="run_1", call_id="c1", sink=log)):
        decision = await approvals.ask(request())
    (event,) = log.of_type("approval.expired")
    assert not decision.approved
    assert dict(event.data) == {"tool": "send", "approval_id": "apr_1", "outcome": "timed-out"}


async def test_a_stopped_turn_announces_its_pending_approval() -> None:
    log = EventLog()
    approvals, live = desk(5)
    with use_context(ToolContext(run_id="run_1", call_id="c1", sink=log)):
        waiting = asyncio.ensure_future(approvals.ask(request()))
        await asyncio.sleep(0.02)
        approvals.close_turn(live, "stopped")
        await waiting
    assert [e.data["outcome"] for e in log.of_type("approval.expired")] == ["stopped"]


async def test_a_decided_approval_or_no_context_sends_nothing() -> None:
    log = EventLog()
    approvals, _ = desk(5)
    with use_context(ToolContext(run_id="run_1", call_id="c1", sink=log)):
        waiting = asyncio.ensure_future(approvals.ask(request()))
        await asyncio.sleep(0.02)
        approvals.decide("apr_1", approved=True)
        assert (await waiting).approved
    assert log.of_type("approval.expired") == []
    quiet, _ = desk(0.01)
    assert not (await quiet.ask(request("apr_2"))).approved  # no context: no event, no error
