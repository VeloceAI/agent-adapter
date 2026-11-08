"""fail_closed: a lost audit record stops calls before their side effect; the default goes on."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from agent_adapter import AutoApprove, CallbackApproval, Executor, Registry
from agent_adapter.audit import AuditSink, AuditUnavailable, CallableAuditStore, audit
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.errors import ConfigError
from tests.audit.helpers import KEY
from tests.sandbox.support import recording

ran: list[str] = []


def registry() -> Registry:
    reg = Registry()

    @reg.tool(confirm=True)
    def refund(order: str) -> str:
        """Refund an order."""
        ran.append(f"refund {order}")
        return "ok"

    @reg.tool
    def lookup(order: str) -> str:
        """Look an order up."""
        ran.append(f"lookup {order}")
        return order

    return reg


class Flaky:
    """A shipper that fails from the `fail_at`-th record on."""

    def __init__(self, fail_at: int) -> None:
        self.fail_at, self.seen = fail_at, 0

    def __call__(self, record: dict[str, Any]) -> None:
        self.seen += 1
        if self.seen >= self.fail_at:
            raise RuntimeError("collector down")


@pytest.fixture(autouse=True)
def clear() -> None:
    ran.clear()


def test_a_lost_request_record_stops_the_call_before_anyone_is_asked() -> None:
    asked: list[ApprovalRequest] = []

    def reviewer(request: ApprovalRequest) -> bool:
        asked.append(request)
        return True

    executor = Executor(registry(), approval=CallbackApproval(reviewer))
    sink = audit(executor, store=CallableAuditStore(Flaky(1), key=KEY), fail_closed=True)
    with pytest.raises(AuditUnavailable) as caught:
        asyncio.run(executor.run("refund", {"order": "A-1"}))
    assert caught.value.kind == "audit-unavailable" and not caught.value.retryable
    assert asked == [] and ran == []
    with pytest.raises(AuditUnavailable):  # every later call too, approval or not
        asyncio.run(executor.run("lookup", {"order": "A-2"}))
    assert ran == []
    sink.stop()
    assert asyncio.run(executor.run("lookup", {"order": "A-3"})) == "A-3"  # detached


def test_a_lost_decision_record_stops_the_approved_call() -> None:
    executor = Executor(registry(), approval=AutoApprove())
    audit(executor, store=CallableAuditStore(Flaky(3), key=KEY), fail_closed=True)  # open, req
    with pytest.raises(AuditUnavailable):
        asyncio.run(executor.run("refund", {"order": "A-1"}))
    assert ran == []


def test_the_default_logs_and_lets_calls_go_on() -> None:
    executor = Executor(registry(), approval=AutoApprove())
    sink = audit(executor, store=CallableAuditStore(Flaky(1), key=KEY))
    assert asyncio.run(executor.run("refund", {"order": "A-1"})) == "ok"
    assert ran == ["refund A-1"] and sink.failures > 0 and not sink.fail_closed
    sink.require()  # never raises when not fail-closed


def test_a_sandbox_does_not_run_when_its_start_is_not_recorded() -> None:
    sandbox, service = recording()
    executor = Executor(registry())
    sink = AuditSink(CallableAuditStore(Flaky(1), key=KEY), fail_closed=True)
    executor.add_sink(sink)
    ctx = executor.context(run_id="run_box")
    with pytest.raises(AuditUnavailable):
        asyncio.run(sandbox.run("print(1)", ctx=ctx))
    assert service.requests == []


def test_a_healthy_fail_closed_log_changes_nothing(tmp_path: Path) -> None:
    executor = Executor(registry(), approval=AutoApprove())
    sink = audit(executor, folder=tmp_path, key=KEY, fsync=False, fail_closed=True)
    assert asyncio.run(executor.run("refund", {"order": "A-1"})) == "ok"
    sink.require()
    with pytest.raises(ConfigError, match="fail_closed"):
        audit(executor, folder=tmp_path, key=KEY)  # same folder, other mode
    sink.stop()


def test_edited_arguments_are_flagged_and_still_audited(tmp_path: Path) -> None:
    def edit(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, edited_args={"order": "A-9"})

    sent: list[dict[str, Any]] = []
    executor = Executor(registry(), approval=CallbackApproval(edit))
    sink = audit(executor, store=CallableAuditStore(sent.append, key=KEY))
    asyncio.run(executor.run("refund", {"order": "A-1"}))
    sink.stop()
    decided = [r for r in sent if r.get("action") == "approval.decided"]
    assert decided[0]["data"]["edited"] is True and ran == ["refund A-9"]
    assert "A-9" not in repr(sent)
