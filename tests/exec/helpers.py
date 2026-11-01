"""Shared builders and fakes for the executor tests."""

from __future__ import annotations

from typing import Any

from agent_adapter import ApprovalDecision, ApprovalPending, ApprovalRequest, Registry
from agent_adapter.exec import Executor
from agent_adapter.tracing.events import REQUIRED_DATA
from tests.fakes.clock import FakeClock
from tests.fakes.sinks import RecordingLedger, RecordingSink


class GraphBubbleUp(Exception):
    """Named like LangGraph's interrupt base: control flow by name."""


class DeferredFake:
    """ApprovalPort that defers each new (run, call) and returns decisions set later."""

    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []
        self.decisions: dict[tuple[str, str], ApprovalDecision] = {}

    def decide(self, run_id: str, call_id: str, decision: ApprovalDecision) -> None:
        self.decisions[(run_id, call_id)] = decision

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        decision = self.decisions.pop((request.run_id, request.call_id), None)
        if decision is None:
            raise ApprovalPending(
                "apr_pending", tool=request.tool, run_id=request.run_id, call_id=request.call_id
            )
        return decision


def make(
    registry: Registry | None = None, **options: Any
) -> tuple[Executor, RecordingSink, RecordingLedger, FakeClock]:
    """An executor wired to a recording sink and ledger and a fake clock."""
    sink, ledger, clock = RecordingSink(), RecordingLedger(), FakeClock()
    options.setdefault("sink", sink)
    options.setdefault("ledger", ledger)
    options.setdefault("clock", clock)
    options.setdefault("sleep", clock.sleep)
    options.setdefault("rng", lambda: 0.0)
    return Executor(registry, **options), sink, ledger, clock


def check_events(sink: RecordingSink) -> list[str]:
    """Event types in order, after asserting each carries its catalogue's required data."""
    for event in sink.events:
        missing = REQUIRED_DATA[event.type] - set(event.data)
        assert not missing, f"{event.type} lacks {sorted(missing)}"
    return sink.types()
