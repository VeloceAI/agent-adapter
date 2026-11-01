"""ApprovalPort fakes: scripted decisions and a recording wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest


class ScriptedApproval:
    """Returns (or raises) the given decisions in order; records every request."""

    def __init__(self, decisions: Sequence[ApprovalDecision | Exception]) -> None:
        self._decisions = list(decisions)
        self.requests: list[ApprovalRequest] = []

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        if not self._decisions:
            raise AssertionError(f"no scripted decision left for {request.tool!r}")
        decision = self._decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision


class RecordingApproval:
    """Wraps a port and records each request (and the decision it returned).

    Wrapping a `PreparedApproval` (a port with `prepare` / `consume`, e.g. `DeferredApproval`)
    gives a `PreparedApproval` too: `prepare` records the request, `consume` the decision.
    """

    def __new__(cls, port: ApprovalPort) -> RecordingApproval:
        prepared = hasattr(port, "prepare") and hasattr(port, "consume")
        if cls is RecordingApproval and prepared:
            return super().__new__(_RecordingPreparedApproval)
        return super().__new__(cls)

    def __init__(self, port: ApprovalPort) -> None:
        self.port = port
        self.requests: list[ApprovalRequest] = []
        self.decisions: list[ApprovalDecision] = []

    @property
    def store(self) -> Any:
        """The wrapped port's approval store, if it has one."""
        return getattr(self.port, "store", None)

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        decision = await self.port.request(request)
        self.decisions.append(decision)
        return decision


class _RecordingPreparedApproval(RecordingApproval):
    """`RecordingApproval` over a port that can prepare and consume (two-phase batches)."""

    async def prepare(self, request: ApprovalRequest) -> Any:
        self.requests.append(request)
        return await self.port.prepare(request)  # type: ignore[attr-defined]

    async def consume(self, record: Any, request: ApprovalRequest) -> ApprovalDecision:
        decision: ApprovalDecision = await self.port.consume(record, request)  # type: ignore[attr-defined]
        self.decisions.append(decision)
        return decision
