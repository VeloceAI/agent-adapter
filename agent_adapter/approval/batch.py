"""Ask an approval port about a batch of calls; two phases when the port can prepare."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest

logger = logging.getLogger(__name__)

RESET_SINCE_VETTED = "approval reset since it was vetted"

# Checks each decision before any is applied (or, for a PreparedApproval, consumed).
_Vet = Callable[[ApprovalRequest, ApprovalDecision], None]


@runtime_checkable
class PreparedApproval(Protocol):
    """A port that can find-or-create a record without consuming it (`DeferredApproval`)."""

    async def prepare(self, request: ApprovalRequest) -> ApprovalRecord: ...

    async def consume(
        self, record: ApprovalRecord, request: ApprovalRequest
    ) -> ApprovalDecision: ...


async def ask_each(
    port: ApprovalPort,
    requests: Sequence[ApprovalRequest],
    *,
    vet: _Vet | None = None,
) -> list[ApprovalDecision]:
    """The port's decision for each request, in order; `vet` sees each before any is used.

    A `PreparedApproval` is asked in two phases: every record is prepared; while any is
    undecided, `ApprovalPending` is raised for the first one and nothing is consumed; then
    `vet` runs, and exactly the prepared records are consumed (a record consumed or reset
    meanwhile by someone else denies its call). Other ports get `request` per call, in order. Build
    the requests identically on every pass.

    Raises:
        ApprovalPending: the first undecided call, after every call was asked.
        Exception: whatever `vet` raises, before any decision is consumed.
    """
    if isinstance(port, PreparedApproval):
        decisions = await _two_phase(port, requests, vet)
    else:
        decisions = await _in_order(port, requests, vet)
    for request, decision in zip(requests, decisions, strict=True):
        verdict = "approved" if decision.approved else "denied"
        logger.info("%s %s (call %s)", verdict, request.tool, request.call_id)
    return decisions


async def _in_order(
    port: ApprovalPort, requests: Sequence[ApprovalRequest], vet: _Vet | None
) -> list[ApprovalDecision]:
    decisions: list[ApprovalDecision] = []
    pending: ApprovalPending | None = None
    for request in requests:
        try:
            decisions.append(await port.request(request))
        except ApprovalPending as exc:
            logger.info("approval pending for %s (call %s)", request.tool, request.call_id)
            pending = pending or exc
    if pending is not None:
        raise pending
    _vet_all(vet, requests, decisions)
    return decisions


async def _two_phase(
    port: PreparedApproval, requests: Sequence[ApprovalRequest], vet: _Vet | None
) -> list[ApprovalDecision]:
    records = [await port.prepare(request) for request in requests]
    decided: list[ApprovalDecision] = []
    for request, record in zip(requests, records, strict=True):
        if record.decision is None:
            logger.info("approval pending for %s (call %s)", request.tool, request.call_id)
            raise ApprovalPending(
                record.approval_id,
                tool=request.tool,
                run_id=request.run_id,
                call_id=request.call_id,
            )
        decided.append(record.decision)
    _vet_all(vet, requests, decided)
    return [
        await _consume(port, record, request)
        for record, request in zip(records, requests, strict=True)
    ]


async def _consume(
    port: PreparedApproval, record: ApprovalRecord, request: ApprovalRequest
) -> ApprovalDecision:
    """Consume one prepared record; anything but its vetted decision denies the call."""
    try:
        decision = await port.consume(record, request)
    except ApprovalDenied as exc:
        logger.warning(
            "approval %s for %s (call %s) was not usable: denying",
            record.approval_id,
            request.tool,
            request.call_id,
        )
        logger.debug("approval %s not usable: %s", record.approval_id, exc.reason)
        return _denied(exc.reason)
    except ApprovalPending:
        # Reset or replaced since it was vetted: stopping here would leave the batch
        # half-consumed, so this call alone is denied.
        logger.warning(
            "approval %s for %s (call %s) is pending again since it was vetted: denying",
            record.approval_id,
            request.tool,
            request.call_id,
        )
        return _denied(RESET_SINCE_VETTED)
    if decision != record.decision:
        logger.warning(
            "approval %s for %s (call %s): the consumed decision differs from the vetted one;"
            " denying",
            record.approval_id,
            request.tool,
            request.call_id,
        )
        return _denied("decision differs from the vetted one")
    return decision


def _denied(reason: str) -> ApprovalDecision:
    return ApprovalDecision(approved=False, reason=reason, decided_by="agent_adapter")


def _vet_all(
    vet: _Vet | None, requests: Sequence[ApprovalRequest], decisions: Sequence[ApprovalDecision]
) -> None:
    if vet is not None:
        for request, decision in zip(requests, decisions, strict=True):
            vet(request, decision)
