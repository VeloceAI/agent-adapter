"""Withdraw the approval records a pause leaves behind (unsealable or superseded)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import ApprovalAlreadyDecided, ApprovalNotFound, log_failure
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec._resume import Pending
from agent_adapter.flow._seal import NOT_SEALABLE
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)

SUPERSEDED = "superseded"  # a pending record the resume asks for afresh under a new call id


async def withdraw(
    flow: str, ctx: ToolContext, pendings: Sequence[Pending], reason: str = NOT_SEALABLE
) -> None:
    """Deny and consume each pending record; a record no store reaches is logged.

    Args:
        reason: The denial reason: `NOT_SEALABLE` (the pause failed; ERROR when a record
            cannot be withdrawn) or `SUPERSEDED` (the pause stands; WARNING).
    """
    for request, store in pendings:
        done = await _withdraw(flow, request.approval_id, store, reason)
        event = make_event(
            EventType.APPROVAL_DECIDED,
            ctx,
            call_id=request.call_id,
            tool=request.tool,
            approval_id=request.approval_id,
            approved=False,
            withdrawn=done,
            reason=reason,
        )
        emit(ctx.sink, event)


async def _withdraw(flow: str, approval_id: str, store: ApprovalStore | None, reason: str) -> bool:
    failed = logging.ERROR if reason == NOT_SEALABLE else logging.WARNING
    if store is not None:
        try:
            try:
                decision = ApprovalDecision(approved=False, reason=reason, decided_by="flow")
                await store.decide(approval_id, decision)
            except ApprovalAlreadyDecided:
                logger.warning("flow %s: approval %s was already decided", flow, approval_id)
            await store.consume(approval_id)  # consumed, even an approval cannot run
        except ApprovalNotFound:
            logger.debug("approval %s is not in the store", approval_id)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            log_failure(
                logger,
                failed,
                exc,
                "flow %s: approval %s could not be withdrawn from its store (%s); deny it there",
                flow,
                approval_id,
                reason,
            )
            return False
        else:
            logger.warning("flow %s: approval %s withdrawn (%s)", flow, approval_id, reason)
            return True
    logger.log(
        failed,
        "flow %s: approval %s cannot be withdrawn (no approval store reaches it; %s), so deny "
        "it at its port",
        flow,
        approval_id,
        reason,
    )
    return False
