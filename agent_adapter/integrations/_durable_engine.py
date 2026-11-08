"""What engines share around `DurableFlows`: input checks, retry rules, safe error text."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from agent_adapter.core.errors import AdapterError, CheckpointConflict, ToolUsageError
from agent_adapter.integrations._durable_flows import (
    ApprovalNotice,
    DurableFlows,
    FlowStepReport,
    RunIdentity,
)


def retryable(exc: BaseException) -> bool:
    """Whether an engine should retry a failed step: store and transport blips, never logic."""
    if isinstance(exc, CheckpointConflict):
        return True  # another attempt holds the run; the next one continues it
    return isinstance(exc, Exception) and not isinstance(exc, AdapterError)


def failure_message(exc: BaseException) -> str:
    """A step failure as engine history may keep it: the error type and kind, never its text."""
    kind = getattr(exc, "kind", "") if isinstance(exc, AdapterError) else ""
    return f"{type(exc).__name__} ({kind})" if kind else type(exc).__name__


def refusal(pending: Mapping[str, Any] | None, queued: object, notice: object) -> str | None:
    """Why a notice cannot wake the run now, or None: one valid notice per pending approval."""
    if pending is None:
        return "the run is not waiting for an approval"
    if queued is not None:
        return "a notice for this approval was already received"
    try:
        found = ApprovalNotice.model_validate(notice)
    except ValidationError:
        return "not a valid notice (approval_id only)"
    if found.approval_id != pending.get("approval_id"):
        return "the notice is for another approval"
    return None


async def advanced(
    flows: DurableFlows, flow: str, run_id: str, state: object, identity: object = None
) -> FlowStepReport:
    """Advance with the engine's input checked first; bad input is a usage error, not a retry."""
    if state is not None and not isinstance(state, Mapping):
        raise ToolUsageError("state must be a mapping")
    return await flows.advance(flow, run_id, state, RunIdentity.of(identity))


async def resumed(
    flows: DurableFlows, flow: str, run_id: str, request: Mapping[str, Any]
) -> FlowStepReport:
    """Resume from engine input checked first; bad input is a usage error, not a retry."""
    try:
        notice = ApprovalNotice.model_validate(request.get("notice"))
    except ValidationError:
        raise ToolUsageError("the notice is not a valid ApprovalNotice") from None
    timed_out = request.get("timed_out") is True
    identity = RunIdentity.of(request.get("identity"))
    return await flows.resume(
        flow, run_id, notice.approval_id, timed_out=timed_out, identity=identity
    )


__all__ = ["advanced", "failure_message", "refusal", "resumed", "retryable"]
