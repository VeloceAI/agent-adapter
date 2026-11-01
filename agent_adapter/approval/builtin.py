"""Confirm-rule evaluation and the in-process approval ports."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import ApprovalPending, ToolUsageError, log_failure
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest, ToolCall

logger = logging.getLogger(__name__)

ConfirmRule = bool | Callable[[ToolCall, ToolContext], bool | Awaitable[bool]]
ApprovalCallback = Callable[
    [ApprovalRequest], ApprovalDecision | bool | Awaitable[ApprovalDecision | bool]
]


async def evaluate_confirm(rule: ConfirmRule, call: ToolCall, ctx: ToolContext) -> bool:
    """Whether `call` needs approval under a `ToolSpec.confirm` rule; the one implementation.

    A bool rule is returned as is; a callable gets `(call, ctx)` and may be sync or async.

    Raises:
        ToolUsageError: the rule is not a bool or callable, or returned a non-bool.
        Exception: whatever the rule raised: logged without the arguments, never retried.
            `ApprovalPending` and LangGraph's `GraphBubbleUp` pass through without a log.
    """
    if isinstance(rule, bool):
        return rule
    if not callable(rule):
        raise ToolUsageError(f"confirm for tool {call.name!r} must be a bool or a callable")
    try:
        answer = rule(call, ctx)
        if inspect.isawaitable(answer):
            answer = await answer
    except Exception as exc:
        if not _is_expected_signal(exc):
            # Never log the arguments: confirm tools are the sensitive ones.
            log_failure(
                logger,
                logging.ERROR,
                exc,
                "confirm rule for tool %r failed (run %s, call %s)",
                call.name,
                ctx.run_id,
                call.id,
            )
        raise
    if not isinstance(answer, bool):
        raise ToolUsageError(
            f"confirm rule for tool {call.name!r} must return a bool, got {type(answer).__name__}"
        )
    return answer


class AutoApprove:
    """ApprovalPort that approves every request."""

    def __init__(self, *, reason: str = "auto") -> None:
        self.reason = reason

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(
            approved=True, reason=self.reason, decided_by="auto", decided_at=time.time()
        )


class AutoDeny:
    """ApprovalPort that denies every request."""

    def __init__(self, *, reason: str = "denied by policy") -> None:
        self.reason = reason

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(
            approved=False, reason=self.reason, decided_by="auto", decided_at=time.time()
        )


class CallbackApproval:
    """ApprovalPort over a sync or async function returning a decision or a bool."""

    def __init__(self, fn: ApprovalCallback) -> None:
        self.fn = fn

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask `fn`; a bool becomes a decision by `"callback"`.

        Raises:
            ToolUsageError: `fn` returned something other than a decision or a bool.
            ApprovalPending: `fn` deferred the decision.
        """
        try:
            answer: Any = self.fn(request)
            if inspect.isawaitable(answer):
                answer = await answer
        except Exception as exc:
            if not _is_expected_signal(exc):
                log_failure(
                    logger,
                    logging.ERROR,
                    exc,
                    "approval callback failed for tool %r (run %s, call %s)",
                    request.tool,
                    request.run_id,
                    request.call_id,
                )
            raise
        if isinstance(answer, ApprovalDecision):
            return answer
        if isinstance(answer, bool):
            return ApprovalDecision(approved=answer, decided_by="callback", decided_at=time.time())
        raise ToolUsageError(
            f"approval callback for tool {request.tool!r} must return an ApprovalDecision "
            f"or bool, got {type(answer).__name__}"
        )


def _is_expected_signal(exc: BaseException) -> bool:
    """Pending approvals and control flow (incl. registered types) are signals, not failures."""
    return isinstance(exc, ApprovalPending) or is_control_flow(exc)
