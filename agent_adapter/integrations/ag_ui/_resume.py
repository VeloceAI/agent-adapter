"""Answer resume entries: deny, or ask the port and run the call with a one-use token."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import ApprovalDenied, ApprovalPending
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import (
    ApprovalDecision,
    ApprovalRequest,
    ErrorInfo,
    ToolCall,
    ToolResult,
    to_jsonable,
)
from agent_adapter.integrations._native_gate import TokenTable, denial, plan_call, take_approval
from agent_adapter.integrations.ag_ui._events import Paused
from agent_adapter.integrations.ag_ui._interrupts import InterruptTable, PendingCall
from agent_adapter.integrations.ag_ui._support import METADATA_KEY, TARGET
from agent_adapter.integrations.ag_ui.run import AGUIRun

if TYPE_CHECKING:
    from ag_ui.core import ResumeEntry

logger = logging.getLogger(__name__)

NO_PORT = "the AG-UI bridge has no approval port"
NOT_APPROVED = "the resume entry did not approve the call"
UNKNOWN_ARGUMENTS = "the paused call's arguments are unknown"
NOT_A_TOOL = "the paused call is not a registry tool"
RESUME_IGNORED = f"{METADATA_KEY}.resume_ignored"


@dataclasses.dataclass(frozen=True)
class Resumer:
    """What resuming needs from the bridge: its registry, port, interrupts and tokens."""

    registry: Registry
    port: ApprovalPort | None
    interrupts: InterruptTable
    tokens: TokenTable

    async def resume(self, run: AGUIRun) -> None:
        """Decide and run every call the input's resume entries answer, in entry order.

        Entries for unknown, expired, answered, foreign-thread or foreign-principal
        interrupts are skipped with a warning (the AG-UI rule); nothing runs for them.
        """
        for entry in run.input.resume or ():
            pending = self.interrupts.take(
                entry.interrupt_id, thread_id=run.thread_id, principal=run.principal
            )
            if pending is None:
                logger.warning(
                    "ag_ui: resume entry for an unknown interrupt ignored (thread %s)",
                    run.thread_id,
                )
                run.custom(RESUME_IGNORED, {"interruptId": entry.interrupt_id})
                continue
            run.events.announced(pending.call_id)
            await self._answer(run, pending, entry)

    async def _answer(self, run: AGUIRun, pending: PendingCall, entry: ResumeEntry) -> None:
        try:
            decision = await self._decide(run, pending, entry)
        except ApprovalPending as exc:
            logger.info("ag_ui: %s (call %s) is still pending", pending.tool, pending.call_id)
            run.events.pause(
                Paused(
                    pending.run_id,
                    pending.call_id,
                    pending.tool,
                    pending.arguments,
                    exc.approval_id,
                )
            )
            return
        if not decision.approved or pending.arguments is None:
            denied = ApprovalDenied(pending.tool, call_id=pending.call_id, reason=decision.reason)
            logger.info("ag_ui: resumed %s denied (call %s)", pending.tool, pending.call_id)
            run.events.denied(pending.call_id, pending.tool, denied)
            run.resumed.append(_denied_result(pending, denied))
            return
        run.run_ids.add(pending.run_id)
        ctx = dataclasses.replace(run.ctx, run_id=pending.run_id, call_id=pending.call_id)
        # No token (the rule needed no approval): the executor applies its own gates.
        spec = self.registry.get(pending.tool)
        ctx.approval = take_approval(self.tokens, spec, pending.arguments, ctx)
        call = ToolCall(name=pending.tool, arguments=dict(pending.arguments), id=pending.call_id)
        run.resumed.append(await run.executor.run_call(call, ctx))

    async def _decide(
        self, run: AGUIRun, pending: PendingCall, entry: ResumeEntry
    ) -> ApprovalDecision:
        """The port's decision if the entry approves, else a denial (or ApprovalPending)."""
        if entry.status != "resolved" or not _approves(entry.payload):
            return denial(NOT_APPROVED, by=TARGET)
        if pending.arguments is None:
            return denial(UNKNOWN_ARGUMENTS, by=TARGET)
        if pending.tool not in self.registry:
            return denial(NOT_A_TOOL, by=TARGET)
        plan = await plan_call(
            self.registry, pending.tool, pending.arguments, pending.call_id, pending.run_id, TARGET
        )
        if isinstance(plan, ApprovalDecision):
            return plan  # bad arguments (denied) or the rule needs no approval
        if self.port is None:
            logger.warning("ag_ui: %s approved remotely but %s; denied", pending.tool, NO_PORT)
            return denial(NO_PORT, by=TARGET)
        request = plan.model_copy(update={"metadata": _metadata(run, pending, entry)})
        decision = await self.port.request(request)
        if decision.approved:
            self._grant(request, decision)
        return decision

    def _grant(self, request: ApprovalRequest, decision: ApprovalDecision) -> None:
        self.tokens.issue(
            request.tool,
            request.arguments,
            decision,
            call_id=request.call_id,
            run_id=request.run_id,
        )


def _approves(payload: Any) -> bool:
    """Only a JSON object whose `approved` is exactly `true` asks for approval."""
    return isinstance(payload, dict) and payload.get("approved") is True


def _metadata(run: AGUIRun, pending: PendingCall, entry: ResumeEntry) -> dict[str, Any]:
    """What the port sees about the resume: who answered, from where, and what they sent."""
    return {
        "integration": TARGET,
        "principal": run.principal,
        "thread_id": run.thread_id,
        "interrupt_id": pending.interrupt_id,
        "resume_payload": to_jsonable(entry.payload),
    }


def _denied_result(pending: PendingCall, denied: ApprovalDenied) -> ToolResult:
    return ToolResult(
        call_id=pending.call_id,
        name=pending.tool,
        status="error",
        error=ErrorInfo.from_exception(denied),
        exception=denied,
    )
