"""DBOS: a `Flow` as a `@DBOS.workflow`, one superstep per `@DBOS.step`; approvals from your store.

A reviewer decides in the approval store; a DBOS message only wakes the workflow to
re-check it.

Each step runs a superstep through the `DurableFlows` runner, so the executor's approvals,
hooks, budgets, cache, ledger and events apply. DBOS records each finished step; a step cut
short by a crash runs again, and the runner's checkpoint store replays its finished tool
calls instead of calling them twice.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import FlowError, MissingDependency, ToolUsageError
from agent_adapter.integrations._durable_engine import (
    advanced,
    failure_message,
    refusal,
    resumed,
    retryable,
)
from agent_adapter.integrations._durable_flows import (
    ApprovalNotice,
    DurableFlows,
    FlowStepReport,
    PendingApproval,
    RunIdentity,
)

if TYPE_CHECKING:
    from dbos import WorkflowHandleAsync

logger = logging.getLogger(__name__)

WORKFLOW = "agent_adapter.flow"
ADVANCE = "agent_adapter.flow.advance"
RESUME = "agent_adapter.flow.resume"
NOTIFY_TOPIC = "agent_adapter.notify"
PENDING_EVENT = "agent_adapter.pending"
_WAIT_S = 3600.0  # one wait for a notice when there is no timeout; it then waits again


class StepRetry(Exception):
    """A step failed on something passing (a store or network blip): DBOS runs it again."""


def _require() -> Any:
    try:
        from dbos import DBOS
    except ImportError as exc:
        raise MissingDependency('DBOS needs dbos: pip install "agent-adapter[dbos]"') from exc
    return DBOS


def _should_retry(exc: BaseException) -> bool:
    return isinstance(exc, StepRetry)


async def _step(flow: str, work: Any) -> dict[str, Any]:
    """One step; a failure keeps only its type and kind, never its text."""
    try:
        report: FlowStepReport = await work
    except Exception as exc:
        again = retryable(exc)
        logger.warning("durable flow step failed (%s, retry=%s)", type(exc).__name__, again)
        if again:
            raise StepRetry(failure_message(exc)) from None
        raise FlowError(f"flow step failed: {failure_message(exc)}", flow=flow) from None
    return report.model_dump(mode="json")


class DbosFlows:
    """The DBOS workflow and steps for `flows`; build it once per process, before `DBOS.launch()`.

    Whoever can send messages to the workflow can approve: keep DBOS's system database
    (and its admin access) private.
    """

    def __init__(
        self, flows: DurableFlows, *, max_attempts: int = 5, interval_s: float = 1.0
    ) -> None:
        """Register the workflow and its two steps with DBOS.

        Raises:
        MissingDependency: dbos is not installed.
        ToolUsageError: `flows` is not a `DurableFlows`, or a bad retry setting.
        """
        dbos = _require()
        if not isinstance(flows, DurableFlows):
            raise ToolUsageError(f"flows must be a DurableFlows, got {type(flows).__name__}")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ToolUsageError("max_attempts must be an int >= 1")
        if (
            isinstance(interval_s, bool)
            or not isinstance(interval_s, int | float)
            or interval_s <= 0
        ):
            raise ToolUsageError("interval_s must be a number > 0")
        self.flows = flows
        retry = {
            "retries_allowed": True,
            "max_attempts": max_attempts,
            "interval_seconds": float(interval_s),
            "should_retry": _should_retry,
        }

        async def advance_step(request: dict[str, Any]) -> dict[str, Any]:
            flow, run_id, who = (
                str(request["flow"]),
                str(request["run_id"]),
                request.get("identity"),
            )
            return await _step(flow, advanced(flows, flow, run_id, request.get("state"), who))

        async def resume_step(request: dict[str, Any]) -> dict[str, Any]:
            flow, run_id = str(request["flow"]), str(request["run_id"])
            return await _step(flow, resumed(flows, flow, run_id, request))

        advance = dbos.step(name=ADVANCE, **retry)(advance_step)
        resume = dbos.step(name=RESUME, **retry)(resume_step)

        async def run_flow(request: dict[str, Any]) -> dict[str, Any]:
            return await _drive(dbos, advance, resume, request)

        self._workflow = dbos.workflow(name=WORKFLOW)(run_flow)

    async def start(
        self,
        flow: str,
        state: Mapping[str, Any],
        *,
        run_id: str,
        approval_timeout_s: float | None = None,
        identity: RunIdentity | None = None,
    ) -> WorkflowHandleAsync[dict[str, Any]]:
        """Start a flow run as a workflow whose id is the run id; starting it again is a no-op.

        `identity` (user, thread, principal) is sealed into the run; every step acts for it.

        Raises:
            ToolUsageError: a bad flow name, state, run id, timeout or identity.
        """
        from dbos import DBOS, SetWorkflowID

        for label, text in (("flow", flow), ("run_id", run_id)):
            if not isinstance(text, str) or not text:
                raise ToolUsageError(f"{label} must be a non-empty str")
        if not isinstance(state, Mapping):
            raise ToolUsageError("state must be a mapping")
        timeout = approval_timeout_s
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0
        ):
            raise ToolUsageError("approval_timeout_s must be a number > 0 or None")
        self.flows.flow(flow)  # an unknown flow fails here, not in the workflow
        if identity is not None and not isinstance(identity, RunIdentity):
            raise ToolUsageError(f"identity must be a RunIdentity, got {type(identity).__name__}")
        who = identity.model_dump() if identity is not None else None
        request = {
            "flow": flow,
            "state": dict(state),
            "approval_timeout_s": timeout,
            "identity": who,
        }
        with SetWorkflowID(run_id):
            handle: WorkflowHandleAsync[dict[str, Any]] = await DBOS.start_workflow_async(
                self._workflow, request
            )
        return handle

    async def notify(self, run_id: str, approval_id: str) -> None:
        """Wake a paused run after you decided `approval_id` in the approval store.

        The message carries no verdict: the run re-checks the store and goes on only with a
        decided record there, once.

        Raises:
            ToolUsageError: an empty run id or approval id.
        """
        from dbos import DBOS

        for label, text in (("run_id", run_id), ("approval_id", approval_id)):
            if not isinstance(text, str) or not text:
                raise ToolUsageError(f"{label} must be a non-empty str")
        notice = ApprovalNotice(approval_id=approval_id).model_dump(mode="json")
        await DBOS.send_async(run_id, notice, topic=NOTIFY_TOPIC)

    async def pending(self, run_id: str, *, timeout_s: float = 0.0) -> PendingApproval | None:
        """The approval the run waits for (no arguments), or None."""
        from dbos import DBOS

        raw = await DBOS.get_event_async(run_id, PENDING_EVENT, timeout_seconds=timeout_s)
        return None if raw is None else PendingApproval.model_validate(raw)

    @staticmethod
    async def result(handle: WorkflowHandleAsync[dict[str, Any]]) -> FlowStepReport:
        """Wait for the run to end and return its report (done or denied)."""
        return FlowStepReport.model_validate(await handle.get_result())


async def _drive(dbos: Any, advance: Any, resume: Any, request: dict[str, Any]) -> dict[str, Any]:
    """The workflow body: superstep by superstep; at a pause, wait until the store decides."""
    flow = request.get("flow") if isinstance(request, dict) else None
    if not isinstance(flow, str) or not flow:
        raise ToolUsageError("bad workflow input")
    base = {"flow": flow, "run_id": str(dbos.workflow_id), "identity": request.get("identity")}
    report = await advance({**base, "state": request.get("state")})
    while report["status"] in ("running", "paused"):
        if report["status"] == "running":
            report = await advance(base)
            continue
        pending = report["pending"]
        await dbos.set_event_async(PENDING_EVENT, pending)
        timed_out = await _wait(dbos, pending, request.get("approval_timeout_s"))
        await dbos.set_event_async(PENDING_EVENT, None)
        notice = {"approval_id": pending["approval_id"]}
        report = await resume({**base, "notice": notice, "timed_out": timed_out})
    if report["status"] == "failed":
        raise FlowError("the flow run failed", flow=flow)
    return dict(report)


async def _wait(dbos: Any, pending: Mapping[str, Any], timeout: float | None) -> bool:
    """Wait for a notice for `pending`; True when the wait timed out instead."""
    while True:
        message = await dbos.recv_async(NOTIFY_TOPIC, timeout_seconds=timeout or _WAIT_S)
        if message is None:
            if timeout:
                logger.info("approval %s timed out", pending["approval_id"])
                return True
            continue
        problem = refusal(pending, None, message)
        if problem is None:
            return False
        logger.warning("notice message dropped: %s", problem)


__all__ = [
    "ADVANCE",
    "NOTIFY_TOPIC",
    "PENDING_EVENT",
    "RESUME",
    "WORKFLOW",
    "ApprovalNotice",
    "DbosFlows",
    "DurableFlows",
    "FlowStepReport",
    "PendingApproval",
    "RunIdentity",
    "StepRetry",
]
