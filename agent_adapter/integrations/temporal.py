"""Temporal: a `Flow` as a workflow, one superstep per activity; approvals from your store.

A reviewer decides in the approval store; an update or signal only wakes the workflow to
re-check it. The workflow holds no I/O: it calls two activities by name and waits. The
activities run each superstep through the `DurableFlows` runner, so the executor's
approvals, hooks, budgets, cache, ledger and events apply, and its checkpoint store lets a
retried activity continue without re-running a finished tool call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import timedelta
from functools import cache
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import MissingDependency, ToolUsageError
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
    from temporalio.client import Client, WorkflowHandle
    from temporalio.worker import Worker

logger = logging.getLogger(__name__)

WORKFLOW = "agent_adapter.flow"
ADVANCE = "agent_adapter.flow.advance"
RESUME = "agent_adapter.flow.resume"
NOTIFY = "notify"  # the update (waits for acceptance) and the signal share this name
PENDING = "pending"  # the query: the approval the run waits for, or None


def _require() -> None:
    try:
        import temporalio  # noqa: F401
    except ImportError as exc:
        raise MissingDependency(
            'Temporal needs temporalio: pip install "agent-adapter[temporal]"'
        ) from exc


def flow_activities(flows: DurableFlows) -> list[Callable[..., Any]]:
    """The two activities a worker registers: advance a run, and resume it from the store.

    Raises:
        MissingDependency: temporalio is not installed.
        ToolUsageError: `flows` is not a `DurableFlows`.
    """
    _require()
    from temporalio import activity

    if not isinstance(flows, DurableFlows):
        raise ToolUsageError(f"flows must be a DurableFlows, got {type(flows).__name__}")

    async def advance(request: dict[str, Any]) -> dict[str, Any]:
        run_id, flow = str(activity.info().workflow_id or ""), str(request.get("flow"))
        who = request.get("identity")
        return await _step(advanced(flows, flow, run_id, request.get("state"), who))

    async def resume(request: dict[str, Any]) -> dict[str, Any]:
        run_id, flow = str(activity.info().workflow_id or ""), str(request.get("flow"))
        return await _step(resumed(flows, flow, run_id, request))

    return [activity.defn(name=ADVANCE)(advance), activity.defn(name=RESUME)(resume)]


async def _step(work: Any) -> dict[str, Any]:
    """Run one step; an adapter error becomes a Temporal error carrying its type and kind only."""
    from temporalio.exceptions import ApplicationError

    try:
        report: FlowStepReport = await work
    except Exception as exc:
        again = retryable(exc)
        logger.warning("durable flow step failed (%s, retry=%s)", type(exc).__name__, again)
        kind = str(getattr(exc, "kind", "")) or type(exc).__name__
        raise ApplicationError(failure_message(exc), type=kind, non_retryable=not again) from None
    return report.model_dump(mode="json")


class FlowWorkflow:
    """The flow workflow: `run(request)` drives one flow run to its end (no I/O here)."""

    def __init__(self) -> None:
        self._pending: dict[str, Any] | None = None
        self._notice: dict[str, Any] | None = None
        self._early: dict[str, Any] | None = None  # a signal that came before its pause

    async def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """Advance superstep by superstep; at a pause, wait for notices until the store decides."""
        from temporalio import workflow
        from temporalio.common import RetryPolicy
        from temporalio.exceptions import ApplicationError

        flow = request.get("flow") if isinstance(request, dict) else None
        if not isinstance(flow, str) or not flow:
            raise ApplicationError(
                "bad workflow input", type="tool-usage-error", non_retryable=True
            )
        opts = _activity_options(request, RetryPolicy)
        base = {"flow": flow, "identity": request.get("identity")}
        report = await workflow.execute_activity(
            ADVANCE, {**base, "state": request.get("state")}, **opts
        )
        while report["status"] in ("running", "paused"):
            if report["status"] == "running":
                report = await workflow.execute_activity(ADVANCE, base, **opts)
                continue
            pending = report["pending"]
            timed_out = await self._wait(pending, request.get("approval_timeout_s"))
            notice = {"approval_id": pending["approval_id"]}
            payload = {**base, "notice": notice, "timed_out": timed_out}
            report = await workflow.execute_activity(RESUME, payload, **opts)
        if report["status"] == "failed":
            raise ApplicationError("the flow run failed", type="flow-error", non_retryable=True)
        return dict(report)

    async def _wait(self, pending: dict[str, Any], timeout: float | None) -> bool:
        """Wait for a notice for `pending`; True when the wait timed out instead."""
        from temporalio import workflow

        self._pending, self._notice, early = pending, None, self._early
        self._early = None
        if early is not None:
            problem = refusal(pending, None, early)
            if problem is None:
                self._notice = early
            else:
                workflow.logger.warning("early notice dropped: %s", problem)
        timed_out = False
        try:
            limit = timedelta(seconds=timeout) if timeout else None
            await workflow.wait_condition(lambda: self._notice is not None, timeout=limit)
        except TimeoutError:  # the store records a denial unless a reviewer decided first
            workflow.logger.info("approval %s timed out", pending["approval_id"])
            timed_out = True
        self._pending, self._notice = None, None
        return timed_out

    async def notify(self, notice: dict[str, Any]) -> str:
        """The update: wake the run to re-check the approval store."""
        self._notice = dict(notice)
        return str(notice["approval_id"])

    def check(self, notice: dict[str, Any]) -> None:
        """The update's validator: refuse a notice that is not for the pending approval."""
        problem = refusal(self._pending, self._notice, notice)
        if problem:
            raise ValueError(problem)

    def notify_signal(self, notice: dict[str, Any]) -> None:
        """The signal: like the update, but a refused notice is dropped and logged."""
        from temporalio import workflow

        if self._pending is None and self._early is None and isinstance(notice, dict):
            self._early = dict(notice)  # checked when the next pause starts
            return
        problem = refusal(self._pending, self._notice, notice)
        if problem:
            workflow.logger.warning("notice signal dropped: %s", problem)
            return
        self._notice = dict(notice)

    def pending(self) -> dict[str, Any] | None:
        """The query: the approval the run waits for, or None."""
        return self._pending


@cache
def flow_workflow() -> type[FlowWorkflow]:
    """A `FlowWorkflow` subclass with Temporal's decorators applied (built once).

    Unsandboxed: the sandbox re-imports workflow modules, and this one imports temporalio
    lazily. The workflow code does no I/O; activities do the work.

    Raises:
        MissingDependency: temporalio is not installed.
    """
    _require()
    from temporalio import workflow

    plain = FlowWorkflow.__dict__
    update = workflow.update(name=NOTIFY)(plain["notify"])
    update.validator(plain["check"])
    body = {
        "__module__": __name__,
        "__qualname__": FlowWorkflow.__qualname__,  # Temporal matches method qualnames to it
        "run": workflow.run(plain["run"]),
        "notify": update,
        "notify_signal": workflow.signal(name=NOTIFY)(plain["notify_signal"]),
        "pending": workflow.query(name=PENDING)(plain["pending"]),
    }
    defined = type(FlowWorkflow.__name__, (FlowWorkflow,), body)
    workflow.defn(name=WORKFLOW, sandboxed=False)(defined)  # marks the class itself
    return defined


def _activity_options(request: Mapping[str, Any], retry_policy: Any) -> dict[str, Any]:
    timeout = float(request.get("activity_timeout_s") or 600.0)
    return {
        "start_to_close_timeout": timedelta(seconds=timeout),
        "retry_policy": retry_policy(maximum_attempts=int(request.get("max_attempts") or 0)),
    }


def flow_worker(client: Client, flows: DurableFlows, *, task_queue: str, **options: Any) -> Worker:
    """A Temporal `Worker` for the flow workflow and its activities; `options` go to `Worker`.

    Raises:
        MissingDependency: temporalio is not installed.
        ToolUsageError: `flows` is not a `DurableFlows`, or `task_queue` is empty.
    """
    _require()
    from temporalio.worker import Worker

    if not isinstance(task_queue, str) or not task_queue:
        raise ToolUsageError("task_queue must be a non-empty str")
    activities = flow_activities(flows)
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[flow_workflow()],
        activities=activities,
        **options,
    )


async def start_flow(
    client: Client,
    flow: str,
    state: Mapping[str, Any],
    *,
    run_id: str,
    task_queue: str,
    approval_timeout_s: float | None = None,
    activity_timeout_s: float = 600.0,
    max_attempts: int = 0,
    identity: RunIdentity | None = None,
) -> WorkflowHandle[Any, Any]:
    """Start a flow run as a workflow whose id is the run id; `max_attempts=0` retries forever.

    `identity` (user, thread, principal) is sealed into the run; every step acts for it.

    Raises:
        ToolUsageError: a bad flow name, state, run id, queue, timeout or identity.
    """
    _require()
    request = _request(flow, state, run_id, task_queue, approval_timeout_s, activity_timeout_s)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
        raise ToolUsageError("max_attempts must be an int >= 0")
    request["max_attempts"] = max_attempts
    if identity is not None and not isinstance(identity, RunIdentity):
        raise ToolUsageError(f"identity must be a RunIdentity, got {type(identity).__name__}")
    request["identity"] = identity.model_dump() if identity is not None else None
    return await client.start_workflow(WORKFLOW, request, id=run_id, task_queue=task_queue)


def _request(
    flow: object,
    state: object,
    run_id: object,
    task_queue: object,
    approval_timeout_s: object,
    activity_timeout_s: object,
) -> dict[str, Any]:
    for label, text in (("flow", flow), ("run_id", run_id), ("task_queue", task_queue)):
        if not isinstance(text, str) or not text:
            raise ToolUsageError(f"{label} must be a non-empty str")
    if not isinstance(state, Mapping):
        raise ToolUsageError("state must be a mapping (it travels as JSON)")
    for label, value, optional in (
        ("approval_timeout_s", approval_timeout_s, True),
        ("activity_timeout_s", activity_timeout_s, False),
    ):
        if value is None and optional:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise ToolUsageError(f"{label} must be a number > 0")
    return {
        "flow": flow,
        "state": dict(state),
        "approval_timeout_s": approval_timeout_s,
        "activity_timeout_s": activity_timeout_s,
    }


async def pending(handle: WorkflowHandle[Any, Any]) -> PendingApproval | None:
    """The approval the run waits for (no arguments), or None."""
    raw = await handle.query(PENDING)
    return None if raw is None else PendingApproval.model_validate(raw)


async def notify(handle: WorkflowHandle[Any, Any], approval_id: str, *, wait: bool = True) -> None:
    """Wake a paused run after you decided `approval_id` in the approval store.

    The notice carries no verdict: the run re-checks the store and goes on only with a
    decided record there, once. `wait=True` is an update the workflow refuses unless it
    waits for that approval; `wait=False` is a signal.

    Raises:
        ToolUsageError: `approval_id` is empty.
        temporalio.client.WorkflowUpdateFailedError: the workflow refused it (`wait=True`).
    """
    if not isinstance(approval_id, str) or not approval_id:
        raise ToolUsageError("approval_id must be a non-empty str")
    payload = ApprovalNotice(approval_id=approval_id).model_dump(mode="json")
    if wait:
        await handle.execute_update(NOTIFY, payload)
    else:
        await handle.signal(NOTIFY, payload)


async def result(handle: WorkflowHandle[Any, Any]) -> FlowStepReport:
    """Wait for the run to end and return its report (done or denied)."""
    return FlowStepReport.model_validate(await handle.result())


__all__ = [
    "ADVANCE",
    "NOTIFY",
    "PENDING",
    "RESUME",
    "WORKFLOW",
    "ApprovalNotice",
    "DurableFlows",
    "FlowStepReport",
    "FlowWorkflow",
    "PendingApproval",
    "RunIdentity",
    "flow_activities",
    "flow_worker",
    "flow_workflow",
    "notify",
    "pending",
    "result",
    "start_flow",
]
