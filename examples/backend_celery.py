"""Celery: tasks call tools through run_sync; an approval parks between two tasks."""

import functools
from typing import Any

from backend_common import Backend, Reply, build_backend, call, record_decision

from agent_adapter import ApprovalPending, MemoryApprovalStore, run_sync
from agent_adapter.errors import ApprovalAlreadyDecided, ToolTimeout
from agent_adapter.runtime import new_id

TIMEOUT_S = 60
# Stand-in: the web process and every worker must share one store (a database ApprovalStore).
SHARED_STORE = MemoryApprovalStore()


# --8<-- [start:backend]
@functools.cache
def backend() -> Backend:
    """One per worker process, built on first use (after the prefork fork)."""
    return build_backend(store=SHARED_STORE)


# --8<-- [end:backend]


# --8<-- [start:tasks]
def run_tool(tool: str, arguments: dict[str, Any], run_id: str, call_id: str) -> dict[str, Any]:
    """Task body: one tool call; the sender picks the ids, so a retry is the same call."""
    work = call(backend(), tool, arguments, run_id=run_id, call_id=call_id)
    try:
        value = run_sync(work, timeout=TIMEOUT_S)
    except ApprovalPending as pending:  # parked: a later task resumes it
        return {"status": "pending_approval", "approval_id": pending.approval_id}
    return {"status": "ok", "result": value}  # other errors raise: Celery records the failure


def resume_tool(approval_id: str) -> dict[str, Any]:
    """Task body: run a decided call once with its original ids; never auto-retried."""
    record = run_sync(backend().store.get(approval_id), timeout=TIMEOUT_S)
    if record.consumed:  # a duplicate delivery: the approval was already used
        raise ApprovalAlreadyDecided(approval_id, status=record.status)
    request = record.request
    return run_tool(request.tool, request.arguments, request.run_id, request.call_id)


def make_celery(broker: str = "memory://", result_backend: str = "cache+memory://") -> Any:
    """The Celery app with both tasks registered (needs `celery`)."""
    from celery import Celery

    app = Celery("tools", broker=broker, backend=result_backend)
    app.conf.task_acks_late = True  # a crashed worker's task runs again: make tools idempotent
    retry = {"autoretry_for": (ToolTimeout,), "retry_backoff": True, "max_retries": 3}
    app.task(name="tools.run", **retry)(run_tool)
    app.task(name="tools.resume")(resume_tool)
    return app


def enqueue(app: Any, tool: str, arguments: dict[str, Any]) -> Any:
    """Send `tools.run` with fresh run and call ids; returns the Celery `AsyncResult`."""
    return app.tasks["tools.run"].delay(tool, arguments, new_id("run_"), new_id("call_"))


# --8<-- [end:tasks]


# --8<-- [start:review]
def review(app: Any, approval_id: str, *, reviewer: str, approved: bool) -> tuple[Reply, Any]:
    """In the web process: store the decision, then queue `tools.resume` for an approval."""
    reply = run_sync(
        record_decision(backend(), approval_id, reviewer=reviewer, approved=approved),
        timeout=TIMEOUT_S,
    )
    queued = None
    if reply[0] == 200 and approved:
        queued = app.tasks["tools.resume"].delay(approval_id)
    return reply, queued


# --8<-- [end:review]
