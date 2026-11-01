"""Task records of the MCP tasks extension: unguessable ids, bound to one caller."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from agent_adapter.integrations.mcp.annotations import APPROVAL_ID_META
from agent_adapter.integrations.mcp.tasks import TaskOptions

if TYPE_CHECKING:
    import anyio

logger = logging.getLogger(__name__)

TaskStatus = Literal["working", "input_required", "completed", "failed", "cancelled"]
TERMINAL: frozenset[str] = frozenset({"completed", "failed", "cancelled"})
TASK_ID_BYTES = 32  # 256 bits: the spec asks for ids no third party can guess
JSONRPC_INTERNAL_ERROR = -32603
WORKER_GONE = "the worker running this task stopped"


def _iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, UTC).isoformat().replace("+00:00", "Z")


class TaskRecord:
    """One task: its status, outcome and owner; a terminal status never changes."""

    def __init__(self, task_id: str, principal: str | None, tool: str, options: TaskOptions):
        import anyio

        self.task_id = task_id
        self.principal = principal
        self.tool = tool
        self.options = options
        self.status: TaskStatus = "working"
        self.status_message: str | None = None
        self.created_at = self.updated_at = time.time()
        self.deadline = time.monotonic() + options.ttl_s
        self.result: dict[str, Any] | None = None  # the CallToolResult, wire form
        self.error: dict[str, Any] | None = None  # a JSON-RPC error object
        self.input_requests: dict[str, Any] | None = None
        self.approval_id: str | None = None
        self.settled = anyio.Event()  # terminal, or waiting for someone else
        self.nudged = anyio.Event()  # a client answered an input request
        self.cancel_scope: anyio.CancelScope | None = None
        self.shared = False  # handed to the caller: mirrored to the task store
        self.save_lock = anyio.Lock()  # one store write at a time, newest state last

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def expired(self, now: float | None = None) -> bool:
        return (time.monotonic() if now is None else now) >= self.deadline

    def _move(self, status: TaskStatus, message: str | None = None) -> bool:
        if self.terminal:
            return False
        self.status, self.status_message = status, message
        self.updated_at = time.time()
        if status != "input_required":
            self.input_requests = None
        if status in TERMINAL or status == "input_required" or self.approval_id is not None:
            self.settled.set()
        return True

    def working(self, message: str | None = None) -> None:
        self._move("working", message)

    def waiting(self, approval_id: str, input_requests: dict[str, Any] | None) -> None:
        """Waiting for approval `approval_id`; `input_requests` link a person to decide it."""
        self.approval_id = approval_id
        message = f"waiting for approval {approval_id}"
        if input_requests:
            self._move("input_required", message)
            self.input_requests = input_requests
        else:
            self._move("working", message)

    def answered(self, keys: set[str]) -> None:
        """The client answered some input requests; any other key is ignored (spec)."""
        if self.status != "input_required" or not self.input_requests:
            return
        if keys & set(self.input_requests):
            self._move("working", self.status_message)
            self.nudged.set()

    def renew_nudge(self) -> None:
        """A fresh event for the next wait."""
        import anyio

        self.nudged = anyio.Event()

    def complete(self, result: dict[str, Any]) -> None:
        if self._move("completed"):
            self.result = result

    def fail(self, code: int, message: str) -> None:
        if self._move("failed", message):
            self.error = {"code": code, "message": message}

    def cancelled(self) -> None:
        self._move("cancelled", "cancelled")

    def wire(self) -> dict[str, Any]:
        """The spec `Task` (with its status-specific field) for `tasks/get` and creation."""
        task: dict[str, Any] = {
            "taskId": self.task_id,
            "status": self.status,
            "createdAt": _iso(self.created_at),
            "lastUpdatedAt": _iso(self.updated_at),
            "ttlMs": int(self.options.ttl_s * 1000),
            "pollIntervalMs": int(self.options.poll_interval_s * 1000),
        }
        if self.status_message is not None:
            task["statusMessage"] = self.status_message
        if self.status == "completed" and self.result is not None:
            task["result"] = self.result
        elif self.status == "failed" and self.error is not None:
            task["error"] = self.error
        elif self.status == "input_required" and self.input_requests:
            task["inputRequests"] = self.input_requests
        if self.approval_id is not None:
            task["_meta"] = {APPROVAL_ID_META: self.approval_id}
        return task


class TaskTable:
    """Live tasks of this process; a lookup by another caller finds nothing."""

    def __init__(self, options: TaskOptions) -> None:
        self.options = options
        self._tasks: dict[str, TaskRecord] = {}

    def create(self, principal: str | None, tool: str) -> TaskRecord | None:
        """A new working task, or None when this process or this caller is at its limit."""
        self._purge()
        if len(self._tasks) >= self.options.max_tasks:
            logger.warning("MCP tasks: %d tasks live; answering without a task", len(self._tasks))
            return None
        owned = Counter(record.principal for record in self._tasks.values())
        if owned[principal] >= self.options.max_per_principal:
            logger.warning("MCP tasks: a caller is at its task limit; answering without a task")
            return None
        task_id = secrets.token_urlsafe(TASK_ID_BYTES)
        record = TaskRecord(task_id, principal, tool, self.options)
        self._tasks[task_id] = record
        return record

    def get(self, task_id: str, principal: str | None) -> TaskRecord | None:
        """The caller's live task `task_id`; None when unknown, expired or someone else's."""
        record = self._tasks.get(task_id)
        if record is None:
            return None
        if record.expired():
            self._drop(record)
            return None
        if record.principal != principal:
            logger.warning("MCP tasks: a caller asked for another caller's task; refused")
            return None
        return record

    def drop(self, record: TaskRecord) -> None:
        """Forget `record` (its outcome went straight to the caller)."""
        self._drop(record)

    def __len__(self) -> int:
        return len(self._tasks)

    def _drop(self, record: TaskRecord) -> None:
        if self._tasks.get(record.task_id) is record:
            del self._tasks[record.task_id]
        if record.cancel_scope is not None and not record.terminal:
            record.cancel_scope.cancel()

    def _purge(self) -> None:
        now = time.monotonic()
        for record in [r for r in self._tasks.values() if r.expired(now)]:
            logger.info("MCP task for tool %s expired (%s)", record.tool, record.status)
            self._drop(record)


def owner_digest(principal: str | None) -> str:
    """What the task store keeps of the caller: a digest, never the principal itself."""
    text = "anonymous" if principal is None else "principal:" + principal
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def shared_state(record: TaskRecord, alive_s: float) -> dict[str, Any]:
    """The stored form of `record`; `alive_until` says when its worker must write again."""
    now = time.time()
    return {
        "v": 1,
        "owner": owner_digest(record.principal),
        "task": record.wire(),
        "expires_at": record.created_at + record.options.ttl_s,
        "alive_until": now + alive_s,
    }


def remote_task(state: dict[str, Any], principal: str | None) -> dict[str, Any] | None:
    """The caller's `Task` from a stored state; None when expired, malformed or someone else's.

    A task whose worker stopped writing is shown as failed.
    """
    task, owner = state.get("task"), state.get("owner")
    expires_at, alive_until = state.get("expires_at"), state.get("alive_until")
    if not isinstance(task, dict) or not isinstance(owner, str):
        logger.error("MCP task store state has no task or owner; refused")
        return None
    if not hmac.compare_digest(owner, owner_digest(principal)):
        logger.warning("MCP tasks: a caller asked for another caller's task; refused")
        return None
    now = time.time()
    if not isinstance(expires_at, int | float) or now >= expires_at:
        return None
    if task.get("status") not in TERMINAL and (
        not isinstance(alive_until, int | float) or now >= alive_until
    ):
        logger.warning("MCP task: its worker stopped; reported as failed")
        task = {**task, "status": "failed", "statusMessage": WORKER_GONE}
        task.pop("inputRequests", None)
        task["error"] = {"code": JSONRPC_INTERNAL_ERROR, "message": WORKER_GONE}
    return task
