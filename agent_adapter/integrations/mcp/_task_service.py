"""Serve the MCP tasks extension: slow and pending calls become task handles to poll.

A task's approval is decided only in the executor's approval store; `tasks/update` answers
never approve anything. With a shared `TaskStore` any worker answers
`tasks/*`; only the worker running a task writes its state, the others raise signals.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.errors import ApprovalPending, log_failure
from agent_adapter.integrations.mcp._resume import refuse
from agent_adapter.integrations.mcp._scope import PrincipalHook, Scope, principal_of
from agent_adapter.integrations.mcp._task_params import (
    accepts_url,
    client_accepts_tasks,
    params_types,
)
from agent_adapter.integrations.mcp._task_store import MemoryTaskStore, OutageLog, TaskStore
from agent_adapter.integrations.mcp._task_table import (
    TERMINAL,
    TaskRecord,
    TaskTable,
    remote_task,
    shared_state,
)
from agent_adapter.integrations.mcp.annotations import APPROVAL_ID_META
from agent_adapter.integrations.mcp.tasks import TASKS_EXTENSION, TaskOptions

if TYPE_CHECKING:
    import anyio.abc
    import mcp.types as t
    from mcp.server.context import ServerRequestContext
    from mcp.server.lowlevel import Server

logger = logging.getLogger(__name__)

LIFESPAN_KEY = "agent-adapter/tasks"
URL_INPUT_KEY = "agent-adapter/approval-url"
UNKNOWN_TASK = "Unknown or expired task"
MCP_NAME_HEADER = "mcp-name"
CANCEL_FLAG = "cancel"
ALIVE_POLLS = 3  # a worker that misses this many store writes (plus slack) is gone
ALIVE_SLACK_S = 5.0  # clock skew between hosts

Attempt = Callable[[], Awaitable[Any]]  # -> CallToolResult | ApprovalPending
Settle = Callable[[ApprovalPending], "t.CallToolResult"]


class TaskService:
    """The task table, the background runs and the `tasks/*` handlers of one server."""

    def __init__(
        self,
        options: TaskOptions,
        *,
        principal: PrincipalHook | None,
        allow_anonymous: bool | None,
        store: TaskStore | None = None,
    ) -> None:
        self.options = options
        self.principal = principal
        self.allow_anonymous = allow_anonymous
        self.table = TaskTable(options)
        self.store: TaskStore = MemoryTaskStore() if store is None else store
        self._outage = OutageLog(logger)  # one WARNING per task while the store is down

    def install(self, server: Server[Any]) -> None:
        """Advertise the extension and serve `tasks/get`, `tasks/update` and `tasks/cancel`."""
        task_id_params, update_params = params_types()
        server.extensions[TASKS_EXTENSION] = {}
        server.add_request_handler("tasks/get", task_id_params, self.get)
        server.add_request_handler("tasks/update", update_params, self.update)
        server.add_request_handler("tasks/cancel", task_id_params, self.cancel)

    @asynccontextmanager
    async def lifespan(self, server: Server[Any]) -> AsyncIterator[dict[str, Any]]:
        """The server lifespan: background runs live in its task group and end with it."""
        import anyio

        async with anyio.create_task_group() as group:
            yield {LIFESPAN_KEY: group}
            group.cancel_scope.cancel()

    def accepts(self, ctx: ServerRequestContext[Any, Any], scope: Scope) -> bool:
        """Whether this call may become a task (never a resume: it finishes in the request)."""
        return (
            scope.keys_honoured
            and not scope.resuming
            and _group(ctx) is not None
            and client_accepts_tasks(ctx)
        )

    async def call(
        self,
        ctx: ServerRequestContext[Any, Any],
        tool: str,
        scope: Scope,
        store: ApprovalStore | None,
        attempt: Attempt,
        settle: Settle,
    ) -> Any:
        """The call's result if it settles in time, else a `CreateTaskResult` (wire dict)."""
        import anyio

        group = _group(ctx)
        record = self.table.create(scope.principal, tool)
        if record is None or group is None:
            outcome = await attempt()
            return settle(outcome) if isinstance(outcome, ApprovalPending) else outcome
        url_ok = accepts_url(ctx)
        group.start_soon(self._drive, record, attempt, settle, store, url_ok)
        wait = self.options.threshold_s
        with anyio.move_on_after(math.inf if wait is None else wait):
            await record.settled.wait()
        if record.approval_id is None and record.status in ("completed", "failed"):
            self.table.drop(record)  # finished in time: an ordinary result or protocol error
            return _ordinary(record)
        handle = {**record.wire(), "resultType": "task"}  # before the store write yields
        record.shared = True
        await self._save(record)
        group.start_soon(self._watch, record)
        logger.info("MCP tool %s became task (%s)", tool, handle["status"])
        return handle

    async def _drive(
        self,
        record: TaskRecord,
        attempt: Attempt,
        settle: Settle,
        store: ApprovalStore | None,
        url_ok: bool,
    ) -> None:
        import anyio
        from mcp import MCPError
        from mcp.types import INTERNAL_ERROR

        if record.terminal:  # cancelled before it started
            return
        with anyio.CancelScope() as cancel_scope:
            record.cancel_scope = cancel_scope
            try:
                await self._attempts(record, attempt, settle, store, url_ok)
            except MCPError as exc:
                log_failure(
                    logger, logging.WARNING, exc, "MCP task for tool %s failed", record.tool
                )
                record.fail(exc.error.code, exc.error.message)
            except Exception as exc:
                log_failure(logger, logging.ERROR, exc, "MCP task for tool %s raised", record.tool)
                record.fail(INTERNAL_ERROR, "the task failed")
        if cancel_scope.cancelled_caught:
            logger.info("MCP task for tool %s cancelled", record.tool)
            record.cancelled()
        await self._save(record)

    async def _attempts(
        self,
        record: TaskRecord,
        attempt: Attempt,
        settle: Settle,
        store: ApprovalStore | None,
        url_ok: bool,
    ) -> None:
        while True:
            outcome = await attempt()
            if not isinstance(outcome, ApprovalPending):
                record.complete(outcome.model_dump(mode="json", by_alias=True, exclude_none=True))
                return
            if store is None:  # no store to watch: the classic pending result, resumable
                result = settle(outcome)
                record.complete(result.model_dump(mode="json", by_alias=True, exclude_none=True))
                return
            record.waiting(outcome.approval_id, self._link(outcome, url_ok))
            await self._save(record)
            if not await self._decided(record, store, outcome.approval_id):
                return
            record.working("approval decided: running")
            await self._save(record)

    def _link(self, pending: ApprovalPending, url_ok: bool) -> dict[str, Any] | None:
        link = self.options.approval_link(pending.approval_id)
        if link is None or not url_ok:
            return None
        import mcp.types as t

        request = t.ElicitRequest(
            params=t.ElicitRequestURLParams(
                message=f"Review the call to tool {pending.tool!r} ({pending.approval_id})",
                url=link,
            )
        )
        return {URL_INPUT_KEY: request.model_dump(mode="json", by_alias=True, exclude_none=True)}

    async def _decided(self, record: TaskRecord, store: ApprovalStore, approval_id: str) -> bool:
        """Wait until the store holds a decision; False when the task expires first."""
        import anyio

        while not record.expired():
            with anyio.move_on_after(self.options.poll_interval_s):
                await record.nudged.wait()
            record.renew_nudge()
            current = await store.get(approval_id)  # ApprovalNotFound fails the task
            if current.status != "pending":
                logger.info("MCP task for tool %s: approval %s decided", record.tool, approval_id)
                return True
        return False

    async def _save(self, record: TaskRecord) -> None:
        """Mirror `record` to the task store; a failed write leaves it served by this worker."""
        import anyio

        if not record.shared:
            return
        try:  # uncontended: no checkpoint, so the handle goes out before the run moves on
            record.save_lock.acquire_nowait()
        except anyio.WouldBlock:
            await record.save_lock.acquire()
        try:  # snapshot inside the lock: the newest state lands last
            ttl = record.deadline - time.monotonic()
            if ttl <= 0:
                return
            alive = ALIVE_POLLS * self.options.poll_interval_s + ALIVE_SLACK_S
            await self.store.put(record.task_id, shared_state(record, alive), ttl)
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "MCP task %s not stored", record.tool)
        finally:
            record.save_lock.release()

    async def _watch(self, record: TaskRecord) -> None:
        """Apply other workers' cancel and answer signals; keep the stored state fresh."""
        import anyio

        try:
            while not record.terminal and not record.expired():
                await anyio.sleep(self.options.poll_interval_s)
                if record.terminal:
                    return
                await self._poll(record)
        finally:
            self._outage.forget(record.task_id)

    async def _poll(self, record: TaskRecord) -> None:
        answered = None if record.approval_id is None else _answered_flag(record.approval_id)
        names = [CANCEL_FLAG] if answered is None else [CANCEL_FLAG, answered]
        try:
            raised = await self.store.flagged(record.task_id, names)
        except Exception as exc:
            self._outage.failed(record.task_id, exc, "MCP task %s: store unread", record.tool)
            raised = set()
        else:
            self._outage.ok(record.task_id, "MCP task %s: store readable again", record.tool)
        if CANCEL_FLAG in raised:
            self._cancel(record)
        elif answered in raised:
            record.answered(set(record.input_requests or ()))
        await self._save(record)

    def _check(self, ctx: ServerRequestContext[Any, Any], task_id: str) -> str | None:
        """The caller's principal, after the capability, `Mcp-Name` and anonymity checks."""
        from mcp import MCPError
        from mcp.types import MISSING_REQUIRED_CLIENT_CAPABILITY

        if not client_accepts_tasks(ctx):
            raise MCPError(
                code=MISSING_REQUIRED_CLIENT_CAPABILITY,
                message="tasks/* need the io.modelcontextprotocol/tasks client capability",
                data={"requiredCapabilities": {"extensions": {TASKS_EXTENSION: {}}}},
            )
        headers = getattr(ctx.request, "headers", None)
        named = headers.get(MCP_NAME_HEADER) if isinstance(headers, Mapping) else None
        if named is not None and named != task_id:
            refuse("the Mcp-Name header differs from params.taskId")
        principal = principal_of(ctx, self.principal)
        anonymous_ok = ctx.request is None if self.allow_anonymous is None else self.allow_anonymous
        if not principal and not anonymous_ok:
            refuse(UNKNOWN_TASK)
        return principal

    async def _find(
        self, ctx: ServerRequestContext[Any, Any], task_id: str
    ) -> TaskRecord | dict[str, Any]:
        """This worker's record of the caller's task, else its `Task` from the shared store."""
        principal = self._check(ctx, task_id)
        record = self.table.get(task_id, principal)
        if record is not None:
            return record
        state = await self._load(task_id)
        task = None if state is None else remote_task(state, principal)
        if task is None:
            refuse(UNKNOWN_TASK)
        return task

    async def _load(self, task_id: str) -> dict[str, Any] | None:
        from mcp import MCPError
        from mcp.types import INTERNAL_ERROR

        try:
            return await self.store.get(task_id)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "MCP task store cannot be read")
            raise MCPError(code=INTERNAL_ERROR, message="the task store is unavailable") from exc

    async def get(self, ctx: ServerRequestContext[Any, Any], params: Any) -> dict[str, Any]:
        """`tasks/get`: the caller's task, with its result once terminal."""
        found = await self._find(ctx, params.task_id)
        task = found.wire() if isinstance(found, TaskRecord) else found
        return {**task, "resultType": "complete"}

    async def update(self, ctx: ServerRequestContext[Any, Any], params: Any) -> dict[str, Any]:
        """`tasks/update`: answers to input requests; they move the task on, never approve."""
        found = await self._find(ctx, params.task_id)
        keys = set(params.input_responses)
        if isinstance(found, TaskRecord):
            found.answered(keys)
            await self._save(found)
            return {"resultType": "complete"}
        requests = found.get("inputRequests")
        meta = found.get("_meta")
        approval_id = meta.get(APPROVAL_ID_META) if isinstance(meta, dict) else None
        if isinstance(requests, dict) and keys & set(requests) and isinstance(approval_id, str):
            await self._signal(params.task_id, _answered_flag(approval_id))
        return {"resultType": "complete"}

    async def cancel(self, ctx: ServerRequestContext[Any, Any], params: Any) -> dict[str, Any]:
        """`tasks/cancel`: cooperative; a finished task keeps its outcome."""
        found = await self._find(ctx, params.task_id)
        if isinstance(found, TaskRecord):
            if not found.terminal:
                self._cancel(found)
                await self._save(found)
        elif found.get("status") not in TERMINAL:
            logger.info("MCP task: cancel requested for a task another worker runs")
            await self._signal(params.task_id, CANCEL_FLAG)
        return {"resultType": "complete"}

    def _cancel(self, record: TaskRecord) -> None:
        logger.info("MCP task for tool %s: cancel requested", record.tool)
        if record.cancel_scope is None:
            record.cancelled()
        else:
            record.cancel_scope.cancel()

    async def _signal(self, task_id: str, name: str) -> None:
        from mcp import MCPError
        from mcp.types import INTERNAL_ERROR

        try:
            await self.store.flag(task_id, name, self.options.ttl_s)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "MCP task signal %s not stored", name)
            raise MCPError(code=INTERNAL_ERROR, message="the task store is unavailable") from exc


def _answered_flag(approval_id: str) -> str:
    """One signal per approval, so an old answer never moves a later wait on."""
    return f"answered:{approval_id}"


def _ordinary(record: TaskRecord) -> Any:
    from mcp import MCPError
    from mcp.types import CallToolResult

    if record.status == "failed" and record.error is not None:
        raise MCPError(code=record.error["code"], message=record.error["message"])
    return CallToolResult.model_validate(record.result)


def _group(ctx: ServerRequestContext[Any, Any]) -> anyio.abc.TaskGroup | None:
    import anyio.abc

    state = ctx.lifespan_context
    group = state.get(LIFESPAN_KEY) if isinstance(state, Mapping) else None
    return group if isinstance(group, anyio.abc.TaskGroup) else None
