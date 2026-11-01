"""Client side of the MCP tasks extension: follow a remote task handle to its result."""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field

from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.mcp.annotations import APPROVAL_ID_META, ERROR_KIND_META
from agent_adapter.integrations.mcp.tasks import TASK_ID_META, TASKS_EXTENSION

if TYPE_CHECKING:
    from mcp.client.extension import ClientExtension
    from mcp.client.session import ClientSession
    from mcp.types import CallToolResult

logger = logging.getLogger(__name__)

MIN_POLL_S = 0.05
MAX_POLL_S = 30.0
DEFAULT_POLL_S = 1.0
TASK_PENDING_KIND = "task_pending"


@dataclass(frozen=True)
class Wire:
    """The tasks wire models, built on first use (the SDK is imported lazily)."""

    TaskHandle: Any  # a CreateTaskResult
    TaskState: Any  # a tasks/get result
    GetTaskRequest: Any
    TaskIdParams: Any
    TasksExtension: Any  # the ClientExtension declaring the tasks extension


@functools.cache
def wire() -> Wire:
    """The wire models and the client extension class."""
    from mcp.client.extension import ClaimContext, ClientExtension, ResultClaim
    from mcp.types import Request, RequestParams, Result

    class TaskHandle(Result):  # type: ignore[misc,unused-ignore]
        """A `CreateTaskResult`: the remote turned the call into a task."""

        model_config = ConfigDict(extra="allow")
        result_type: Literal["task"] = "task"
        task_id: str = Field(min_length=1, max_length=256)
        status: str
        status_message: str | None = None
        poll_interval_ms: int | None = None

    class TaskState(Result):  # type: ignore[misc,unused-ignore]
        """A `tasks/get` result."""

        model_config = ConfigDict(extra="allow")
        task_id: str
        status: str
        status_message: str | None = None
        poll_interval_ms: int | None = None
        result: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        input_requests: dict[str, Any] | None = None

    class TaskIdParams(RequestParams):  # type: ignore[misc,unused-ignore]
        task_id: str

    class GetTaskRequest(Request[TaskIdParams, Literal["tasks/get"]]):  # type: ignore[misc,unused-ignore]
        """`tasks/get`; its `taskId` goes in the `Mcp-Name` header too."""

        method: Literal["tasks/get"] = "tasks/get"
        params: TaskIdParams
        name_param = "taskId"

    class TasksExtension(ClientExtension):  # type: ignore[misc,unused-ignore]
        """Declares the tasks extension; a task handle is followed for up to `wait_s`."""

        identifier = TASKS_EXTENSION

        def __init__(self, wait_s: float) -> None:
            self.wait_s = _wait(wait_s)

        def claims(self) -> tuple[ResultClaim[Any], ...]:
            return (ResultClaim(result_type="task", model=TaskHandle, resolve=self._resolve),)

        async def _resolve(self, handle: Any, ctx: ClaimContext) -> CallToolResult:
            logger.info("MCP tool %s became remote task (%s)", ctx.tool_name, handle.status)
            return await follow(ctx.session, handle, self.wait_s)

    return Wire(TaskHandle, TaskState, GetTaskRequest, TaskIdParams, TasksExtension)


def _wait(wait_s: object) -> float:
    if isinstance(wait_s, bool) or not isinstance(wait_s, int | float):
        raise ConfigError(f"task_wait_s must be a number, got {wait_s!r}")
    if not math.isfinite(wait_s) or wait_s < 0:
        raise ConfigError(f"task_wait_s must be finite and >= 0, got {wait_s!r}")
    return float(wait_s)


def tasks_extension_for(wait_s: float) -> ClientExtension:
    """A client extension declaring tasks, following each task for up to `wait_s` seconds.

    Raises:
        ConfigError: `wait_s` is not a finite number >= 0.
    """
    extension: ClientExtension = wire().TasksExtension(wait_s)
    return extension


async def follow(session: ClientSession, handle: Any, wait_s: float) -> CallToolResult:
    """The task's tool result; a task still open after `wait_s` is an error result to resume."""
    import anyio

    models = wire()
    if isinstance(handle, str):
        handle = models.TaskHandle(task_id=handle, status="working")
    last: Any = handle
    with anyio.move_on_after(wait_s):
        while True:
            request = models.GetTaskRequest(params=models.TaskIdParams(task_id=handle.task_id))
            state = await session.send_request(request, models.TaskState)
            last = state
            if state.status in ("completed", "failed", "cancelled"):
                return finished(state)
            if state.status == "input_required":
                return still_open(state, "the remote task needs input this client cannot give")
            await anyio.sleep(_interval(state.poll_interval_ms or handle.poll_interval_ms))
    logger.info("MCP remote task still open after %.1fs; it can be resumed", wait_s)
    return still_open(last, "the remote task is still running")


def _interval(ms: int | None) -> float:
    if ms is None or ms <= 0:
        return DEFAULT_POLL_S
    return min(max(ms / 1000, MIN_POLL_S), MAX_POLL_S)


def finished(state: Any) -> CallToolResult:
    """The tool result of a terminal task; a failed or cancelled one is an error result."""
    from mcp.types import CallToolResult, TextContent

    if state.status == "completed" and state.result is not None:
        return CallToolResult.model_validate(state.result)
    if state.status == "failed" and state.error is not None:
        message = state.error.get("message")
        text = f"remote task failed: {message if isinstance(message, str) else 'error'}"
    else:
        text = "remote task cancelled" if state.status == "cancelled" else "remote task ended"
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=True,
        meta={ERROR_KIND_META: f"task_{state.status}"},
    )


def still_open(state: Any, text: str) -> CallToolResult:
    """An error result naming the task, so the same local call polls it again."""
    from mcp.types import CallToolResult, TextContent

    meta: dict[str, Any] = {TASK_ID_META: state.task_id, ERROR_KIND_META: TASK_PENDING_KIND}
    remote_meta = state.meta or {}
    if isinstance(remote_meta.get(APPROVAL_ID_META), str):
        meta[APPROVAL_ID_META] = remote_meta[APPROVAL_ID_META]
    if state.status_message:
        text = f"{text}: {state.status_message}"
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True, meta=meta)


def tasks_extension(client: Any) -> Any:
    """The tasks extension an `mcp.Client` was built with, if any."""
    cls = wire().TasksExtension
    for extension in getattr(client, "extensions", None) or ():
        if isinstance(extension, cls):
            return extension
    return None
