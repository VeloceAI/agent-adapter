"""Shared fakes for the MCP tasks tests: a registry, raw tasks/* requests and a principal switch."""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any, Literal

from mcp import Client
from mcp_types import Request, RequestParams, Result

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.integrations.mcp import TASK_ID_META, TaskOptions, mcp_server
from agent_adapter.integrations.mcp._tasks_client import wire

CALLER: contextvars.ContextVar[str | None] = contextvars.ContextVar("caller", default="alice")
FAST = TaskOptions(threshold_s=0.2, poll_interval_s=0.02)


def caller(ctx: Any) -> str | None:
    """A principal hook: whoever the test says is calling."""
    return CALLER.get()


def make_registry(log: list[str], gate: asyncio.Event | None = None) -> Registry:
    reg = Registry()

    @reg.tool(confirm=True)
    async def pay(amount: int) -> str:
        """Pay (needs approval)."""
        log.append(f"pay {amount}")
        return f"paid {amount}"

    @reg.tool
    async def slow(seconds: float) -> str:
        """Sleep, or wait for the gate."""
        if gate is not None:
            await gate.wait()
        else:
            await asyncio.sleep(seconds)
        log.append("slow")
        return "done"

    @reg.tool
    async def boom(seconds: float = 0) -> str:
        """Fail, after a while."""
        await asyncio.sleep(seconds)
        raise ValueError("boom")

    return reg


def make_server(
    log: list[str],
    store: MemoryApprovalStore,
    *,
    options: TaskOptions = FAST,
    gate: asyncio.Event | None = None,
    **kw: Any,
) -> Any:
    kw.setdefault("principal", caller)
    return mcp_server(
        make_registry(log, gate),
        executor=Executor(approval=DeferredApproval(store)),
        tasks=options,
        **kw,
    )


class TaskIdParams(RequestParams):
    task_id: str


class UpdateParams(TaskIdParams):
    input_responses: dict[str, Any]


class GetTask(Request[TaskIdParams, Literal["tasks/get"]]):
    method: Literal["tasks/get"] = "tasks/get"
    params: TaskIdParams
    name_param = "taskId"


class UpdateTask(Request[UpdateParams, Literal["tasks/update"]]):
    method: Literal["tasks/update"] = "tasks/update"
    params: UpdateParams
    name_param = "taskId"


class CancelTask(Request[TaskIdParams, Literal["tasks/cancel"]]):
    method: Literal["tasks/cancel"] = "tasks/cancel"
    params: TaskIdParams
    name_param = "taskId"


class Ack(Result):
    result_type: str | None = None


async def get_task(client: Client, task_id: str) -> Any:
    return await client.session.send_request(
        GetTask(params=TaskIdParams(task_id=task_id)), wire().TaskState
    )


async def update_task(client: Client, task_id: str, responses: dict[str, Any]) -> Ack:
    request = UpdateTask(params=UpdateParams(task_id=task_id, input_responses=responses))
    return await client.session.send_request(request, Ack)


async def cancel_task(client: Client, task_id: str) -> Ack:
    return await client.session.send_request(CancelTask(params=TaskIdParams(task_id=task_id)), Ack)


async def until(client: Client, task_id: str, *statuses: str, timeout: float = 5.0) -> Any:
    """Poll `tasks/get` until the task reaches one of `statuses`."""
    with anyio_timeout(timeout):
        while True:
            state = await get_task(client, task_id)
            if state.status in statuses:
                return state
            await asyncio.sleep(0.01)


def anyio_timeout(seconds: float) -> Any:
    import anyio

    return anyio.fail_after(seconds)


def task_id_of(result: Any) -> str:
    meta = result.meta or {}
    task_id = meta.get(TASK_ID_META)
    assert isinstance(task_id, str), meta
    return task_id
