"""MCP tasks across workers: two `mcp_server`s with one `TaskOptions(store=...)` share tasks."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp import Client, MCPError

from agent_adapter.approval import MemoryApprovalStore
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations.mcp import (
    APPROVAL_ID_META,
    MemoryTaskStore,
    RedisTaskStore,
    TaskOptions,
    TaskStore,
    mcp_client,
)
from agent_adapter.integrations.mcp._task_service import UNKNOWN_TASK, URL_INPUT_KEY
from agent_adapter.integrations.mcp._task_table import WORKER_GONE, owner_digest
from tests.integrations.mcp.tasks_support import (
    CALLER,
    cancel_task,
    get_task,
    make_server,
    task_id_of,
    until,
    update_task,
)
from tests.integrations.mcp.test_tasks_security import url_client

pytestmark = pytest.mark.contract

FAST = TaskOptions(threshold_s=0.2, poll_interval_s=0.02)


def redis_store() -> TaskStore:
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    return RedisTaskStore(lambda: fakeredis.FakeAsyncRedis(server=server))


STORES: dict[str, Callable[[], TaskStore]] = {"memory": MemoryTaskStore, "redis": redis_store}


def raw(server: object) -> Client:
    return mcp_client(server, tasks=True, task_wait_s=0)


@contextlib.asynccontextmanager
async def workers(
    log: list[str],
    approvals: MemoryApprovalStore,
    store: TaskStore,
    client: Callable[[Any], Client] = raw,
    **kw: Any,
) -> AsyncIterator[tuple[Client, Client]]:
    """Two workers of one service (one approval store, one task store) and a client of each."""
    kw["options"] = dataclasses.replace(kw.get("options", FAST), store=store)
    first, second = make_server(log, approvals, **kw), make_server(log, approvals, **kw)
    async with client(first) as one, client(second) as two:
        yield one, two


@pytest.mark.parametrize("backend", sorted(STORES))
async def test_a_task_started_on_one_worker_is_followed_and_finished_on_another(
    backend: str,
) -> None:
    store = STORES[backend]()
    log: list[str] = []
    approvals = MemoryApprovalStore()
    async with workers(log, approvals, store) as (one, two):
        task_id = task_id_of(await one.call_tool("pay", {"amount": 3}))
        waiting = await get_task(two, task_id)
        assert waiting.status == "working" and "waiting for approval" in waiting.status_message
        approval_id = waiting.meta[APPROVAL_ID_META]
        await approvals.decide(approval_id, ApprovalDecision(approved=True))
        done = await until(two, task_id, "completed")
    assert done.result is not None and done.result["structuredContent"] == {"result": "paid 3"}
    assert log == ["pay 3"]  # run once, by the worker that owns the task
    assert (await approvals.get(approval_id)).consumed


async def test_another_worker_cancels_a_running_task() -> None:
    store = MemoryTaskStore()
    gate = asyncio.Event()
    log: list[str] = []
    async with workers(log, MemoryApprovalStore(), store, gate=gate) as (one, two):
        task_id = task_id_of(await one.call_tool("slow", {"seconds": 0}))
        await cancel_task(two, task_id)
        assert (await until(two, task_id, "cancelled")).status == "cancelled"
        assert (await get_task(one, task_id)).status == "cancelled"
        gate.set()
    assert log == []


async def test_another_caller_on_another_worker_finds_nothing() -> None:
    store = MemoryTaskStore()
    log: list[str] = []
    approvals = MemoryApprovalStore()
    async with workers(log, approvals, store) as (one, two):
        task_id = task_id_of(await one.call_tool("pay", {"amount": 9}))
        token = CALLER.set("mallory")
        try:
            for call in (get_task, cancel_task):
                with pytest.raises(MCPError, match=UNKNOWN_TASK) as caught:
                    await call(two, task_id)
                assert caught.value.error.code == -32602
            with pytest.raises(MCPError, match=UNKNOWN_TASK):
                await update_task(two, task_id, {URL_INPUT_KEY: {"action": "accept"}})
        finally:
            CALLER.reset(token)
        assert (await get_task(two, task_id)).status == "working"
    assert log == []


async def test_an_answer_on_another_worker_moves_the_task_on_but_never_approves() -> None:
    store = MemoryTaskStore()
    log: list[str] = []
    approvals = MemoryApprovalStore()
    options = TaskOptions(poll_interval_s=0.02, approval_url="https://ops.example/a/{approval_id}")
    async with workers(log, approvals, store, client=url_client, options=options) as (one, two):
        task_id = task_id_of(await one.call_tool("pay", {"amount": 2}))
        state = await until(two, task_id, "input_required")
        await update_task(two, task_id, {URL_INPUT_KEY: {"action": "accept"}})
        moved = await until(two, task_id, "working")
        assert moved.meta[APPROVAL_ID_META] == state.meta[APPROVAL_ID_META]
        await asyncio.sleep(0.1)
        assert (await get_task(two, task_id)).status == "working"
        assert log == []  # the answer approved nothing: the approval store still decides
        await approvals.decide(state.meta[APPROVAL_ID_META], ApprovalDecision(approved=True))
        await until(two, task_id, "completed")
    assert log == ["pay 2"]


async def test_a_task_whose_worker_stopped_reads_as_failed() -> None:
    store = MemoryTaskStore()
    now = time.time()
    task = {"taskId": "t1", "status": "working", "createdAt": "x", "lastUpdatedAt": "x"}
    state = {"owner": owner_digest("alice"), "task": task, "expires_at": now + 60}
    await store.put("t1", {**state, "alive_until": now - 1}, 60)
    async with raw(
        make_server([], MemoryApprovalStore(), options=dataclasses.replace(FAST, store=store))
    ) as client:
        gone = await get_task(client, "t1")
        assert gone.status == "failed" and gone.status_message == WORKER_GONE
        await store.put("t1", {**state, "expires_at": now - 1, "alive_until": now + 60}, 60)
        with pytest.raises(MCPError, match=UNKNOWN_TASK):
            await get_task(client, "t1")
        await store.put("t1", {"task": "not a task"}, 60)
        with pytest.raises(MCPError, match=UNKNOWN_TASK):
            await get_task(client, "t1")


class Broken(MemoryTaskStore):
    async def get(self, task_id: str) -> dict[str, Any] | None:
        raise ConnectionError("redis down: secret-host")

    async def put(self, task_id: str, state: Any, ttl_s: float) -> None:
        raise ConnectionError("redis down: secret-host")


async def test_a_store_outage_fails_closed_elsewhere_and_the_owner_still_serves(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = Broken()
    log: list[str] = []
    caplog.set_level(logging.WARNING)
    async with workers(log, MemoryApprovalStore(), store) as (one, two):
        task_id = task_id_of(await one.call_tool("slow", {"seconds": 0.4}))
        with pytest.raises(MCPError, match="task store is unavailable"):
            await get_task(two, task_id)
        assert (await until(one, task_id, "completed")).status == "completed"
    assert log == ["slow"]
    assert "not stored (ConnectionError)" in caplog.text
    assert "secret-host" not in caplog.text


async def test_task_options_refuse_a_store_that_is_not_a_task_store() -> None:
    from agent_adapter.core.errors import ConfigError

    with pytest.raises(ConfigError, match="TaskStore"):
        TaskOptions(store="redis://x")  # type: ignore[arg-type]
    assert TaskOptions(store=MemoryTaskStore()) == TaskOptions()  # the store is not compared


async def test_two_workers_with_their_own_redis_clients_share_tasks() -> None:
    """Each worker builds its own RedisTaskStore; only the Redis server is shared."""
    fakeredis = pytest.importorskip("fakeredis")
    redis_server = fakeredis.FakeServer()

    def worker_options() -> TaskOptions:
        store = RedisTaskStore(lambda: fakeredis.FakeAsyncRedis(server=redis_server))
        return TaskOptions(threshold_s=0.2, poll_interval_s=0.02, store=store)

    log: list[str] = []
    approvals = MemoryApprovalStore()
    first = make_server(log, approvals, options=worker_options())
    second = make_server(log, approvals, options=worker_options())
    async with raw(first) as one, raw(second) as two:
        task_id = task_id_of(await one.call_tool("slow", {"seconds": 0.4}))
        assert (await get_task(two, task_id)).status == "working"
        done = await until(two, task_id, "completed")
    assert done.result is not None and done.result["structuredContent"] == {"result": "done"}
    assert log == ["slow"]


class Unreadable(MemoryTaskStore):
    async def flagged(self, task_id: str, names: Any) -> set[str]:
        raise ConnectionError("redis down: secret-host")


async def test_a_store_outage_is_logged_once_per_task_not_on_every_poll(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    async with workers([], MemoryApprovalStore(), Unreadable()) as (one, _):
        task_id = task_id_of(await one.call_tool("slow", {"seconds": 0.4}))
        assert (await until(one, task_id, "completed")).status == "completed"
    assert caplog.text.count("store unread") == 1
