"""MCP tasks extension, client side: imported tools follow remote tasks and resume them."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp")

import httpx2

from agent_adapter.approval import MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ToolError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.mcp import TASK_ID_META, import_mcp, mcp_client
from agent_adapter.integrations.mcp import client as client_module
from agent_adapter.integrations.mcp._tasks_client import (
    _interval,
    finished,
    still_open,
    tasks_extension,
    tasks_extension_for,
    wire,
)
from agent_adapter.integrations.mcp.server import streamable_http_app
from tests.integrations.mcp.tasks_support import FAST, make_registry, make_server

pytestmark = pytest.mark.contract

URL = "http://127.0.0.1:8000/mcp"


async def call(executor: Executor, spec: Any, args: dict[str, Any]) -> Any:
    ctx = ToolContext(run_id="run_local", call_id="call_local")
    return await executor.run_spec(spec, args, ctx, call_id="call_local")


async def test_an_imported_tool_follows_a_slow_remote_task_to_its_result():
    reg = Registry()
    async with mcp_client(make_server([], MemoryApprovalStore()), tasks=True) as client:
        await import_mcp(reg, client, prefix="r_")
        assert await call(Executor(), reg.get("r_slow"), {"seconds": 0.4}) == "done"


async def test_an_imported_pending_task_resumes_once_the_remote_decides():
    log: list[str] = []
    store = MemoryApprovalStore()
    reg = Registry()
    async with mcp_client(make_server(log, store), tasks=True, task_wait_s=0.1) as client:
        await import_mcp(reg, client, prefix="r_")
        with pytest.raises(ToolError, match="waiting for approval") as caught:
            await call(Executor(), reg.get("r_pay"), {"amount": 6})
        details = caught.value.details
        assert details["remote_kind"] == "task_pending" and details["task_id"]
        with pytest.raises(ToolError):  # still undecided: the same task, polled again
            await call(Executor(), reg.get("r_pay"), {"amount": 6})
        await store.decide(details["approval_id"], ApprovalDecision(approved=True))
        assert await call(Executor(), reg.get("r_pay"), {"amount": 6}) == "paid 6"
    assert log == ["pay 6"]


@asynccontextmanager
async def serving(
    app: Any, monkeypatch: pytest.MonkeyPatch, seen: list[Any]
) -> AsyncIterator[None]:
    async def record(request: httpx2.Request) -> None:
        seen.append(request)

    def asgi_client(headers: Mapping[str, str]) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            headers=dict(headers),
            event_hooks={"request": [record]},
        )

    monkeypatch.setattr(client_module, "_new_http_client", asgi_client)
    async with app.router.lifespan_context(app):
        yield


async def test_tasks_over_stateless_http_send_the_mcp_name_header(monkeypatch: pytest.MonkeyPatch):
    from agent_adapter.approval import DeferredApproval

    seen: list[Any] = []
    app = streamable_http_app(
        make_registry([]),
        executor=Executor(approval=DeferredApproval(MemoryApprovalStore())),
        tasks=FAST,
        allow_anonymous=True,
    )
    async with serving(app, monkeypatch, seen), mcp_client(URL, tasks=True) as client:
        result = await client.call_tool("slow", {"seconds": 0.4})
    assert result.structured_content == {"result": "done"}
    polls = [r for r in seen if r.headers.get("mcp-method") == "tasks/get"]
    assert polls and all(r.headers.get("mcp-name") for r in polls)


async def test_a_host_that_drops_meta_keys_still_completes_a_confirm_tool(
    monkeypatch: pytest.MonkeyPatch,
):
    """A standard tasks client sends no agent-adapter `_meta`: it resumes by task id only."""
    import anyio
    from mcp import Client

    from agent_adapter.approval import DeferredApproval

    log: list[str] = []
    store = MemoryApprovalStore()
    seen: list[Any] = []
    app = streamable_http_app(
        make_registry(log),
        executor=Executor(approval=DeferredApproval(store)),
        tasks=FAST,
        allow_anonymous=True,
    )

    async def reviewer() -> None:
        while not (records := await store.list()):
            await anyio.sleep(0.01)
        await store.decide(records[0].approval_id, ApprovalDecision(approved=True))

    async with serving(app, monkeypatch, seen):
        transport = mcp_client(URL).server  # the plain SDK transport, no adapter client logic
        async with (
            Client(transport, extensions=[tasks_extension_for(5)]) as host,
            anyio.create_task_group() as group,
        ):
            group.start_soon(reviewer)
            result = await host.call_tool("pay", {"amount": 8})
    assert (result.is_error, result.structured_content) == (False, {"result": "paid 8"})
    assert log == ["pay 8"]
    (record,) = await store.list()
    assert record.consumed
    sent = [request.content.decode() for request in seen if request.method == "POST"]
    assert sent and not any("agent-adapter/" in body for body in sent)


async def test_an_anonymous_local_http_app_answers_without_tasks(monkeypatch: pytest.MonkeyPatch):
    seen: list[Any] = []
    app = streamable_http_app(make_registry([]), tasks=FAST)
    async with serving(app, monkeypatch, seen), mcp_client(URL, tasks=True) as client:
        result = await client.call_tool("slow", {"seconds": 0.3})
    assert result.structured_content == {"result": "done"}
    assert not [r for r in seen if r.headers.get("mcp-method") == "tasks/get"]


def test_remote_outcomes_map_to_results():
    failed = wire().TaskState(task_id="t", status="failed", error={"code": -1, "message": "bad"})
    assert finished(failed).content[0].text == "remote task failed: bad"  # type: ignore[union-attr]
    cancelled = finished(wire().TaskState(task_id="t", status="cancelled"))
    assert cancelled.is_error and cancelled.meta["agent-adapter/error_kind"] == "task_cancelled"
    asked = still_open(
        wire().TaskState(task_id="t", status="input_required", status_message="sign in"),
        "the remote task needs input",
    )
    assert asked.meta[TASK_ID_META] == "t"
    assert asked.content[0].text == "the remote task needs input: sign in"  # type: ignore[union-attr]
    assert (
        still_open(wire().TaskHandle(task_id="h", status="working"), "x").meta[TASK_ID_META] == "h"
    )


def test_poll_intervals_are_clamped():
    assert _interval(None) == 1.0 and _interval(0) == 1.0
    assert _interval(1) == 0.05 and _interval(10**9) == 30.0


@pytest.mark.parametrize("wait", [-1, float("inf"), True, "5"])
def test_a_bad_task_wait_is_a_config_error(wait: Any):
    with pytest.raises(ConfigError):
        tasks_extension_for(wait)


def test_the_extension_is_found_on_the_client():
    assert tasks_extension(mcp_client(URL)) is None
    assert isinstance(tasks_extension(mcp_client(URL, tasks=True)), wire().TasksExtension)
