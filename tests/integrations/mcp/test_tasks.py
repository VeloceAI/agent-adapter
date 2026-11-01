"""MCP tasks extension, server side: slow calls and pending approvals become task handles."""

from __future__ import annotations

import asyncio
import logging

import pytest

pytest.importorskip("mcp")

from mcp import Client, MCPError

from agent_adapter.approval import MemoryApprovalStore
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations.mcp import (
    APPROVAL_ID_META,
    TASKS_EXTENSION,
    TaskOptions,
    mcp_client,
    mcp_server,
)
from agent_adapter.integrations.mcp._task_service import UNKNOWN_TASK, URL_INPUT_KEY
from tests.integrations.mcp.tasks_support import (
    FAST,
    cancel_task,
    get_task,
    make_registry,
    make_server,
    task_id_of,
    until,
)

pytestmark = pytest.mark.contract


def raw(server: object) -> Client:
    """A client that declares tasks but never waits: every task comes back as its handle."""
    return mcp_client(server, tasks=True, task_wait_s=0)


async def test_the_server_advertises_the_extension():
    async with raw(make_server([], MemoryApprovalStore())) as client:
        assert client.server_capabilities.extensions == {TASKS_EXTENSION: {}}


async def test_a_fast_call_is_an_ordinary_result():
    log: list[str] = []
    async with raw(make_server(log, MemoryApprovalStore())) as client:
        result = await client.call_tool("slow", {"seconds": 0})
    assert (result.is_error, result.structured_content) == (False, {"result": "done"})
    assert TASKS_EXTENSION not in str(result.meta)


async def test_a_slow_call_becomes_a_task_that_completes():
    log: list[str] = []
    async with raw(make_server(log, MemoryApprovalStore())) as client:
        handle = await client.call_tool("slow", {"seconds": 0.4})
        assert handle.is_error and handle.meta["agent-adapter/error_kind"] == "task_pending"
        state = await until(client, task_id_of(handle), "completed")
    assert state.result is not None and state.result["structuredContent"] == {"result": "done"}
    assert (state.model_extra or {}).get("ttlMs") == 3_600_000
    assert log == ["slow"]


async def test_a_pending_approval_becomes_a_task_run_once_decided():
    log: list[str] = []
    store = MemoryApprovalStore()
    async with raw(make_server(log, store)) as client:
        handle = await client.call_tool("pay", {"amount": 3})
        task_id = task_id_of(handle)
        waiting = await get_task(client, task_id)
        assert waiting.status == "working" and "waiting for approval" in waiting.status_message
        approval_id = waiting.meta[APPROVAL_ID_META]
        assert log == []
        await store.decide(approval_id, ApprovalDecision(approved=True))
        done = await until(client, task_id, "completed")
    assert done.result is not None and done.result["structuredContent"] == {"result": "paid 3"}
    assert log == ["pay 3"]
    assert (await store.get(approval_id)).consumed


async def test_a_denied_approval_completes_with_an_error_result():
    log: list[str] = []
    store = MemoryApprovalStore()
    async with raw(make_server(log, store)) as client:
        handle = await client.call_tool("pay", {"amount": 3})
        approval_id = handle.meta[APPROVAL_ID_META]
        await store.decide(approval_id, ApprovalDecision(approved=False, reason="no"))
        done = await until(client, task_id_of(handle), "completed")
    assert done.result is not None and done.result["isError"] is True
    assert log == []


async def test_a_failing_tool_in_a_task_is_an_error_result_not_a_failed_task():
    async with raw(make_server([], MemoryApprovalStore())) as client:
        handle = await client.call_tool("boom", {"seconds": 0.4})
        state = await until(client, task_id_of(handle), "completed", "failed")
    assert state.status == "completed" and state.result is not None
    assert state.result["isError"] is True
    assert state.result["_meta"]["agent-adapter/error_kind"] == "tool-failed"


async def test_cancel_stops_a_running_task():
    gate = asyncio.Event()
    log: list[str] = []
    async with raw(make_server(log, MemoryApprovalStore(), gate=gate)) as client:
        handle = await client.call_tool("slow", {"seconds": 0})
        task_id = task_id_of(handle)
        await cancel_task(client, task_id)
        state = await until(client, task_id, "cancelled")
        gate.set()
        await cancel_task(client, task_id)  # a finished task keeps its outcome
        assert (await get_task(client, task_id)).status == "cancelled"
    assert state.status == "cancelled" and log == []


async def test_threshold_none_makes_tasks_only_of_pending_calls():
    log: list[str] = []
    options = TaskOptions(threshold_s=None, poll_interval_s=0.02)
    async with raw(make_server(log, MemoryApprovalStore(), options=options)) as client:
        result = await client.call_tool("slow", {"seconds": 0.3})
        assert not result.is_error
        pending = await client.call_tool("pay", {"amount": 1})
        assert pending.meta["agent-adapter/error_kind"] == "task_pending"


async def test_url_mode_links_clients_that_accept_url_elicitation():
    store = MemoryApprovalStore()
    options = TaskOptions(poll_interval_s=0.02, approval_url="https://ops.example/a/{approval_id}")
    server = make_server([], store, options=options)
    from tests.integrations.mcp.test_tasks_security import url_client

    async with url_client(server) as client:
        handle = await client.call_tool("pay", {"amount": 2})
        state = await get_task(client, task_id_of(handle))
    assert state.status == "input_required" and state.input_requests is not None
    link = state.input_requests[URL_INPUT_KEY]["params"]
    assert link["mode"] == "url"
    assert link["url"] == f"https://ops.example/a/{state.meta[APPROVAL_ID_META]}"


async def test_a_client_without_the_extension_gets_the_classic_pending_result():
    async with mcp_client(make_server([], MemoryApprovalStore())) as client:
        result = await client.call_tool("pay", {"amount": 3})
        assert result.content[0].text.startswith("approval required: ")  # type: ignore[union-attr]
        with pytest.raises(MCPError, match=r"io\.modelcontextprotocol/tasks"):
            await get_task(client, "anything")


async def test_an_unknown_task_id_is_invalid_params():
    async with raw(make_server([], MemoryApprovalStore())) as client:
        with pytest.raises(MCPError, match=UNKNOWN_TASK) as caught:
            await get_task(client, "not-a-task")
    assert caught.value.error.code == -32602


async def test_an_expired_task_is_gone():
    options = TaskOptions(threshold_s=0, ttl_s=0.1, poll_interval_s=0.05)
    async with raw(make_server([], MemoryApprovalStore(), options=options)) as client:
        handle = await client.call_tool("slow", {"seconds": 0.01})
        await asyncio.sleep(0.15)
        with pytest.raises(MCPError, match=UNKNOWN_TASK):
            await get_task(client, task_id_of(handle))


async def test_over_the_per_caller_limit_calls_are_answered_without_a_task(
    caplog: pytest.LogCaptureFixture,
):
    options = TaskOptions(threshold_s=0, poll_interval_s=0.02, max_per_principal=1)
    gate = asyncio.Event()
    async with raw(make_server([], MemoryApprovalStore(), options=options, gate=gate)) as client:
        first = await client.call_tool("slow", {"seconds": 0})
        assert first.meta["agent-adapter/error_kind"] == "task_pending"
        caplog.set_level(logging.WARNING)
        second = await client.call_tool("pay", {"amount": 1})
        gate.set()
    assert second.content[0].text.startswith("approval required: ")  # type: ignore[union-attr]
    assert "task limit" in caplog.text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold_s": -1},
        {"ttl_s": 0},
        {"poll_interval_s": float("nan")},
        {"poll_interval_s": 10, "ttl_s": 5},
        {"max_tasks": 0},
        {"max_per_principal": True},
        {"approval_url": "https://ops.example/a"},
        {"approval_url": "http://ops.example/a/{approval_id}"},
        {"approval_url": "https://u:p@ops.example/{approval_id}"},
    ],
)
def test_task_options_refuse_bad_values(kwargs: dict[str, object]):
    with pytest.raises(ConfigError):
        TaskOptions(**kwargs)  # type: ignore[arg-type]


def test_tasks_must_be_a_bool_or_options():
    with pytest.raises(ConfigError, match="tasks must be"):
        mcp_server(make_registry([]), tasks="yes")  # type: ignore[arg-type]


def test_approval_links_quote_the_id():
    options = TaskOptions(approval_url="http://127.0.0.1:8080/a/{approval_id}")
    assert options.approval_link("a/b?c") == "http://127.0.0.1:8080/a/a%2Fb%3Fc"
    assert FAST.approval_link("x") is None


async def test_without_a_store_a_pending_call_keeps_the_classic_resumable_result():
    from agent_adapter.core.errors import ApprovalPending
    from agent_adapter.exec import Executor
    from tests.fakes.approval import ScriptedApproval

    port = ScriptedApproval([ApprovalPending("apr_1", tool="pay")])
    server = mcp_server(make_registry([]), executor=Executor(approval=port), tasks=FAST)
    async with raw(server) as client:
        result = await client.call_tool("pay", {"amount": 1})
    assert result.content[0].text == "approval required: apr_1"  # type: ignore[union-attr]
    assert "agent-adapter/call_id" in (result.meta or {})


async def test_a_store_failure_fails_the_task(monkeypatch: pytest.MonkeyPatch):
    store = MemoryApprovalStore()
    async with raw(make_server([], store)) as client:
        handle = await client.call_tool("pay", {"amount": 3})

        async def broken(approval_id: str) -> object:
            raise RuntimeError("store down")

        monkeypatch.setattr(store, "get", broken)
        state = await until(client, task_id_of(handle), "failed")
    assert state.error == {"code": -32603, "message": "the task failed"}


async def test_a_protocol_error_in_a_task_fails_it_and_a_fast_one_is_raised():
    from mcp.types import INVALID_PARAMS

    from agent_adapter.integrations.mcp._scope import Scope
    from agent_adapter.integrations.mcp._task_service import LIFESPAN_KEY, TaskService

    async def attempt() -> object:
        raise MCPError(code=INVALID_PARAMS, message="bad state")

    service = TaskService(FAST, principal=None, allow_anonymous=True)
    scope = Scope(None, "r", "c", "run", "call", keys_honoured=True)
    async with service.lifespan(None) as state:  # type: ignore[arg-type]
        from types import SimpleNamespace

        ctx = SimpleNamespace(
            lifespan_context=state, session=SimpleNamespace(client_capabilities=None)
        )
        with pytest.raises(MCPError, match="bad state"):
            await service.call(ctx, "pay", scope, None, attempt, lambda exc: None)  # type: ignore[arg-type]
        assert state[LIFESPAN_KEY] is not None and len(service.table) == 0
