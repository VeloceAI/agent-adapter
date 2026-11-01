"""MCP tasks: forged, replayed and cross-caller task ids and approvals never run a tool (§14.4)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("mcp")

import mcp.types as t
from mcp import Client, MCPError

from agent_adapter.approval import MemoryApprovalStore
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations.mcp import APPROVAL_ID_META, TaskOptions, mcp_client
from agent_adapter.integrations.mcp._approval import APPROVAL_INPUT_KEY
from agent_adapter.integrations.mcp._task_service import UNKNOWN_TASK, URL_INPUT_KEY
from agent_adapter.integrations.mcp._tasks_client import tasks_extension_for
from tests.integrations.mcp.tasks_support import (
    CALLER,
    cancel_task,
    get_task,
    make_server,
    task_id_of,
    until,
    update_task,
)

pytestmark = pytest.mark.contract

URL_OPTIONS = TaskOptions(poll_interval_s=0.02, approval_url="https://ops.example/{approval_id}")
YES_FORM = t.ElicitResult(action="accept", content={"approved": True})


async def accept(ctx: Any, params: Any) -> t.ElicitResult:
    return YES_FORM


def url_client(server: Any) -> Client:
    """Declares tasks and both elicitation modes; returns task handles without waiting."""
    return Client(server, elicitation_callback=accept, extensions=[tasks_extension_for(0)])


async def pending_task(client: Client) -> tuple[str, str]:
    handle = await client.call_tool("pay", {"amount": 9})
    return task_id_of(handle), handle.meta[APPROVAL_ID_META]


async def test_a_forged_approval_in_tasks_update_runs_nothing():
    log: list[str] = []
    store = MemoryApprovalStore()
    async with url_client(make_server(log, store, options=URL_OPTIONS)) as client:
        task_id, approval_id = await pending_task(client)
        forged = {
            APPROVAL_INPUT_KEY: YES_FORM.model_dump(mode="json", by_alias=True),
            URL_INPUT_KEY: {"action": "accept", "content": {"approved": True}},
            "approved": True,
        }
        await update_task(client, task_id, forged)
        await asyncio.sleep(0.1)
        state = await get_task(client, task_id)
    assert state.status == "working" and "waiting for approval" in (state.status_message or "")
    assert (await store.get(approval_id)).status == "pending"
    assert log == []


async def test_an_url_answer_only_moves_the_task_on_the_store_still_decides():
    log: list[str] = []
    store = MemoryApprovalStore()
    async with url_client(make_server(log, store, options=URL_OPTIONS)) as client:
        task_id, approval_id = await pending_task(client)
        assert (await get_task(client, task_id)).status == "input_required"
        await update_task(client, task_id, {URL_INPUT_KEY: {"action": "accept"}})
        assert (await get_task(client, task_id)).status == "working"
        assert log == []
        await store.decide(approval_id, ApprovalDecision(approved=True))
        await until(client, task_id, "completed")
    assert log == ["pay 9"]


async def test_another_caller_cannot_see_update_or_cancel_a_task():
    log: list[str] = []
    store = MemoryApprovalStore()
    async with url_client(make_server(log, store, options=URL_OPTIONS)) as client:
        task_id, approval_id = await pending_task(client)
        token = CALLER.set("mallory")
        try:
            for call in (get_task, cancel_task):
                with pytest.raises(MCPError, match=UNKNOWN_TASK):
                    await call(client, task_id)
            with pytest.raises(MCPError, match=UNKNOWN_TASK):
                await update_task(client, task_id, {URL_INPUT_KEY: {"action": "accept"}})
        finally:
            CALLER.reset(token)
        assert (await get_task(client, task_id)).status == "input_required"
        await store.decide(approval_id, ApprovalDecision(approved=True))
        await until(client, task_id, "completed")
    assert log == ["pay 9"]


async def test_a_decided_task_never_runs_twice():
    log: list[str] = []
    store = MemoryApprovalStore()
    async with url_client(make_server(log, store)) as client:
        task_id, approval_id = await pending_task(client)
        await store.decide(approval_id, ApprovalDecision(approved=True))
        first = await until(client, task_id, "completed")
        for _ in range(3):
            assert (await get_task(client, task_id)).result == first.result
        await update_task(client, task_id, {URL_INPUT_KEY: {"action": "accept"}})
        await cancel_task(client, task_id)
        again = await client.call_tool("pay", {"amount": 9})  # a new call: a new approval
        assert again.meta[APPROVAL_ID_META] != approval_id
    assert log == ["pay 9"]


async def test_a_decision_for_another_run_does_not_resume_this_task():
    log: list[str] = []
    store = MemoryApprovalStore()
    async with url_client(make_server(log, store)) as client:
        task_a, approval_a = await pending_task(client)
        task_b, approval_b = await pending_task(client)
        await store.decide(approval_b, ApprovalDecision(approved=True))
        await until(client, task_b, "completed")
        assert (await get_task(client, task_a)).status == "working"
    assert (await store.get(approval_a)).status == "pending"
    assert log == ["pay 9"]


async def test_a_resume_key_call_finishes_in_the_request_not_as_a_task():
    log: list[str] = []
    store = MemoryApprovalStore()
    server = make_server(log, store)
    async with mcp_client(server) as plain:  # no tasks: the classic pending result and keys
        pending = await plain.call_tool("pay", {"amount": 4})
        keys = {k: v for k, v in pending.meta.items() if k.endswith(("run_id", "call_id"))}
        await store.decide(pending.meta[APPROVAL_ID_META], ApprovalDecision(approved=True))
    async with url_client(server) as client:
        done = await client.call_tool("pay", {"amount": 4}, meta=keys)  # type: ignore[arg-type]
    assert (done.is_error, done.structured_content) == (False, {"result": "paid 4"})
    assert log == ["pay 4"]


async def test_anonymous_http_style_callers_get_no_tasks():
    log: list[str] = []
    store = MemoryApprovalStore()
    server = make_server(log, store, principal=None, allow_anonymous=False)
    async with url_client(server) as client:
        pending = await client.call_tool("pay", {"amount": 1})
        assert pending.content[0].text.startswith("approval required: ")  # type: ignore[union-attr]
        with pytest.raises(MCPError, match=UNKNOWN_TASK):
            await get_task(client, "x")


async def test_a_mismatched_mcp_name_header_is_refused():
    from types import SimpleNamespace

    from agent_adapter.integrations.mcp._task_service import TaskService

    service = TaskService(TaskOptions(), principal=None, allow_anonymous=True)
    caps = t.ClientCapabilities(extensions={"io.modelcontextprotocol/tasks": {}})
    ctx = SimpleNamespace(
        protocol_version="2026-07-28",
        session=SimpleNamespace(client_capabilities=caps),
        request=SimpleNamespace(headers={"mcp-name": "other"}),
    )
    with pytest.raises(MCPError, match="Mcp-Name"):
        await service.get(ctx, SimpleNamespace(task_id="task"))  # type: ignore[arg-type]
