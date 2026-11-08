"""One policy, every surface: the tool loop, the HTTP service and app, and the MCP server."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.messages import Message
from agent_adapter.exec import Executor, run_tool_loop
from agent_adapter.policy import load_policy
from agent_adapter.server.auth import bearer_auth
from tests.fakes.models import ScriptedModel, calls, text
from tests.policy.helpers import shop
from tests.server.helpers import TOKENS, invoke, make_service

POLICY = {
    "rules": [
        {"tools": "wipe_orders", "allow": False, "reason": "never from agents"},
        {"tools": "refund", "confirm": "amount > 100"},
    ]
}
SMALL = {"order_id": "o-1", "amount": 5.0}
BIG = {"order_id": "o-1", "amount": 500.0}


def governed(**options: Any) -> Executor:
    return load_policy(POLICY).apply(Executor(shop(), **options))


async def test_tool_loop_obeys_the_policy() -> None:
    log: list[str] = []
    model = ScriptedModel(
        [calls(("wipe_orders", {}), ("refund", SMALL)), text("done")],
    )
    executor = load_policy(POLICY).apply(Executor(shop(log)))
    result = await run_tool_loop(
        model, executor.registry, [Message(role="user", content="tidy up")], executor=executor
    )
    kinds = [(r.name, r.status, r.error.kind if r.error else "") for r in result.tool_results]
    assert kinds == [("wipe_orders", "error", "approval-denied"), ("refund", "ok", "")]
    assert log == ["refund o-1 5.0"]


async def test_tool_loop_pauses_on_a_policy_approval() -> None:
    store = MemoryApprovalStore()
    executor = governed(approval=DeferredApproval(store))
    model = ScriptedModel([calls(("refund", BIG))])
    result = await run_tool_loop(
        model, executor.registry, [Message(role="user", content="refund")], executor=executor
    )
    assert result.finish == "pending"
    assert [r.status for r in result.tool_results] == ["pending"]


async def test_http_service_obeys_the_policy() -> None:
    service = make_service(shop(), executor=governed())
    denied = await invoke(service, "wipe_orders", {})
    assert denied.status == 403
    assert (await invoke(service, "refund", SMALL)).status == 200
    parked = await invoke(service, "refund", BIG)
    assert parked.status == 202


async def test_create_app_obeys_the_policy() -> None:
    pytest.importorskip("starlette")
    httpx2 = pytest.importorskip("httpx2")
    from agent_adapter.server import create_app

    app = create_app(shop(), executor=governed(), authenticate=bearer_auth(TOKENS), mcp=False)
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
            headers={"Authorization": "Bearer t-alice"},
        ) as http,
    ):
        denied = await http.post("/tools/wipe_orders/invoke", json={"arguments": {}})
        assert denied.status_code == 403
        ok = await http.post("/tools/refund/invoke", json={"arguments": SMALL})
        assert ok.status_code == 200
        parked = await http.post("/tools/refund/invoke", json={"arguments": BIG})
        assert parked.status_code == 202


async def test_mcp_server_obeys_the_policy() -> None:
    pytest.importorskip("mcp")
    from mcp import Client

    from agent_adapter.integrations.mcp import APPROVAL_ID_META, ERROR_KIND_META, mcp_server

    store = MemoryApprovalStore()
    server = mcp_server(shop(), executor=governed(approval=DeferredApproval(store)))
    async with Client(server) as client:
        denied = await client.call_tool("wipe_orders", {})
        small = await client.call_tool("refund", SMALL)
        big = await client.call_tool("refund", BIG)
    assert denied.is_error
    assert denied.meta is not None and denied.meta[ERROR_KIND_META] == "approval-denied"
    assert not small.is_error
    assert big.is_error and big.meta is not None
    assert big.meta[ERROR_KIND_META] == "approval-pending"
    assert big.meta[APPROVAL_ID_META]
