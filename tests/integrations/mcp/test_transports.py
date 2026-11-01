"""Round trips over every transport: legacy JSON-RPC, stateless Streamable HTTP, stdio."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

import httpx2
import mcp.types as t
from mcp import Client, StdioServerParameters
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.integrations.mcp import (
    ERROR_KIND_META,
    import_mcp,
    mcp_client,
    mcp_server,
    streamable_http_app,
)
from agent_adapter.integrations.mcp import client as client_module
from agent_adapter.integrations.mcp._scope import scoped_id

pytestmark = pytest.mark.contract

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
BASE = "http://127.0.0.1:8000"


class Leg(BaseModel):
    origin: str
    stops: list[str] = []


def make_registry() -> Registry:
    reg = Registry()

    @reg.tool(effects="read_only")
    async def add(a: int, b: int = 1) -> int:
        """Add two integers."""
        return a + b

    @reg.tool
    async def plan(legs: list[Leg]) -> int:
        """Count the stops of a nested itinerary."""
        return sum(len(leg.stops) for leg in legs)

    @reg.tool
    async def whoami(ctx: ToolContext) -> dict[str, Any]:
        """The run id and the request's custom header."""
        request = ctx.extras["mcp_request"].request
        return {"run_id": ctx.run_id, "team": request.headers.get("x-team")}

    @reg.tool(confirm=True)
    async def delete(item: str) -> str:
        """Delete an item (needs approval)."""
        return f"deleted {item}"

    return reg


@asynccontextmanager
async def serving(app: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Run the app's lifespan; `mcp_client` URLs reach it through `httpx2.ASGITransport`."""

    def asgi_client(headers: Mapping[str, str]) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), headers=dict(headers))

    monkeypatch.setattr(client_module, "_new_http_client", asgi_client)
    async with app.router.lifespan_context(app):
        yield


async def exercise(client: Client) -> None:
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert tools["add"].annotations == t.ToolAnnotations(
        read_only_hint=True, destructive_hint=False
    )
    ok = await client.call_tool("add", {"a": 2, "b": 3})
    assert (ok.is_error, ok.structured_content) == (False, {"result": 5})
    bad = await client.call_tool("add", {"a": "x"})
    assert bad.is_error and bad.meta is not None
    assert bad.meta[ERROR_KIND_META] == "tool-args-invalid"


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_in_process_round_trip(mode: str):
    async with Client(mcp_server(make_registry()), mode=mode) as client:
        await exercise(client)


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_streamable_http_round_trip(mode: str, monkeypatch: pytest.MonkeyPatch):
    app = streamable_http_app(make_registry(), allow_anonymous=True)
    async with serving(app, monkeypatch):
        client = mcp_client(f"{BASE}/mcp", headers={"X-Run-Id": "run_http", "X-Team": "tax"})
        async with Client(client.server, mode=mode) as connected:
            await exercise(connected)
            seen = await connected.call_tool("whoami", {})
    assert seen.structured_content == {
        "run_id": scoped_id("run_", None, "run_http"),
        "team": "tax",
    }


async def test_http_app_is_stateless_and_answers_json(monkeypatch: pytest.MonkeyPatch):
    app = streamable_http_app(make_registry())
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "probe", "version": "1"},
        },
    }
    headers = {"Accept": "application/json, text/event-stream"}
    async with serving(app, monkeypatch), client_module._new_http_client(headers) as http:
        response = await http.post(f"{BASE}/mcp", json=initialize)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "mcp-session-id" not in response.headers
    assert response.json()["result"]["serverInfo"]["name"] == "agent-adapter"


async def test_http_options_and_path_reach_the_app(monkeypatch: pytest.MonkeyPatch):
    open_hosts = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = streamable_http_app(make_registry(), path="/tools/mcp", transport_security=open_hosts)
    async with serving(app, monkeypatch), mcp_client("http://tools.example/tools/mcp") as client:
        assert (await client.call_tool("add", {"a": 1})).structured_content == {"result": 2}
    local_only = streamable_http_app(make_registry())
    async with serving(local_only, monkeypatch), client_module._new_http_client({}) as http:
        response = await http.post("http://tools.example/mcp", json={})
    assert response.status_code == 421  # DNS-rebinding protection: localhost hosts only


async def test_client_approval_over_stateless_http(monkeypatch: pytest.MonkeyPatch):
    async def approve(ctx: Any, params: Any) -> t.ElicitResult:
        return t.ElicitResult(action="accept", content={"approved": True})

    app = streamable_http_app(make_registry(), executor=Executor(), client_approval=True)
    async with serving(app, monkeypatch):
        transport = mcp_client(f"{BASE}/mcp").server
        async with Client(transport, elicitation_callback=approve) as client:
            done = await client.call_tool("delete", {"item": "a"})
    assert (done.is_error, done.structured_content) == (False, {"result": "deleted a"})


async def test_import_over_http_keeps_nested_schemas_verbatim(monkeypatch: pytest.MonkeyPatch):
    app = streamable_http_app(make_registry())
    async with serving(app, monkeypatch), mcp_client(f"{BASE}/mcp") as client:
        listed = {tool.name: tool for tool in (await client.list_tools()).tools}
        local = Registry()
        await import_mcp(local, client, names=["plan"])
        assert local.get("plan").json_schema() == listed["plan"].input_schema
        legs = [{"origin": "BOM", "stops": ["GOI", "COK"]}, {"origin": "DEL"}]
        assert await local.get("plan").invoke({"legs": legs}) == 2


async def test_stdio_subprocess_round_trip():
    params = StdioServerParameters(
        command=sys.executable, args=[str(EXAMPLES / "mcp_server.py"), "--stdio"]
    )
    async with mcp_client(params) as client:
        assert client.server_info.name == "shop-tools"
        local = Registry()
        await import_mcp(local, client)
        assert local.names() == ["shipping_rate", "send_invoice"]
        assert await local.get("shipping_rate").invoke({"region": "eu"}) == 5.0
