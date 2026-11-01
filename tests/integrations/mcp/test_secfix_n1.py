"""N1: a custom MCP `principal=` hook is not auth on its own; a None principal is refused."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest

pytest.importorskip("mcp")

import httpx2
from mcp import Client, MCPError
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.integrations.mcp import client as client_module
from agent_adapter.integrations.mcp import mcp_client, streamable_http_app
from agent_adapter.integrations.mcp._scope import GatedPrincipal

pytestmark = pytest.mark.contract

REMOTE = "203.0.113.9"
LOCAL_URL = "http://127.0.0.1:8000/mcp"
PUBLIC_URL = "http://tools.example/mcp"
PUBLIC: dict[str, Any] = {
    "host": "0.0.0.0",
    "transport_security": TransportSecuritySettings(allowed_hosts=["tools.example"]),
}
AUTH = AuthSettings(issuer_url="https://auth.example", resource_server_url=None)
TOKEN = AccessToken(token="tok-a", client_id="alice", scopes=[])
NO_AUTH = "any client that reaches the port can call every tool"


class Verifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        return TOKEN if token == TOKEN.token else None


def header_principal(ctx: Any) -> str | None:
    """A natural hook: the X-User header, or None when it is missing."""
    request = getattr(ctx, "request", None)
    return request.headers.get("x-user") if request is not None else None


def make_registry(ran: list[str]) -> Registry:
    reg = Registry()

    @reg.tool
    async def delete_all(path: str) -> str:
        """Delete everything under a path."""
        ran.append(path)
        return "deleted"

    return reg


@asynccontextmanager
async def connected(
    app: Any,
    monkeypatch: pytest.MonkeyPatch,
    url: str = LOCAL_URL,
    peer: str = "127.0.0.1",
    headers: dict[str, str] | None = None,
) -> AsyncIterator[Client]:
    def asgi_client(extra: Any) -> httpx2.AsyncClient:
        transport = httpx2.ASGITransport(app=app, client=(peer, 5000))
        return httpx2.AsyncClient(transport=transport, headers=dict(extra))

    monkeypatch.setattr(client_module, "_new_http_client", asgi_client)
    async with (
        app.router.lifespan_context(app),
        Client(mcp_client(url, headers=headers).server) as client,
    ):
        yield client


def test_a_custom_hook_on_a_public_host_without_auth_is_a_config_error() -> None:
    ran: list[str] = []
    reg = make_registry(ran)
    with pytest.raises(ConfigError, match="no auth"):
        streamable_http_app(reg, executor=Executor(reg), principal=header_principal, **PUBLIC)


def test_a_custom_hook_on_a_local_host_without_auth_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        streamable_http_app(make_registry([]), principal=header_principal)
    assert NO_AUTH in caplog.text


async def test_a_none_principal_is_refused_for_calls_and_listing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ran: list[str] = []
    app = streamable_http_app(make_registry(ran), principal=header_principal)
    with caplog.at_level(logging.WARNING):
        async with connected(app, monkeypatch) as client:
            with pytest.raises(MCPError, match="authentication required"):
                await client.call_tool("delete_all", {"path": "/"})
            with pytest.raises(MCPError, match="authentication required"):
                await client.list_tools()
    assert ran == []
    assert "principal hook named no caller" in caplog.text


async def test_a_named_principal_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    app = streamable_http_app(make_registry(ran), principal=header_principal)
    async with connected(app, monkeypatch, headers={"X-User": "alice"}) as client:
        assert [tool.name for tool in (await client.list_tools()).tools] == ["delete_all"]
        done = await client.call_tool("delete_all", {"path": "/a"})
    assert (done.is_error, ran) == (False, ["/a"])


async def test_a_custom_hook_keeps_the_loopback_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    app = streamable_http_app(make_registry(ran), principal=header_principal)
    headers = {"Accept": "application/json, text/event-stream", "X-User": "alice"}
    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app, client=(REMOTE, 5000))
        async with httpx2.AsyncClient(transport=transport, headers=headers) as http:
            refused = await http.post(LOCAL_URL, json={"jsonrpc": "2.0", "id": 1, "method": "x"})
    assert (refused.status_code, ran) == (403, [])


@pytest.mark.parametrize("options", [{}, PUBLIC])
async def test_allow_anonymous_lets_a_none_principal_through(
    options: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []
    reg = make_registry(ran)
    app = streamable_http_app(reg, principal=header_principal, allow_anonymous=True, **options)
    url = PUBLIC_URL if options else LOCAL_URL
    async with connected(app, monkeypatch, url=url) as client:
        done = await client.call_tool("delete_all", {"path": "/b"})
    assert (done.is_error, ran) == (False, ["/b"])


async def test_a_custom_hook_with_auth_still_refuses_a_none_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[str] = []
    app = streamable_http_app(
        make_registry(ran), principal=header_principal, auth=AUTH, token_verifier=Verifier()
    )
    bearer = {"Authorization": f"Bearer {TOKEN.token}"}
    async with connected(app, monkeypatch, peer=REMOTE, headers=bearer) as client:
        with pytest.raises(MCPError, match="authentication required"):
            await client.call_tool("delete_all", {"path": "/"})
    assert ran == []


async def test_the_gated_marker_counts_as_auth(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ran: list[str] = []
    with caplog.at_level(logging.WARNING):
        app = streamable_http_app(
            make_registry(ran), principal=GatedPrincipal(lambda ctx: None), **PUBLIC
        )
    assert NO_AUTH not in caplog.text
    async with connected(app, monkeypatch, url=PUBLIC_URL, peer=REMOTE) as client:
        done = await client.call_tool("delete_all", {"path": "/c"})
    assert (done.is_error, ran) == (False, ["/c"])


def test_the_gated_marker_delegates_and_reads_well() -> None:
    gated = GatedPrincipal(lambda ctx: "svc")
    assert gated(cast(Any, object())) == "svc"
    assert "GatedPrincipal" in repr(gated)
