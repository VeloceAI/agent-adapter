"""A local MCP HTTP app without auth serves loopback peers only; explicit opt-outs still work."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp")

import httpx2
from mcp import Client
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings

from agent_adapter.core.registry import Registry
from agent_adapter.integrations.mcp import client as client_module
from agent_adapter.integrations.mcp import mcp_client, streamable_http_app
from agent_adapter.integrations.mcp._peer import is_local_peer
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
CALL = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "delete_all", "arguments": {"path": "/"}},
}
ACCEPT = {"Accept": "application/json, text/event-stream"}


class Verifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        return TOKEN if token == TOKEN.token else None


def make_registry(ran: list[str]) -> Registry:
    reg = Registry()

    @reg.tool
    async def delete_all(path: str) -> str:
        """Delete everything under a path."""
        ran.append(path)
        return "deleted"

    return reg


@asynccontextmanager
async def serving(
    app: Any, monkeypatch: pytest.MonkeyPatch, peer: str = "127.0.0.1"
) -> AsyncIterator[None]:
    """Serve `app` in-process; every request comes from `peer`."""

    def asgi_client(headers: Any) -> httpx2.AsyncClient:
        transport = httpx2.ASGITransport(app=app, client=(peer, 5000))
        return httpx2.AsyncClient(transport=transport, headers=dict(headers))

    monkeypatch.setattr(client_module, "_new_http_client", asgi_client)
    async with app.router.lifespan_context(app):
        yield


async def post_call(url: str, headers: dict[str, str] | None = None) -> httpx2.Response:
    async with client_module._new_http_client({**ACCEPT, **(headers or {})}) as http:
        return await http.post(url, json=CALL)


async def remote_call(
    app: Any, monkeypatch: pytest.MonkeyPatch, url: str, headers: dict[str, str] | None = None
) -> bool:
    """Call `delete_all` from a remote peer; True when the call succeeded."""
    async with (
        serving(app, monkeypatch, peer=REMOTE),
        Client(mcp_client(url, headers=headers).server) as client,
    ):
        done = await client.call_tool("delete_all", {"path": "/x"})
    return not done.is_error


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.8.9.1", "::1", "::ffff:127.0.0.1", None, "", "testclient"]
)
def test_loopback_and_missing_peers_are_local(host: str | None) -> None:
    assert is_local_peer(host)


@pytest.mark.parametrize(
    "host", [REMOTE, "10.0.0.5", "::ffff:10.0.0.1", "fe80::1", "localhost", "evil", "0.0.0.0"]
)
def test_other_peers_are_not_local(host: str) -> None:
    assert not is_local_peer(host)


@pytest.mark.parametrize("anonymous", [False, True])
async def test_a_local_app_without_auth_refuses_a_remote_peer(
    anonymous: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ran: list[str] = []
    app = streamable_http_app(make_registry(ran), allow_anonymous=anonymous)
    with caplog.at_level(logging.WARNING):
        async with serving(app, monkeypatch, peer=REMOTE):
            refused = await post_call(LOCAL_URL, {"Host": "127.0.0.1:8000"})
    assert refused.status_code == 403
    assert refused.json()["error"] == "non-local-client"
    assert ran == []
    assert f"non-local client {REMOTE}" in caplog.text


async def test_a_local_app_without_auth_serves_a_loopback_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[str] = []
    app = streamable_http_app(make_registry(ran))
    async with serving(app, monkeypatch), Client(mcp_client(LOCAL_URL).server) as client:
        done = await client.call_tool("delete_all", {"path": "/tmp/x"})
    assert (done.is_error, ran) == (False, ["/tmp/x"])


def test_starlettes_test_client_counts_as_local() -> None:
    from starlette.testclient import TestClient

    app = streamable_http_app(make_registry([]))
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
    with TestClient(app, base_url="http://127.0.0.1:8000") as http:
        response = http.post("/mcp", json=initialize, headers=ACCEPT)
    assert response.status_code == 200


async def test_a_public_app_with_explicit_anonymous_opt_in_serves_remote_peers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[str] = []
    app = streamable_http_app(make_registry(ran), allow_anonymous=True, **PUBLIC)
    assert await remote_call(app, monkeypatch, PUBLIC_URL)
    assert ran == ["/x"]


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
async def test_an_app_with_auth_has_no_peer_guard(
    host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []
    options = PUBLIC if host == "0.0.0.0" else {}
    app = streamable_http_app(make_registry(ran), auth=AUTH, token_verifier=Verifier(), **options)
    url = LOCAL_URL if host == "127.0.0.1" else PUBLIC_URL
    bearer = {"Authorization": f"Bearer {TOKEN.token}"}
    assert await remote_call(app, monkeypatch, url, bearer)
    assert ran == ["/x"]


async def test_an_app_with_a_gated_principal_has_no_peer_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[str] = []
    app = streamable_http_app(make_registry(ran), principal=GatedPrincipal(lambda ctx: "svc"))
    assert await remote_call(app, monkeypatch, LOCAL_URL)
    assert ran == ["/x"]


async def test_a_remote_websocket_is_closed_at_the_handshake() -> None:
    app = streamable_http_app(make_registry([]))
    sent: list[MutableMapping[str, Any]] = []

    async def receive() -> MutableMapping[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "websocket", "path": "/mcp", "client": (REMOTE, 1), "headers": []}
    await app(scope, receive, send)
    assert sent == [{"type": "websocket.close", "code": 1008}]
