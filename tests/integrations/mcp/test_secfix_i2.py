"""I2: the standalone MCP HTTP app never serves unauthenticated callers on a public host."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("mcp")

import httpx2
from mcp import Client
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

PUBLIC: dict[str, Any] = {
    "host": "0.0.0.0",
    "transport_security": TransportSecuritySettings(allowed_hosts=["tools.example"]),
}
URL = "http://tools.example/mcp"
AUTH = AuthSettings(issuer_url="https://auth.example", resource_server_url=None)
TOKEN = AccessToken(token="tok-a", client_id="alice", scopes=[])
NO_AUTH = "any client that reaches the port can call every tool"
CALL = {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}


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
async def serving(app: Any, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    def asgi_client(headers: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), headers=dict(headers))

    monkeypatch.setattr(client_module, "_new_http_client", asgi_client)
    async with app.router.lifespan_context(app):
        yield


def test_public_host_without_auth_is_a_config_error() -> None:
    ran: list[str] = []
    reg = make_registry(ran)
    with pytest.raises(ConfigError, match="no auth"):
        streamable_http_app(reg, executor=Executor(reg), **PUBLIC)


def test_public_host_with_principal_none_and_no_auth_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="no auth"):
        streamable_http_app(Registry(), principal=None, **PUBLIC)


@pytest.mark.parametrize("half", [{"auth": AUTH}, {"token_verifier": Verifier()}])
@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_auth_and_token_verifier_alone_are_config_errors(half: dict[str, Any], host: str) -> None:
    options = {**PUBLIC, "host": host, **half}
    with pytest.raises(ConfigError, match="go together"):
        streamable_http_app(Registry(), **options)


async def test_public_host_with_auth_refuses_requests_without_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[str] = []
    app = streamable_http_app(make_registry(ran), auth=AUTH, token_verifier=Verifier(), **PUBLIC)
    params = {"name": "delete_all", "arguments": {"path": "/"}}
    headers = {"Accept": "application/json, text/event-stream"}
    bearer = {"Authorization": f"Bearer {TOKEN.token}"}
    async with serving(app, monkeypatch):
        async with client_module._new_http_client(headers) as http:
            refused = await http.post(URL, json={**CALL, "params": params})
        assert (refused.status_code, ran) == (401, [])
        async with Client(mcp_client(URL, headers=bearer).server) as client:
            done = await client.call_tool("delete_all", {"path": "/tmp/x"})
    assert (done.is_error, ran) == (False, ["/tmp/x"])


def test_public_host_with_a_custom_principal_alone_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="no auth"):
        streamable_http_app(Registry(), principal=lambda ctx: "svc", **PUBLIC)


def test_public_host_with_a_gated_principal_is_accepted(caplog: pytest.LogCaptureFixture) -> None:
    gated = GatedPrincipal(lambda ctx: "svc")
    with caplog.at_level(logging.WARNING):
        assert streamable_http_app(Registry(), principal=gated, **PUBLIC)
    assert NO_AUTH not in caplog.text


def test_public_host_with_explicit_anonymous_opt_in_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert streamable_http_app(Registry(), allow_anonymous=True, **PUBLIC)
    assert NO_AUTH in caplog.text
    assert "0.0.0.0" in caplog.text


def test_local_host_without_auth_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert streamable_http_app(Registry(), name="tax-tools")
    assert NO_AUTH in caplog.text
    assert "tax-tools" in caplog.text


def test_local_host_with_auth_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert streamable_http_app(Registry(), auth=AUTH, token_verifier=Verifier())
    assert NO_AUTH not in caplog.text
