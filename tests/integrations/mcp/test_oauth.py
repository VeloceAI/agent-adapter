"""Client-side OAuth for `mcp_client`: CIMD set-up, auth checks, and the auth reaching requests."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Any

import pytest

pytest.importorskip("mcp")

import httpx2
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import AuthorizationCodeResult, OAuthToken

from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.integrations.mcp import (
    MemoryTokenStorage,
    client_metadata_document,
    mcp_client,
    oauth_client,
    streamable_http_app,
)
from agent_adapter.integrations.mcp import client as client_module
from tests.integrations.mcp.tasks_support import make_registry

pytestmark = pytest.mark.contract

CIMD = "https://agent.example/oauth/client.json"
REDIRECT = "http://127.0.0.1:33418/callback"


async def go(url: str) -> None:
    raise AssertionError("no browser in tests")


async def back() -> AuthorizationCodeResult:
    raise AssertionError("no redirect in tests")


def provider(**kw: Any) -> OAuthClientProvider:
    options: dict[str, Any] = {
        "client_metadata_url": CIMD,
        "redirect_uri": REDIRECT,
        "redirect_handler": go,
        "callback_handler": back,
    }
    return oauth_client("https://mcp.example/mcp", **{**options, **kw})


def test_oauth_client_uses_the_client_id_metadata_document():
    auth = provider(scope="tools")
    assert isinstance(auth, OAuthClientProvider)
    assert auth.context.client_metadata_url == CIMD
    assert auth.context.client_metadata.scope == "tools"
    assert [str(uri) for uri in auth.context.client_metadata.redirect_uris or []] == [REDIRECT]
    assert isinstance(auth.context.storage, MemoryTokenStorage)


@pytest.mark.parametrize(
    ("server", "kw"),
    [
        ("http://mcp.example/mcp", {}),
        ("https://mcp.example/mcp", {"client_metadata_url": "https://agent.example/"}),
        ("https://mcp.example/mcp", {"client_metadata_url": "http://agent.example/c.json"}),
        ("https://mcp.example/mcp", {"redirect_uri": ""}),
        ("https://mcp.example/mcp", {"redirect_uri": "not a url"}),
    ],
)
def test_oauth_client_refuses_unsafe_settings(server: str, kw: dict[str, Any]):
    options: dict[str, Any] = {
        "client_metadata_url": CIMD,
        "redirect_uri": REDIRECT,
        "redirect_handler": go,
        "callback_handler": back,
        **kw,
    }
    with pytest.raises(ConfigError):
        oauth_client(server, **options)


def test_the_metadata_document_names_itself_as_client_id():
    document = client_metadata_document(CIMD, redirect_uris=[REDIRECT], scope="a b")
    assert document["client_id"] == CIMD
    assert document["token_endpoint_auth_method"] == "none"
    assert document["scope"] == "a b"
    with pytest.raises(ConfigError):
        client_metadata_document(CIMD, redirect_uris=[])


async def test_memory_storage_keeps_tokens_and_hides_them():
    storage = MemoryTokenStorage()
    assert await storage.get_tokens() is None and await storage.get_client_info() is None
    await storage.set_tokens(OAuthToken(access_token="secret-token", token_type="Bearer"))
    assert (await storage.get_tokens()).access_token == "secret-token"  # type: ignore[union-attr]
    assert "secret" not in repr(storage)


def test_mcp_client_checks_auth():
    auth = provider()
    with pytest.raises(ConfigError, match="https"):
        mcp_client("http://mcp.example/mcp", auth=auth)
    with pytest.raises(ToolUsageError, match="not both"):
        mcp_client("https://mcp.example/mcp", auth=auth, headers={"authorization": "Bearer x"})
    with pytest.raises(ToolUsageError, match=r"httpx2\.Auth"):
        mcp_client("https://mcp.example/mcp", auth="token")  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError, match="auth= only applies"):
        mcp_client(object(), auth=auth)
    mcp_client("http://127.0.0.1:8000/mcp", auth=auth)  # loopback http is allowed


class Stamp(httpx2.Auth):
    """Stands in for OAuth: adds a bearer token and counts the requests it signs."""

    def __init__(self) -> None:
        self.signed = 0

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, Any, None]:
        self.signed += 1
        request.headers["Authorization"] = "Bearer t"
        yield request


async def test_the_auth_signs_every_request(monkeypatch: pytest.MonkeyPatch):
    app = streamable_http_app(make_registry([]))
    seen: list[str | None] = []

    async def record(request: httpx2.Request) -> None:
        seen.append(request.headers.get("authorization"))

    def asgi_client(headers: Mapping[str, str]) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), event_hooks={"request": [record]}
        )

    monkeypatch.setattr(client_module, "_new_http_client", asgi_client)
    stamp = Stamp()
    async with (
        app.router.lifespan_context(app),
        mcp_client("http://127.0.0.1:8000/mcp", auth=stamp) as client,
    ):
        result = await client.call_tool("slow", {"seconds": 0})
    assert result.structured_content == {"result": "done"}
    assert stamp.signed >= 1 and seen and all(value == "Bearer t" for value in seen)


@pytest.mark.parametrize(
    "uri",
    [
        "http://app.example/cb",
        "https:///cb",
        "https://user:pw@app.example/cb",
        "https://app.example/cb#frag",
        "http://127.0.0.1:8765/cb#x",
        "javascript:alert(1)",
        "myapp:/cb",
        "ftp://app.example/cb",
    ],
)
def test_unsafe_redirect_uris_are_refused(uri: str):
    with pytest.raises(ConfigError, match="redirect"):
        client_metadata_document(CIMD, redirect_uris=[uri])
    with pytest.raises(ConfigError, match="redirect"):
        provider(redirect_uri=uri)


@pytest.mark.parametrize(
    "uri",
    [
        "https://app.example/cb",
        "http://127.0.0.1:8765/cb",
        "http://[::1]:8765/cb",
        "http://localhost/cb",
        "com.example.app:/oauth2redirect",
    ],
)
def test_https_loopback_and_private_use_redirects_are_accepted(uri: str):
    assert client_metadata_document(CIMD, redirect_uris=[uri])["redirect_uris"] == [uri]
