"""A local anonymous HTTP server serves loopback peers only (ASGI and WSGI); opt-outs still work."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from wsgiref.util import setup_testing_defaults

import pytest

from agent_adapter.server.auth import bearer_auth
from tests.server.helpers import TOKENS, make_registry

pytest.importorskip("starlette")
pytest.importorskip("httpx2")

import httpx2

from agent_adapter.server import create_app

BASE = "http://127.0.0.1:8000"
REMOTE = "203.0.113.9"
NON_LOCAL = "urn:agent-adapter:error:non-local-client"
ACCEPT = {"Accept": "application/json, text/event-stream"}


@asynccontextmanager
async def client(
    app: Any, peer: str = "127.0.0.1", token: str | None = None
) -> AsyncIterator[httpx2.AsyncClient]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    transport = httpx2.ASGITransport(app=app, client=(peer, 5000))
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(transport=transport, base_url=BASE, headers=headers) as http,
    ):
        yield http


def anonymous_app(**options: Any) -> Any:
    options.setdefault("mcp", False)
    return create_app(make_registry(), allow_anonymous=True, **options)


@pytest.mark.parametrize("mcp", [False, True])
async def test_a_local_anonymous_app_refuses_a_remote_peer(
    mcp: bool, caplog: pytest.LogCaptureFixture
) -> None:
    if mcp:
        pytest.importorskip("mcp")
    app = anonymous_app(mcp=mcp)
    with caplog.at_level(logging.WARNING):
        async with client(app, peer=REMOTE) as http:
            tools = await http.get("/tools")
            invoked = await http.post("/tools/add/invoke", json={"arguments": {"a": 1, "b": 2}})
            over_mcp = await http.post("/mcp", json={}, headers=ACCEPT)
    for refused in (tools, invoked, over_mcp):
        assert refused.status_code == 403
        assert refused.headers["content-type"] == "application/problem+json"
        assert refused.json()["type"] == NON_LOCAL
    assert f"non-local client {REMOTE}" in caplog.text


async def test_a_local_anonymous_app_serves_loopback_peers() -> None:
    for peer in ("127.0.0.1", "::1"):
        async with client(anonymous_app(), peer=peer) as http:
            assert (await http.get("/tools")).status_code == 200


def test_starlettes_test_client_counts_as_local() -> None:
    from starlette.testclient import TestClient

    with TestClient(anonymous_app(), base_url=BASE) as http:
        assert http.get("/tools").status_code == 200


async def test_the_public_opt_out_serves_remote_peers() -> None:
    app = anonymous_app(allow_anonymous_public=True)
    async with client(app, peer=REMOTE) as http:
        assert (await http.get("/tools")).status_code == 200


async def test_an_authenticated_app_has_no_peer_guard() -> None:
    app = create_app(make_registry(), authenticate=bearer_auth(TOKENS), mcp=False)
    async with client(app, peer=REMOTE, token="t-alice") as http:
        assert (await http.get("/tools")).status_code == 200


def wsgi_get(app: Any, path: str, **environ: str) -> tuple[str, dict[str, str], bytes]:
    env: dict[str, Any] = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": path,
        "SERVER_NAME": "127.0.0.1",
        "SERVER_PORT": "8000",
        "HTTP_HOST": "127.0.0.1:8000",
        "wsgi.input": io.BytesIO(b""),
        **environ,
    }
    setup_testing_defaults(env)
    seen: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]], exc_info: Any = None) -> Any:
        seen["status"], seen["headers"] = status, dict(headers)
        return lambda data: None

    content = b"".join(app(env, start_response))
    return seen["status"], seen["headers"], content


@pytest.mark.parametrize(
    "remote", [{"REMOTE_ADDR": REMOTE}, {"REMOTE_ADDR": REMOTE, "REMOTE_PORT": "1"}]
)
def test_wsgi_refuses_a_remote_addr(
    remote: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("a2wsgi")
    from agent_adapter.server import create_wsgi_app

    with (
        create_wsgi_app(make_registry(), allow_anonymous=True, mcp=False) as app,
        caplog.at_level(logging.WARNING),
    ):
        status, headers, content = wsgi_get(app, "/tools", **remote)
    assert status == "403 Forbidden"
    assert headers["Content-Type"] == "application/problem+json"
    assert json.loads(content)["type"] == NON_LOCAL
    assert f"non-local client {REMOTE}" in caplog.text


@pytest.mark.parametrize("local", [{}, {"REMOTE_ADDR": "127.0.0.1", "REMOTE_PORT": "1"}])
def test_wsgi_serves_local_or_missing_remote_addr(local: dict[str, str]) -> None:
    pytest.importorskip("a2wsgi")
    from agent_adapter.server import create_wsgi_app

    with create_wsgi_app(make_registry(), allow_anonymous=True, mcp=False) as app:
        status, _, _ = wsgi_get(app, "/tools", **local)
    assert status.startswith("200")


def test_wsgi_opt_outs_are_not_wrapped() -> None:
    pytest.importorskip("a2wsgi")
    from agent_adapter.server import create_wsgi_app
    from agent_adapter.server.wsgi import LoopbackOnlyWsgi

    with (
        create_wsgi_app(
            make_registry(), allow_anonymous=True, allow_anonymous_public=True, mcp=False
        ) as public,
        create_wsgi_app(make_registry(), authenticate=bearer_auth(TOKENS), mcp=False) as authed,
    ):
        assert not isinstance(public, LoopbackOnlyWsgi)
        assert not isinstance(authed, LoopbackOnlyWsgi)
        status, _, _ = wsgi_get(public, "/tools", REMOTE_ADDR=REMOTE, REMOTE_PORT="1")
    assert status.startswith("200")
