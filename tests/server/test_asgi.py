"""The Starlette app: routes over real ASGI, prefixes, body limits, SSE and the MCP mount."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx2")

import httpx2

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.tool import ToolSpec
from agent_adapter.server import create_app
from agent_adapter.server.asgi import mcp_principal
from agent_adapter.server.auth import Principal, bearer_auth
from tests.server.helpers import TOKENS, make_registry

BASE = "http://127.0.0.1:8000"
EMAIL = {"to": "a@example.com", "body": "hi"}


def app_for(**options: Any) -> Any:
    options.setdefault("authenticate", bearer_auth(TOKENS))
    options.setdefault("mcp", False)
    return create_app(make_registry(), **options)


@asynccontextmanager
async def client(app: Any, token: str | None = "t-alice") -> AsyncIterator[Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url=BASE, headers=headers
        ) as http,
    ):
        yield http


async def test_routes_answer_over_asgi():
    async with client(app_for()) as http:
        health = await http.get("/healthz")
        assert health.status_code == 200 and health.json()["status"] == "ok"
        ok = await http.post("/tools/add/invoke", json={"arguments": {"a": 2, "b": 3}})
        assert ok.status_code == 200 and ok.json()["result"] == 5
        bad = await http.post("/tools/add/invoke", json={"arguments": {"a": "x"}})
        assert bad.status_code == 422
        assert bad.headers["content-type"] == "application/problem+json"
        catalogue = await http.get("/tools")
        etag = catalogue.headers["etag"]
        cached = await http.get("/tools", headers={"If-None-Match": etag})
        assert cached.status_code == 304 and cached.content == b""
        assert "content-type" not in cached.headers


async def test_unauthenticated_requests_get_401_problems():
    async with client(app_for(), token=None) as http:
        reply = await http.get("/tools")
        assert reply.status_code == 401 and reply.headers["www-authenticate"] == "Bearer"


async def test_a_prefix_moves_every_route_and_the_location_header():
    async with client(app_for(prefix="/api")) as http:
        assert (await http.get("/healthz")).status_code == 404
        parked = await http.post("/api/tools/send_email/invoke", json={"arguments": EMAIL})
        assert parked.status_code == 202
        location = parked.headers["location"]
        assert location == f"/api/approvals/{parked.json()['approval_id']}"
        assert (await http.get(location)).json()["status"] == "pending"


async def test_bodies_over_the_limit_are_refused_without_buffering():
    app = app_for()
    app.state.service.max_body_bytes = 32
    async with client(app) as http:
        declared = await http.post("/tools/add/invoke", content=b"x" * 64)
        assert declared.status_code == 413

        async def chunks() -> AsyncIterator[bytes]:
            for _ in range(8):
                yield b"x" * 8

        streamed = await http.post("/tools/add/invoke", content=chunks())
        assert streamed.status_code == 413


async def test_sse_streams_through_starlette():
    async with client(app_for()) as http:
        await http.post("/tools/add/invoke", json={"arguments": {"a": 1, "b": 1}, "run_id": "r1"})
        reply = await http.get("/runs/r1/events", params={"close_on": "tool.finished"})
        assert reply.headers["content-type"].startswith("text/event-stream")
        assert reply.text.endswith("event: end\ndata: {}\n\n")
        assert "event: tool.finished" in reply.text


async def test_unencodable_bodies_become_500(monkeypatch: pytest.MonkeyPatch):
    from agent_adapter.server import asgi

    real = asgi.encode_body
    calls = iter([ValueError("nan"), None])

    def flaky(response: Any) -> bytes:
        error = next(calls)
        if error is not None:
            raise error
        return real(response)

    monkeypatch.setattr(asgi, "encode_body", flaky)
    async with client(app_for()) as http:
        reply = await http.get("/healthz")
        assert reply.status_code == 500 and reply.json()["detail"] == "unencodable response"


@pytest.mark.parametrize(
    "options", [{"prefix": "api"}, {"prefix": "/api/"}, {"mcp_path": ""}, {"mcp_path": "/{x}"}]
)
def test_bad_paths_fail_fast(options: dict[str, Any]):
    with pytest.raises(ConfigError):
        app_for(**options)


def test_mcp_principal_needs_the_gate():
    assert mcp_principal(None) is None


class TestMcpMount:
    """`/mcp` sits behind the same authenticator, `authorize` and approval store."""

    URL = f"{BASE}/mcp"

    @pytest.fixture(autouse=True)
    def _route_mcp_clients(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("mcp")
        from agent_adapter.integrations.mcp import client as client_module

        self.app: Any = None

        def asgi_client(headers: Mapping[str, str]) -> Any:
            return httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=self.app), headers=dict(headers)
            )

        monkeypatch.setattr(client_module, "_new_http_client", asgi_client)

    def build(self, **options: Any) -> Any:
        self.app = app_for(mcp=True, **options)
        return self.app

    async def test_mcp_calls_need_a_token(self):
        app = self.build()
        async with client(app, token=None) as http:
            reply = await http.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
            assert reply.status_code == 401
            assert reply.json()["type"] == "urn:agent-adapter:error:unauthorized"

    async def test_mcp_calls_run_as_the_principal_and_share_the_approval_store(self):
        from mcp import Client

        from agent_adapter.integrations.mcp import CALL_ID_META, RUN_ID_META, mcp_client

        app = self.build()
        async with client(app) as http:
            transport = mcp_client(self.URL, headers={"Authorization": "Bearer t-alice"}).server
            async with Client(transport) as mcp:
                who = await mcp.call_tool("whoami", {}, meta={RUN_ID_META: "r1"})
                seen = who.structured_content
                assert seen["principal"] == "alice@example.com"
                pending = await mcp.call_tool("send_email", EMAIL, meta={RUN_ID_META: "r1"})
                assert pending.is_error and pending.meta is not None
                approval_id = pending.meta["agent-adapter/approval_id"]
                listed = await http.get("/approvals", params={"run_id": "r1"})
                assert [a["approval_id"] for a in listed.json()["approvals"]] == [approval_id]
                decided = await http.post(
                    f"/approvals/{approval_id}/decide", json={"approved": True}
                )
                assert decided.status_code == 202 and decided.json()["status"] == "approved"
                resumed = await mcp.call_tool(
                    "send_email",
                    EMAIL,
                    meta={RUN_ID_META: "r1", CALL_ID_META: pending.meta[CALL_ID_META]},
                )
                assert not resumed.is_error
            events = await http.get("/runs/r1/events", params={"close_on": "tool.finished"})
            assert "event: tool.finished" in events.text
            bob = await http.get(
                "/approvals", params={"run_id": "r1"}, headers={"Authorization": "Bearer t-bob"}
            )
            assert bob.json()["approvals"] == []

    async def test_authorize_applies_to_mcp_calls(self):
        from mcp import Client

        from agent_adapter.integrations.mcp import mcp_client

        async def only_add(principal: Principal, spec: ToolSpec, args: Any) -> bool:
            return spec.name == "add"

        app = self.build(authorize=only_add)
        async with client(app):
            transport = mcp_client(self.URL, headers={"Authorization": "Bearer t-bob"}).server
            async with Client(transport) as mcp:
                ok = await mcp.call_tool("add", {"a": 1, "b": 2})
                assert ok.structured_content == {"result": 3}
                refused = await mcp.call_tool("whoami", {})
                assert refused.is_error
                assert refused.meta["agent-adapter/error_kind"] == "forbidden"

    async def test_mcp_listings_show_only_tools_authorize_allows(self):
        from mcp import Client

        from agent_adapter.integrations.mcp import mcp_client

        async def bob_only_add(principal: Principal, spec: ToolSpec, args: Any) -> bool:
            return principal.id != "bob@example.com" or spec.name == "add"

        app = self.build(authorize=bob_only_add)
        async with client(app):
            for token, expect_all in (("t-bob", False), ("t-alice", True)):
                headers = {"Authorization": f"Bearer {token}"}
                async with Client(mcp_client(self.URL, headers=headers).server) as mcp:
                    names = [tool.name for tool in (await mcp.list_tools()).tools]
                    assert (names == ["add"]) is not expect_all
                    assert "add" in names

    async def test_a_raising_authorize_hides_tools_from_mcp_listings(self):
        from mcp import Client

        from agent_adapter.integrations.mcp import mcp_client

        async def broken(principal: Principal, spec: ToolSpec, args: Any) -> bool:
            raise RuntimeError("down")

        app = self.build(authorize=broken)
        async with client(app):
            headers = {"Authorization": "Bearer t-bob"}
            async with Client(mcp_client(self.URL, headers=headers).server) as mcp:
                assert (await mcp.list_tools()).tools == []

    async def test_mcp_options_reach_the_mcp_app(self):
        with pytest.raises(ConfigError, match="transport_security"):
            self.build(mcp_options={"host": "0.0.0.0"})


def test_serve_runs_uvicorn(monkeypatch: pytest.MonkeyPatch):
    uvicorn = pytest.importorskip("uvicorn")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: seen.update(host=host, port=port))
    from agent_adapter.server import serve

    serve(object(), port=8123)
    assert seen == {"host": "127.0.0.1", "port": 8123}


@pytest.mark.parametrize("module", ["starlette", "uvicorn"])
def test_missing_server_dependencies(monkeypatch: pytest.MonkeyPatch, module: str):
    import sys

    from agent_adapter.core.errors import MissingDependency
    from agent_adapter.server import asgi

    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[server\]"):
        asgi._require(module)


async def test_the_mcp_gate_fails_closed_when_the_authenticator_breaks():
    pytest.importorskip("mcp")

    async def broken(request: Any) -> Any:
        raise RuntimeError("t-alice")

    app = app_for(mcp=True, authenticate=broken)
    async with client(app) as http:
        reply = await http.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert reply.status_code == 500 and "t-alice" not in reply.text


async def test_the_mcp_runner_refuses_calls_without_a_gate_principal():
    from agent_adapter.core.context import ToolContext
    from agent_adapter.core.errors import Forbidden
    from agent_adapter.server.asgi import _AuthorizedRunner

    app = app_for()
    runner = _AuthorizedRunner(app.state.service)
    spec = app.state.service.registry.get("add")
    with pytest.raises(Forbidden):
        await runner.run_spec(spec, {"a": 1, "b": 1}, ToolContext())
    with pytest.raises(Forbidden):
        await runner.run_spec(spec, {"a": 1, "b": 1}, None)


def test_a_public_bind_host_reaches_the_mcp_mount():
    pytest.importorskip("mcp")
    with pytest.raises(ConfigError, match="transport_security"):
        app_for(mcp=True, host="0.0.0.0")
    app_for(mcp=False, host="0.0.0.0")  # REST only: no MCP transport settings needed
    app_for(mcp=True, host="localhost")
