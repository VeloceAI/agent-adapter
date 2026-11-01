"""`/readyz`: 200 once the app can serve, 503 before an MCP app starts and while draining."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx2")

import httpx2

from agent_adapter.server import create_app
from agent_adapter.server._health import Readiness
from agent_adapter.server.auth import bearer_auth
from tests.server.helpers import TOKENS, make_registry

BASE = "http://127.0.0.1:8000"


def app_for(**options: Any) -> Any:
    options.setdefault("authenticate", bearer_auth(TOKENS))
    options.setdefault("mcp", False)
    return create_app(make_registry(), **options)


def http_for(app: Any) -> Any:
    return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE)


async def test_ready_without_auth_then_503_once_the_lifespan_drains():
    app = app_for()
    async with http_for(app) as http:
        async with app.router.lifespan_context(app):
            ready = await http.get("/readyz")
            assert ready.status_code == 200 and ready.json()["status"] == "ready"
            assert "version" in ready.json()
        drained = await http.get("/readyz")
        assert drained.status_code == 503
        assert drained.headers["content-type"] == "application/problem+json"
        assert drained.json()["detail"] == "the server is draining"
        health = await http.get("/healthz")
        assert health.status_code == 200  # liveness stays green while draining


async def test_drain_by_hand_keeps_other_routes_serving():
    app = app_for()
    async with app.router.lifespan_context(app), http_for(app) as http:
        app.state.readiness.drain()
        app.state.readiness.drain()  # idempotent
        assert (await http.get("/readyz")).status_code == 503
        headers = {"Authorization": "Bearer t-alice"}
        ok = await http.post(
            "/tools/add/invoke", json={"arguments": {"a": 1, "b": 2}}, headers=headers
        )
        assert ok.status_code == 200


async def test_a_restarted_lifespan_is_ready_again():
    app = app_for()
    async with http_for(app) as http:
        async with app.router.lifespan_context(app):
            pass
        async with app.router.lifespan_context(app):
            assert (await http.get("/readyz")).status_code == 200


async def test_without_mcp_a_mounted_app_is_ready_without_its_lifespan():
    async with http_for(app_for()) as http:
        assert (await http.get("/readyz")).status_code == 200


async def test_with_mcp_it_is_starting_until_the_lifespan_runs():
    app = app_for(mcp=True)
    async with http_for(app) as http:
        starting = await http.get("/readyz")
        assert starting.status_code == 503
        assert starting.json()["detail"] == "the server is starting"
        async with app.router.lifespan_context(app):
            assert (await http.get("/readyz")).status_code == 200


async def test_prefix_method_and_host_rules_apply():
    app = app_for(prefix="/agent")
    async with http_for(app) as http:
        assert (await http.get("/agent/readyz")).status_code == 200
        assert (await http.post("/agent/readyz")).status_code == 405
        wrong = await http.get("/agent/readyz", headers={"Host": "evil.example"})
        assert wrong.status_code == 421


def test_readiness_states():
    readiness = Readiness(needs_lifespan=True)
    assert not readiness.ready and not readiness.draining
    readiness.start()
    assert readiness.ready
    readiness.drain()
    assert readiness.draining and not readiness.ready
    readiness.start()
    assert readiness.ready
