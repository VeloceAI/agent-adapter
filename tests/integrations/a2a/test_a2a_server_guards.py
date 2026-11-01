"""`a2a_app` guards: auth, Host allow-list, body limit, loopback, config errors, bad calls."""

from __future__ import annotations

import logging
from typing import Any

import pytest

pytest.importorskip("a2a")
pytest.importorskip("starlette")

import httpx

from agent_adapter.core.errors import ConfigError, Forbidden, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import build_spec
from agent_adapter.exec import Executor
from agent_adapter.integrations.a2a import CARD_PATH, Caller, a2a_app, static_bearer
from tests.integrations.a2a.support import RawClient, data, serving, state, status_data, text

URL = "http://agent.test/"
RPC = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {}}


def _registry() -> Registry:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def hidden(a: int) -> int:
        """Not exported to A2A."""
        return a

    registry = Registry()
    registry.add(build_spec(add))
    registry.add(build_spec(hidden, targets=["mcp"]))
    return registry


def _app(**options: Any) -> Any:
    options.setdefault("allowed_hosts", ["agent.test"])
    return a2a_app(_registry(), url=URL, executor=Executor(), **options)


async def _http(
    app: Any, method: str, path: str, *, client: tuple[str, int] = ("127.0.0.1", 1), **kwargs: Any
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, client=client)
    async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as http:
        return await http.request(method, path, **kwargs)


async def test_bearer_auth_refuses_missing_and_wrong_tokens():
    app = _app(authenticate=static_bearer({"tok": "alice"}))
    async with serving(app):
        assert (await _http(app, "POST", "/", json=RPC)).status_code == 401
        wrong = {"Authorization": "Bearer nope"}
        refused = await _http(app, "POST", "/", json=RPC, headers=wrong)
        assert refused.status_code == 401 and refused.headers["www-authenticate"] == "Bearer"
        assert (await _http(app, "GET", CARD_PATH)).status_code == 401
        ok = await _http(app, "GET", CARD_PATH, headers={"Authorization": "Bearer tok"})
        assert ok.status_code == 200 and ok.json()["skills"][0]["id"] == "add"


async def test_public_card_needs_no_credentials():
    app = _app(authenticate=static_bearer({"tok": "alice"}), public_card=True)
    async with serving(app):
        assert (await _http(app, "GET", CARD_PATH)).status_code == 200
        assert (await _http(app, "POST", "/", json=RPC)).status_code == 401


async def test_anonymous_fallback_with_an_authenticator():
    app = _app(authenticate=static_bearer({"tok": "alice"}), allow_anonymous=True)
    async with serving(app), RawClient(app) as anonymous:
        task = await anonymous.call("add", {"a": 1, "b": 2})
    assert state(task) == "completed"


@pytest.mark.parametrize(
    ("raised", "status"), [(Forbidden("no"), 403), (RuntimeError("verifier down"), 401)]
)
async def test_raising_authenticator_refuses(raised: Exception, status: int):
    async def authenticate(request: Any) -> Caller:
        raise raised

    app = _app(authenticate=authenticate)
    async with serving(app):
        assert (await _http(app, "POST", "/", json=RPC)).status_code == status


async def test_authenticator_returning_a_bad_principal_is_a_server_error(caplog):
    async def authenticate(request: Any) -> Any:
        return "alice"  # no `.id`

    app = _app(authenticate=authenticate)
    with caplog.at_level(logging.ERROR):
        async with serving(app):
            assert (await _http(app, "POST", "/", json=RPC)).status_code == 500


async def test_host_header_must_be_allowed():
    app = _app()
    async with serving(app):
        refused = await _http(app, "GET", CARD_PATH, headers={"Host": "evil.test"})
        assert refused.status_code == 421
        assert (
            await _http(app, "GET", CARD_PATH, headers={"Host": "agent.test:8080"})
        ).status_code == 200


async def test_local_default_hosts_refuse_a_rebound_name():
    app = a2a_app(_registry(), url="http://127.0.0.1:8000/", executor=Executor())
    async with serving(app):
        assert (
            await _http(app, "GET", CARD_PATH, headers={"Host": "attacker.test"})
        ).status_code == 421
        local = await _http(app, "GET", CARD_PATH, headers={"Host": "[::1]:8000"})
        assert local.status_code == 200


async def test_body_over_the_limit_is_refused():
    app = _app(max_body_bytes=64)
    async with serving(app):
        declared = await _http(app, "POST", "/", content=b"x" * 65)
        assert declared.status_code == 413

        async def chunks():
            for _ in range(5):
                yield b"x" * 20

        streamed = await _http(app, "POST", "/", content=chunks())
        assert streamed.status_code == 413


async def test_anonymous_local_app_refuses_remote_peers():
    app = _app()
    async with serving(app):
        refused = await _http(app, "GET", CARD_PATH, client=("10.0.0.5", 1234))
        assert refused.status_code == 403 and refused.json()["error"] == "non-local-client"


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"host": "0.0.0.0", "allowed_hosts": None}, "allowed_hosts"),
        ({"host": "0.0.0.0"}, "no auth"),
        ({"max_body_bytes": 0}, "max_body_bytes"),
        ({"allowed_hosts": []}, "at least one"),
    ],
)
def test_unsafe_configurations_are_refused(options: dict[str, Any], error: str):
    with pytest.raises(ConfigError, match=error):
        _app(**options)


def test_public_anonymous_app_needs_explicit_consent():
    app = _app(host="0.0.0.0", allow_anonymous=True)
    assert app.state.a2a_card.skills[0].id == "add"


def test_bad_arguments_are_usage_errors():
    with pytest.raises(ToolUsageError):
        a2a_app(object(), url=URL)  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError):
        a2a_app(_registry(), url=URL, authenticate="tok")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="http"):
        a2a_app(_registry(), url="ftp://agent.test/")
    with pytest.raises(ToolUsageError):
        static_bearer({"": "alice"})
    with pytest.raises(ToolUsageError):
        static_bearer({"tok": ""})


@pytest.mark.parametrize(
    ("parts", "kind"),
    [
        ([text("add 1 and 2")], "bad-request"),
        ([data({"tool": "nope", "arguments": {}})], "tool-not-found"),
        ([data({"tool": "hidden", "arguments": {"a": 1}})], "tool-not-found"),
        ([data({"tool": "add", "arguments": [1, 2]})], "bad-request"),
        ([data({"tool": "add", "arguments": {"a": "x" * 200, "b": 1}})], "bad-request"),
    ],
)
async def test_bad_calls_are_rejected(parts: list[Any], kind: str):
    app = _app(max_data_bytes=128)
    async with serving(app), RawClient(app) as client:
        task = await client.send(parts)
    assert state(task) == "rejected"
    assert status_data(task)["error"]["kind"] == kind


@pytest.mark.parametrize("verdict", [False, "yes", RuntimeError("hook down")])
async def test_authorize_must_say_true(verdict: Any):
    async def authorize(principal: Any, spec: Any, arguments: Any) -> Any:
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    app = _app(authorize=authorize)
    async with serving(app), RawClient(app) as client:
        task = await client.call("add", {"a": 1, "b": 2})
    assert state(task) == "rejected"
    assert status_data(task)["error"]["kind"] == "forbidden"


async def test_results_keep_their_json_types():
    app = _app()
    async with serving(app), RawClient(app) as client:
        task = await client.call("add", {"a": 2, "b": 40})
    assert state(task) == "completed"
    assert task.artifacts[0].parts[0].text == "42"
