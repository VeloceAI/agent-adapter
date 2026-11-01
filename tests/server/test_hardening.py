"""Request hardening (fix round 1): media types, Host allowlist, duplicate headers, timeouts,
MCP option keys, path-free logs and keyed log references."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.server import auth
from agent_adapter.server.auth import RequestInfo, principal_ref, set_log_key
from tests.server.helpers import call, invoke, make_service

pytest.importorskip("starlette")
pytest.importorskip("httpx2")

import httpx2

from tests.server.test_asgi import app_for, client


@pytest.mark.parametrize("media_type", ["text/plain", "application/x-www-form-urlencoded", None])
async def test_posts_must_be_json(media_type: str | None):
    service = make_service()
    headers = {"Content-Type": media_type} if media_type else {}
    reply = await call(
        service,
        "POST",
        "/tools/add/invoke",
        raw=b'{"arguments": {"a": 1, "b": 2}}',
        headers=headers,
        json_body=False,
    )
    assert reply.status == 415 and reply.body["type"].endswith(":unsupported-media-type")


async def test_json_with_parameters_is_fine():
    reply = await invoke(
        make_service(),
        "add",
        {"a": 1, "b": 2},
        headers={"Content-Type": "Application/JSON; charset=utf-8"},
    )
    assert reply.status == 200


@pytest.mark.parametrize("name", ["Authorization", "Idempotency-Key"])
async def test_case_variant_duplicate_headers_are_refused(name: str):
    service = make_service()
    headers = {name.lower(): "Bearer t-alice" if name == "Authorization" else "k1"}
    reply = await invoke(
        service, "add", {"a": 1, "b": 2}, headers={name: headers[name.lower()], **headers}
    )
    assert reply.status == 400 and name.lower() in reply.body["detail"].lower()


async def test_unexpected_errors_log_the_route_not_the_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO)
    service = make_service()

    async def explode(*args: Any) -> Any:
        raise KeyError("x")

    monkeypatch.setattr(service, "_tool", explode)
    await call(service, "GET", "/tools/sekrit-name")
    await call(service, "GET", "/approvals/apr_sekrit")
    assert "sekrit" not in caplog.text
    assert "GET tools/{} failed (KeyError)" in caplog.text
    assert "GET approvals/{} -> 404 approval-not-found" in caplog.text


def test_principal_refs_are_keyed_hashes():
    plain = hashlib.sha256(b"alice").hexdigest()[:12]
    assert principal_ref("alice") != plain and principal_ref("alice") == principal_ref("alice")
    previous = auth._log_key
    try:
        set_log_key(b"k" * 32)
        expected = hmac.new(b"k" * 32, b"alice", hashlib.sha256).hexdigest()[:12]
        assert principal_ref("alice") == expected
        with pytest.raises(ConfigError):
            set_log_key(b"short")
    finally:
        auth._log_key = previous


class TestAsgi:
    async def test_cross_site_form_posts_are_refused(self):
        async with client(app_for()) as http:
            reply = await http.post(
                "/tools/add/invoke",
                content=b'{"arguments": {"a": 1, "b": 2}}',
                headers={"Content-Type": "text/plain"},
            )
            assert reply.status_code == 415

    async def test_foreign_hosts_are_misdirected(self):
        async with client(app_for()) as http:
            reply = await http.get("/tools", headers={"Host": "evil.example"})
            assert reply.status_code == 421
            assert reply.json()["type"].endswith(":misdirected-request")
            for host in ("localhost:9000", "127.0.0.1", "[::1]:8000"):
                assert (await http.get("/tools", headers={"Host": host})).status_code == 200

    async def test_allowed_hosts_are_configurable(self):
        async with client(app_for(allowed_hosts=["tools.example.com", "api.example:8443"])) as h:
            assert (
                await h.get("/tools", headers={"Host": "tools.example.com:443"})
            ).status_code == 200
            assert (await h.get("/tools", headers={"Host": "api.example:8443"})).status_code == 200
            assert (await h.get("/tools", headers={"Host": "api.example:9999"})).status_code == 421
            assert (await h.get("/tools", headers={"Host": "127.0.0.1:8000"})).status_code == 421
        async with client(app_for(allowed_hosts=["*"])) as h:
            assert (await h.get("/tools", headers={"Host": "anything"})).status_code == 200

    async def test_public_authenticated_apps_skip_the_check_by_default(self):
        async with client(app_for(mcp_options={"host": "0.0.0.0"})) as h:
            assert (await h.get("/tools", headers={"Host": "tools.example"})).status_code == 200
        with pytest.raises(ConfigError, match="allow_anonymous_public"):
            app_for(mcp_options={"host": "0.0.0.0"}, authenticate=None, allow_anonymous=True)
        anonymous = app_for(
            mcp_options={"host": "0.0.0.0"},
            authenticate=None,
            allow_anonymous=True,
            allow_anonymous_public=True,
        )
        async with client(anonymous) as h:
            assert (await h.get("/tools", headers={"Host": "tools.example"})).status_code == 421

    async def test_duplicate_authorization_headers_are_refused(self):
        async with client(app_for(), token=None) as http:
            headers = [("Authorization", "Bearer t-bob"), ("Authorization", "Bearer t-alice")]
            reply = await http.get("/tools", headers=httpx2.Headers(headers))
            assert reply.status_code == 400

    async def test_duplicate_idempotency_keys_are_refused(self):
        async with client(app_for()) as http:
            headers = httpx2.Headers([("Idempotency-Key", "a"), ("Idempotency-Key", "b")])
            reply = await http.post(
                "/tools/add/invoke", json={"arguments": {"a": 1, "b": 1}}, headers=headers
            )
            assert reply.status_code == 400

    async def test_slow_bodies_time_out_with_408(self):
        app = app_for()
        app.state.service.body_timeout_s = 0.02

        async def slow() -> AsyncIterator[bytes]:
            yield b'{"arguments":'
            await asyncio.sleep(0.05)
            yield b" {}}"

        async with client(app) as http:
            reply = await http.post(
                "/tools/add/invoke", content=slow(), headers={"Content-Type": "application/json"}
            )
            assert reply.status_code == 408 and reply.json()["type"].endswith(":request-timeout")

    def test_mcp_options_accept_only_transport_keys(self):
        pytest.importorskip("mcp")
        for key in ("client_approval", "approval_store", "request_state_keys", "principal"):
            with pytest.raises(ConfigError, match="mcp_options"):
                app_for(mcp=True, mcp_options={key: True})
        app_for(mcp=True, mcp_options={"name": "tools", "instructions": "x", "version": "1"})


def test_request_info_header_lookup_still_works():
    info = RequestInfo("GET", "/", {"X-A": "1"}, {}, None)
    assert info.header("x-a") == "1"


def test_host_rules_parse_ports_and_refuse_bad_entries():
    from agent_adapter.server._hosts import HostRule, default_hosts, split_host

    assert split_host("[::1]") == ("[::1]", None)
    assert split_host("[::1]:80") == ("[::1]", "80")
    assert split_host("a.b") == ("a.b", None)
    rule = HostRule(["a.example:*"])
    assert rule.allows("A.example:1") and not rule.allows(None) and not rule.allows("")
    for bad in ([""], ["  "], [7]):
        with pytest.raises(ConfigError):
            HostRule(bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        default_hosts("a.example", {}, anonymous=False)  # type: ignore[arg-type]

    class Security:
        allowed_hosts = ("tools.example:*",)

    assert default_hosts(None, {"transport_security": Security()}, anonymous=False) == [
        "tools.example:*"
    ]
    assert default_hosts(None, {"host": "0.0.0.0"}, anonymous=False) == ["*"]


class TestBindHost:
    """The Host default follows the real bind host: `serve(host=...)` or `create_app(host=...)`."""

    @pytest.fixture(autouse=True)
    def _no_uvicorn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        uvicorn = pytest.importorskip("uvicorn")
        monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)

    async def test_serving_publicly_with_auth_accepts_any_host_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ):
        from agent_adapter.server import serve

        app = app_for()
        async with client(app) as http:
            assert (await http.get("/tools", headers={"Host": "tools.example"})).status_code == 421
            serve(app, host="0.0.0.0")
            assert "allowed_hosts" in caplog.text and "any Host" in caplog.text
            assert (await http.get("/tools", headers={"Host": "tools.example"})).status_code == 200

    async def test_create_app_takes_the_bind_host(self, caplog: pytest.LogCaptureFixture):
        async with client(app_for(host="0.0.0.0")) as http:
            assert (await http.get("/tools", headers={"Host": "tools.example"})).status_code == 200
        assert "allowed_hosts" in caplog.text

    async def test_anonymous_public_binds_stay_local_only(self):
        from agent_adapter.server import serve

        with pytest.raises(ConfigError, match="allow_anonymous_public"):
            serve(app_for(authenticate=None, allow_anonymous=True), host="0.0.0.0")
        app = app_for(authenticate=None, allow_anonymous=True, allow_anonymous_public=True)
        serve(app, host="0.0.0.0")
        async with client(app, token=None) as http:
            assert (await http.get("/tools", headers={"Host": "tools.example"})).status_code == 421

    async def test_explicit_allowed_hosts_win_over_the_bind(self, caplog: pytest.LogCaptureFixture):
        from agent_adapter.server import serve

        app = app_for(allowed_hosts=["tools.example"])
        serve(app, host="0.0.0.0")
        async with client(app) as http:
            assert (await http.get("/tools", headers={"Host": "evil.example"})).status_code == 421
            assert (await http.get("/tools", headers={"Host": "tools.example"})).status_code == 200
        assert "any Host" not in caplog.text

    def test_serving_locally_keeps_local_names_and_plain_apps_are_fine(self):
        from agent_adapter.server import serve

        app = app_for()
        serve(app, host="localhost")
        assert app.state.hosts.rule.allows("localhost:8000")
        assert not app.state.hosts.rule.allows("tools.example")
        serve(object())  # an app this module did not build: just runs
