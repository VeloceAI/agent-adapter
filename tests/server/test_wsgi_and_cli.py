"""The WSGI app through a2wsgi, and the `python -m agent_adapter.server` command line."""

from __future__ import annotations

import io
import json
import sys
from typing import Any
from wsgiref.util import setup_testing_defaults

import pytest

pytest.importorskip("starlette")

from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.core.registry import Registry
from agent_adapter.server import __main__ as cli
from agent_adapter.server.auth import bearer_auth
from tests.server.helpers import TOKENS, make_registry

registry = make_registry()  # `--registry tests.server.test_wsgi_and_cli:registry`
not_a_registry = 42


def wsgi_call(
    app: Any, method: str, path: str, body: bytes = b""
) -> tuple[str, dict[str, str], bytes]:
    environ: dict[str, Any] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "SERVER_NAME": "127.0.0.1",
        "SERVER_PORT": "8000",
        "HTTP_HOST": "127.0.0.1:8000",
        "HTTP_AUTHORIZATION": "Bearer t-alice",
        "CONTENT_TYPE": "application/json",
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    }
    setup_testing_defaults(environ)
    seen: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]], exc_info: Any = None) -> Any:
        seen["status"], seen["headers"] = status, dict(headers)
        return lambda data: None

    content = b"".join(app(environ, start_response))
    return seen["status"], seen["headers"], content


def test_a_wsgi_request_through_a2wsgi():
    pytest.importorskip("a2wsgi")
    from agent_adapter.server import create_wsgi_app

    with create_wsgi_app(make_registry(), authenticate=bearer_auth(TOKENS)) as app:
        status, _, content = wsgi_call(app, "GET", "/healthz")
        assert status.startswith("200") and json.loads(content)["status"] == "ok"
        body = json.dumps({"arguments": {"a": 4, "b": 5}}).encode()
        status, headers, content = wsgi_call(app, "POST", "/tools/add/invoke", body)
    assert status.startswith("200") and json.loads(content)["result"] == 9
    assert headers["content-type"] == "application/json"


def test_wsgi_startup_failures_surface(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("a2wsgi")
    from agent_adapter.server import wsgi

    class Broken:
        class router:
            @staticmethod
            def lifespan_context(app: Any) -> Any:
                raise RuntimeError("no lifespan")

    monkeypatch.setattr(wsgi, "create_app", lambda registry, **options: Broken())
    with pytest.raises(RuntimeError, match="no lifespan"):
        wsgi.create_wsgi_app(Registry())


def test_wsgi_without_a2wsgi(monkeypatch: pytest.MonkeyPatch):
    from agent_adapter.server import wsgi

    monkeypatch.setitem(sys.modules, "a2wsgi", None)
    with pytest.raises(MissingDependency, match="server-wsgi"):
        wsgi.create_wsgi_app(Registry(), allow_anonymous=True)


def test_cli_builds_and_serves(monkeypatch: pytest.MonkeyPatch):
    served: dict[str, Any] = {}
    monkeypatch.setattr(cli, "serve", lambda app, host, port: served.update(app=app, port=port))
    tokens = json.dumps({"tok": {"id": "alice", "scopes": ["approvals:decide"]}})
    code = cli.main(
        ["--registry", "tests.server.test_wsgi_and_cli:registry", "--port", "9001", "--no-mcp"],
        environ={cli.TOKENS_ENV: tokens},
    )
    assert code == 0 and served["port"] == 9001
    assert served["app"].state.service.allow_anonymous is False


def test_cli_mounts_mcp_by_default(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("mcp")
    served: dict[str, Any] = {}
    monkeypatch.setattr(cli, "serve", lambda app, host, port: served.update(app=app))
    code = cli.main(
        ["--registry", "tests.server.test_wsgi_and_cli:registry", "--allow-anonymous"],
        environ={},
    )
    assert code == 0
    assert "/mcp" in [route.path for route in served["app"].routes]


@pytest.mark.parametrize(
    ("argv", "env", "message"),
    [
        (["--registry", "nope"], {}, "package.module:attribute"),
        (["--registry", "tests.no_such_module:x"], {}, "cannot load"),
        (["--registry", "tests.server.test_wsgi_and_cli:missing"], {}, "cannot load"),
        (["--registry", "tests.server.test_wsgi_and_cli:not_a_registry"], {}, "not a Registry"),
        (["--registry", "tests.server.test_wsgi_and_cli:registry", "--no-mcp"], {}, "authenticate"),
        (
            ["--registry", "tests.server.test_wsgi_and_cli:registry", "--host", "0.0.0.0"],
            {cli.TOKENS_ENV: '{"t": {"id": "a"}}'},
            "transport_security",
        ),
    ],
)
def test_cli_errors_exit_2(
    argv: list[str], env: dict[str, str], message: str, capsys: pytest.CaptureFixture[str]
):
    if "--host" in argv:
        pytest.importorskip("mcp")
    assert cli.main(argv, environ=env) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "[1]",
        '{"tok": 1}',
        '{"tok": {"id": ""}}',
        '{"tok": {"id": "a", "scopes": [1]}}',
        '{"": {"id": "a"}}',
    ],
)
def test_bad_token_files_are_refused_without_echoing_tokens(raw: str):
    with pytest.raises(ConfigError) as caught:
        cli.tokens_from(raw)
    assert "tok" not in str(caught.value).replace("token", "")


def test_tokens_parse():
    assert cli.tokens_from("  ") == {}
    parsed = cli.tokens_from('{"tok": {"id": "alice"}}')
    assert parsed["tok"].id == "alice" and parsed["tok"].scopes == frozenset()


def test_cli_refuses_public_anonymous_without_the_explicit_flag(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    served: dict[str, Any] = {}
    monkeypatch.setattr(cli, "serve", lambda app, host, port: served.update(app=app))
    base = ["--registry", "tests.server.test_wsgi_and_cli:registry", "--no-mcp"]
    public = [*base, "--host", "0.0.0.0", "--allow-anonymous"]
    assert cli.main(public, environ={}) == 2
    assert "--allow-anonymous-public" in capsys.readouterr().err
    assert not served
    flag = [*base, "--host", "0.0.0.0", "--allow-anonymous-public", "--allowed-host", "t.example"]
    assert cli.main(flag, environ={}) == 0
    assert served["app"].state.service.allow_anonymous is True


async def test_cli_allowed_hosts_reach_the_app(monkeypatch: pytest.MonkeyPatch):
    import httpx2

    served: dict[str, Any] = {}
    monkeypatch.setattr(cli, "serve", lambda app, host, port: served.update(app=app))
    argv = [
        "--registry",
        "tests.server.test_wsgi_and_cli:registry",
        "--no-mcp",
        "--host",
        "0.0.0.0",
        "--allowed-host",
        "tools.example",
        "--allowed-host",
        "api.example",
    ]
    assert cli.main(argv, environ={cli.TOKENS_ENV: '{"t": {"id": "a"}}'}) == 0
    app = served["app"]
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://tools.example") as http:
        assert (await http.get("/healthz")).status_code == 200
        assert (await http.get("/healthz", headers={"Host": "evil.example"})).status_code == 421


async def test_cli_public_bind_with_tokens_accepts_any_host(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    import httpx2

    served: dict[str, Any] = {}
    monkeypatch.setattr(cli, "serve", lambda app, host, port: served.update(app=app))
    argv = ["--registry", "tests.server.test_wsgi_and_cli:registry", "--no-mcp"]
    assert (
        cli.main([*argv, "--host", "0.0.0.0"], environ={cli.TOKENS_ENV: '{"t": {"id": "a"}}'}) == 0
    )
    assert "allowed_hosts" in caplog.text
    transport = httpx2.ASGITransport(app=served["app"])
    async with httpx2.AsyncClient(transport=transport, base_url="http://any.example") as http:
        assert (await http.get("/healthz")).status_code == 200
