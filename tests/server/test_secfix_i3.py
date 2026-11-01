"""I3: an anonymous app is refused on a non-local bind unless `allow_anonymous_public=True`."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.server.auth import bearer_auth
from tests.server.helpers import TOKENS, make_registry

pytest.importorskip("starlette")
pytest.importorskip("httpx2")

from agent_adapter.server import create_app, serve
from tests.server.test_asgi import client


def anonymous_app(**options: Any) -> Any:
    options.setdefault("mcp", False)
    return create_app(make_registry(), allow_anonymous=True, **options)


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    uvicorn = pytest.importorskip("uvicorn")
    started: list[str] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: started.append(host))
    return started


@pytest.mark.parametrize(
    "options", [{"host": "0.0.0.0"}, {"host": "10.0.0.5"}, {"mcp_options": {"host": "::"}}]
)
def test_a_public_anonymous_app_is_refused(options: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="allow_anonymous_public=True"):
        anonymous_app(**options)


def test_serving_an_anonymous_app_publicly_is_refused(ran: list[str]) -> None:
    app = anonymous_app()
    with pytest.raises(ConfigError, match="allow_anonymous_public=True"):
        serve(app, host="0.0.0.0")
    assert ran == []


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_local_anonymous_apps_are_fine(host: str, ran: list[str]) -> None:
    serve(anonymous_app(host=host), host=host)
    assert ran == [host]


def test_authenticated_public_apps_need_no_flag(ran: list[str]) -> None:
    app = create_app(make_registry(), authenticate=bearer_auth(TOKENS), host="0.0.0.0", mcp=False)
    serve(app, host="0.0.0.0")
    assert ran == ["0.0.0.0"]


async def test_the_explicit_flag_allows_a_public_bind_with_local_names_by_default(
    ran: list[str],
) -> None:
    app = anonymous_app(host="0.0.0.0", allow_anonymous_public=True)
    serve(app, host="0.0.0.0")
    assert ran == ["0.0.0.0"]
    async with client(app, token=None) as http:
        assert (await http.get("/tools", headers={"Host": "tools.example"})).status_code == 421
        assert (await http.get("/tools", headers={"Host": "localhost"})).status_code == 200
    named = anonymous_app(host="0.0.0.0", allow_anonymous_public=True, allowed_hosts=["t.ex"])
    async with client(named, token=None) as http:
        assert (await http.get("/tools", headers={"Host": "t.ex"})).status_code == 200


def test_the_flag_is_not_passed_to_the_service() -> None:
    app = anonymous_app(allow_anonymous_public=True)
    assert app.state.service.allow_anonymous is True


def test_the_wsgi_app_is_refused_too() -> None:
    pytest.importorskip("a2wsgi")
    from agent_adapter.server.wsgi import create_wsgi_app

    with pytest.raises(ConfigError, match="allow_anonymous_public=True"):
        create_wsgi_app(make_registry(), allow_anonymous=True, host="0.0.0.0", mcp=False)
