"""`WsgiApp.close()` with requests in flight, and requests that arrive after it."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any
from wsgiref.util import setup_testing_defaults

import pytest

pytest.importorskip("a2wsgi")
pytest.importorskip("starlette")

from agent_adapter.core.registry import Registry
from agent_adapter.server import wsgi
from tests.server.helpers import make_registry

WAIT_S = 10.0


class SlowApp:
    """An ASGI app whose request never ends; it records lifespan and cancellation order."""

    def __init__(self, *, chunk_first: bool = False) -> None:
        self.phases: list[str] = []
        self.chunk_first = chunk_first
        self.running = threading.Event()
        app = self

        class router:
            @staticmethod
            @asynccontextmanager
            async def lifespan_context(_: Any) -> AsyncIterator[None]:
                app.phases.append("startup")
                yield
                app.phases.append("shutdown")

        self.router = router

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if self.chunk_first:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"first", "more_body": True})
        self.running.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.phases.append("request cancelled")
            raise


class Client:
    """One WSGI request on a worker thread, as a WSGI server would run it."""

    def __init__(self, app: wsgi.WsgiApp, *, read_all: bool = True) -> None:
        self.app = app
        self.read_all = read_all
        self.status = ""
        self.headers: dict[str, str] = {}
        self.chunks: list[bytes] = []
        self.error: BaseException | None = None
        self.body: Iterable[bytes] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> Client:
        self.thread.start()
        return self

    def _start_response(self, status: str, headers: list[tuple[str, str]], exc: Any = None) -> Any:
        self.status, self.headers = status, dict(headers)
        return lambda data: None

    def _run(self) -> None:
        environ: dict[str, Any] = {
            "REQUEST_METHOD": "GET",
            "PATH_INFO": "/healthz",
            "SERVER_NAME": "127.0.0.1",
            "SERVER_PORT": "8000",
            "REMOTE_ADDR": "127.0.0.1",
            "wsgi.input": io.BytesIO(b""),
        }
        setup_testing_defaults(environ)
        try:
            self.body = self.app(environ, self._start_response)
            for chunk in self.body:
                self.chunks.append(chunk)
                if not self.read_all:
                    return  # holds the response open: never read further, never closed
        except BaseException as exc:
            self.error = exc

    def finished(self) -> bool:
        self.thread.join(WAIT_S)
        return not self.thread.is_alive()


def _slow(monkeypatch: pytest.MonkeyPatch, app: SlowApp) -> wsgi.WsgiApp:
    monkeypatch.setattr(wsgi, "create_app", lambda registry, **options: app)
    return wsgi.create_wsgi_app(Registry())


def _problem(client: Client) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(b"".join(client.chunks))
    return body


def test_a_request_after_close_gets_a_503_problem(caplog: pytest.LogCaptureFixture) -> None:
    app = wsgi.create_wsgi_app(make_registry(), allow_anonymous=True, mcp=False)
    app.close()
    client = Client(app).start()
    assert client.finished() and client.error is None
    assert client.status == "503 Service Unavailable"
    assert client.headers["Content-Type"] == "application/problem+json"
    assert client.headers["Retry-After"] == "5"
    assert _problem(client)["type"].endswith(":service-unavailable")
    assert "WSGI app refused a request: it is closed" in caplog.text


def test_close_cancels_requests_before_the_lifespan_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    slow = SlowApp()
    app = _slow(monkeypatch, slow)
    client = Client(app).start()
    assert slow.running.wait(WAIT_S)
    app.close()
    assert slow.phases == ["startup", "request cancelled", "shutdown"]
    assert client.finished() and client.error is None
    assert client.status == "503 Service Unavailable"
    assert _problem(client)["detail"] == "the server is shutting down"


def test_a_worker_mid_response_is_released_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    slow = SlowApp(chunk_first=True)
    app = _slow(monkeypatch, slow)
    client = Client(app).start()
    assert slow.running.wait(WAIT_S)
    app.close()
    assert client.finished()
    assert client.status == "200 OK" and client.chunks == [b"first"]
    assert isinstance(client.error, ConnectionAbortedError)


def test_close_waits_a_bounded_time_for_a_response_left_open(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    slow = SlowApp(chunk_first=True)
    app = _slow(monkeypatch, slow)
    client = Client(app, read_all=False).start()
    assert client.finished() and client.chunks == [b"first"]  # the server stops reading
    monkeypatch.setattr(wsgi, "CLOSE_TIMEOUT_S", 0.2)
    app.close()
    assert "1 WSGI response(s) still open after 0.2 s" in caplog.text
    assert slow.phases == ["startup", "shutdown"]
    assert client.body is not None
    getattr(client.body, "close", lambda: None)()  # the server closes it late: no error


def test_a_cancel_outside_close_still_propagates() -> None:
    lifespan = type("Open", (), {"closing": False})()

    async def app(scope: Any, receive: Any, send: Any) -> None:
        raise asyncio.CancelledError

    guarded = wsgi._CloseAware(app, lifespan)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(guarded({"type": "http"}, None, None))


def test_a_closed_app_logs_and_refuses_each_request(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=wsgi.__name__)
    app = wsgi.create_wsgi_app(make_registry(), allow_anonymous=True, mcp=False)
    app.close()
    for _ in range(2):
        assert Client(app).start().finished()
    refused = [r for r in caplog.records if "refused a request" in r.getMessage()]
    assert len(refused) == 2
