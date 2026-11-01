"""A WSGI response closed unread leaves no request task pending when the app closes."""

from __future__ import annotations

import asyncio
import gc
import io
import json
import logging
import threading
from typing import Any
from wsgiref.util import setup_testing_defaults

import pytest

pytest.importorskip("a2wsgi")
pytest.importorskip("starlette")

from agent_adapter.server import wsgi
from tests.server.helpers import make_registry


def _invoke(app: wsgi.WsgiApp) -> Any:
    body = json.dumps({"arguments": {"a": 1, "b": 2}}).encode()
    environ: dict[str, Any] = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/tools/add/invoke",
        "SERVER_NAME": "127.0.0.1",
        "SERVER_PORT": "8000",
        "HTTP_HOST": "127.0.0.1:8000",
        "REMOTE_ADDR": "127.0.0.1",
        "CONTENT_TYPE": "application/json",
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    }
    setup_testing_defaults(environ)
    return app(environ, lambda status, headers, exc_info=None: lambda data: None)


def _pending(app: wsgi.WsgiApp) -> set[asyncio.Task[Any]]:
    lifespan = app._lifespan
    assert lifespan is not None
    return asyncio.all_tasks(lifespan.loop)


def test_a_response_closed_unread_leaves_no_task_pending(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="asyncio")
    app = wsgi.create_wsgi_app(make_registry(), allow_anonymous=True, mcp=False)
    response = _invoke(app)
    next(iter(response))  # what a WSGI server does to get the status line
    response.close()  # a client that went away: the rest is never read
    app.close()
    assert _pending(app) == set()
    gc.collect()  # a destroyed pending task would log "Task was destroyed but it is pending!"
    assert "Task was destroyed" not in caplog.text


def test_a_response_read_in_full_still_closes_cleanly() -> None:
    with wsgi.create_wsgi_app(make_registry(), allow_anonymous=True, mcp=False) as app:
        content = b"".join(_invoke(app))
        assert json.loads(content)["result"] == 3
    assert _pending(app) == set()


def _lifespan(app: wsgi.WsgiApp) -> Any:
    lifespan = app._lifespan
    assert lifespan is not None
    return lifespan


def test_a_task_failing_on_cancel_is_logged_by_type(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=wsgi.__name__)
    app = wsgi.create_wsgi_app(make_registry(), allow_anonymous=True, mcp=False)
    started = threading.Event()

    async def stubborn() -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise RuntimeError("secret detail") from None

    running = asyncio.run_coroutine_threadsafe(stubborn(), _lifespan(app).loop)
    assert started.wait(5)
    app.close()
    assert isinstance(running.exception(timeout=5), RuntimeError)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "WSGI request task failed while closing (RuntimeError)" in warnings
    assert all("secret detail" not in message for message in warnings)
    assert "cancelled 1 unfinished request task(s)" in caplog.text


def test_tasks_that_will_not_stop_do_not_block_close(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    app = wsgi.create_wsgi_app(make_registry(), allow_anonymous=True, mcp=False)

    async def slow(keep: object = None) -> int:
        await asyncio.sleep(60)
        return 0

    monkeypatch.setattr(wsgi, "_cancel_other_tasks", slow)
    monkeypatch.setattr(wsgi, "CLOSE_TIMEOUT_S", 0.05)
    app.close()
    assert "request tasks did not stop" in caplog.text
    assert not _lifespan(app).thread.is_alive()


def test_cancelling_after_the_loop_stopped_does_nothing() -> None:
    app = wsgi.create_wsgi_app(make_registry(), allow_anonymous=True, mcp=False)
    app.close()
    _lifespan(app)._cancel_requests()  # the loop thread is gone: returns at once
    assert _pending(app) == set()
