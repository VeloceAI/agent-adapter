"""m8: `create_wsgi_app(...).close()` ends the lifespan and joins the event loop thread."""

from __future__ import annotations

import gc
import threading
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.registry import Registry

pytest.importorskip("a2wsgi")
pytest.importorskip("starlette")

from agent_adapter.server import wsgi
from tests.server.helpers import make_registry


class FakeApp:
    """An ASGI app stand-in that records its lifespan phases."""

    def __init__(self, fail_on_exit: bool = False) -> None:
        self.phases: list[str] = []
        self.fail_on_exit = fail_on_exit
        app = self

        class router:
            @staticmethod
            @asynccontextmanager
            async def lifespan_context(_: Any) -> AsyncIterator[None]:
                app.phases.append("startup")
                yield
                app.phases.append("shutdown")
                if app.fail_on_exit:
                    raise RuntimeError("shutdown broke")

        self.router = router

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        raise AssertionError("not called in these tests")


def _fake(monkeypatch: pytest.MonkeyPatch, app: FakeApp) -> None:
    monkeypatch.setattr(wsgi, "create_app", lambda registry, **options: app)


def _loop_thread(app: wsgi.WsgiApp) -> threading.Thread:
    lifespan = app._lifespan
    assert lifespan is not None
    return lifespan.thread


def _loop_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "agent-adapter-wsgi-loop")


def test_close_ends_the_lifespan_and_joins_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeApp()
    _fake(monkeypatch, fake)
    app = wsgi.create_wsgi_app(Registry())
    thread = _loop_thread(app)
    assert fake.phases == ["startup"] and thread.is_alive()
    app.close()
    assert fake.phases == ["startup", "shutdown"] and not thread.is_alive()
    assert app._lifespan is not None and app._lifespan.loop.is_closed()
    assert app._lifespan not in wsgi._LIVE
    app.close()  # a second close does nothing
    assert fake.phases == ["startup", "shutdown"]


def test_the_app_is_a_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeApp()
    _fake(monkeypatch, fake)
    with wsgi.create_wsgi_app(Registry()) as app:
        thread = _loop_thread(app)
        assert thread.is_alive()
    assert fake.phases == ["startup", "shutdown"] and not thread.is_alive()


def test_a_real_loopback_app_closes() -> None:
    with wsgi.create_wsgi_app(make_registry(), allow_anonymous=True, mcp=False) as app:
        assert isinstance(app, wsgi.LoopbackOnlyWsgi)
        thread = _loop_thread(app)
    assert not thread.is_alive()


def test_a_failed_shutdown_raises_after_the_loop_stopped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake(monkeypatch, FakeApp(fail_on_exit=True))
    app = wsgi.create_wsgi_app(Registry())
    thread = _loop_thread(app)
    with pytest.raises(RuntimeError, match="shutdown broke"):
        app.close()
    assert not thread.is_alive()
    assert "lifespan shutdown failed (RuntimeError)" in caplog.text
    assert "shutdown broke" not in [r.getMessage() for r in caplog.records if r.levelno >= 30]


def test_a_failed_startup_stops_the_loop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _loop_threads()

    class Broken:
        class router:
            @staticmethod
            def lifespan_context(app: Any) -> Any:
                raise RuntimeError("no lifespan")

    monkeypatch.setattr(wsgi, "create_app", lambda registry, **options: Broken())
    live = len(wsgi._LIVE)
    with pytest.raises(RuntimeError, match="no lifespan"):
        wsgi.create_wsgi_app(Registry())
    assert len(wsgi._LIVE) == live
    assert _loop_threads() == before  # the failed app's loop thread was joined


def test_close_from_the_loop_thread_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, FakeApp())
    with wsgi.create_wsgi_app(Registry()) as app:
        lifespan = app._lifespan
        assert lifespan is not None
        refused: list[BaseException] = []

        def inside() -> None:
            try:
                app.close()
            except ToolUsageError as exc:
                refused.append(exc)

        lifespan.loop.call_soon_threadsafe(inside)
        done = threading.Event()
        lifespan.loop.call_soon_threadsafe(done.set)
        assert done.wait(5)
        assert len(refused) == 1


def test_an_app_dropped_without_close_keeps_its_lifespan_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeApp()
    _fake(monkeypatch, fake)
    app = wsgi.create_wsgi_app(Registry())
    ref = weakref.ref(app._lifespan) if app._lifespan is not None else None
    del app
    gc.collect()  # a collected pending lifespan task was the "Task was destroyed" warning
    lifespan = ref() if ref is not None else None
    assert lifespan is not None and not lifespan._held.done()
    lifespan.close()
    assert fake.phases == ["startup", "shutdown"]
