"""WSGI hosts (Django, Flask, gunicorn sync workers): the ASGI app through `a2wsgi`."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from http import HTTPStatus
from types import TracebackType
from typing import Any

from agent_adapter.core.errors import MissingDependency, ToolUsageError, log_failure
from agent_adapter.core.registry import Registry
from agent_adapter.integrations.mcp._peer import is_local_peer
from agent_adapter.server._hosts import HostPolicy
from agent_adapter.server._state import ServiceResponse, encode_body, problem_response
from agent_adapter.server.asgi import NON_LOCAL_PROBLEM, create_app
from agent_adapter.server.problems import Problem, problem

logger = logging.getLogger(__name__)

STARTUP_TIMEOUT_S = 30.0
CLOSE_TIMEOUT_S = 30.0
DRAIN_POLL_S = 0.05

CLOSED_PROBLEM = problem("service-unavailable", "the WSGI app is closed")
SHUTTING_DOWN_PROBLEM = problem("service-unavailable", "the server is shutting down")

# Lifespans not closed yet: a strong reference, so a held lifespan task is never collected.
_LIVE: set[_Lifespan] = set()


def create_wsgi_app(registry: Registry, **options: Any) -> WsgiApp:
    """`a2wsgi.ASGIMiddleware(create_app(registry, **options))` with the app's lifespan running.

    The lifespan (which starts the MCP session manager) runs on the middleware's own event
    loop thread until `close()` (or the end of a `with` block); without it, until the process
    ends. SSE responses are buffered by WSGI servers: follow run events over ASGI. An anonymous
    app without `allow_anonymous_public=True` is wrapped so a request whose `REMOTE_ADDR` is
    not loopback gets the 403 `non-local-client` problem.

    Drain the WSGI server first (its worker-exit hook), then call `close()`. A request still
    in flight is cancelled: one whose response has not started gets a 503 problem, one that
    has is aborted. `close()` waits at most `CLOSE_TIMEOUT_S` for open responses, and a
    request after `close()` gets a 503 problem.

    Raises:
        MissingDependency: `a2wsgi` (or `starlette`) is not installed.
    """
    try:
        from a2wsgi import ASGIMiddleware
    except ImportError as exc:
        logger.debug("a2wsgi is not importable: %s", exc)
        raise MissingDependency(
            'the WSGI app needs a2wsgi: pip install "agent-adapter[server-wsgi]"'
        ) from exc
    app = create_app(registry, **options)
    lifespan = _Lifespan(app)
    middleware = ASGIMiddleware(_CloseAware(app, lifespan), loop=lifespan.loop)
    hosts = getattr(getattr(app, "state", None), "hosts", None)
    if isinstance(hosts, HostPolicy) and hosts.local_only:
        return LoopbackOnlyWsgi(middleware, lifespan)
    return WsgiApp(middleware, lifespan)


class WsgiApp:
    """The WSGI callable `create_wsgi_app` returns; `close()` or `with` ends its lifespan."""

    def __init__(self, app: Any, lifespan: _Lifespan | None = None) -> None:
        self.app = app
        self._lifespan = lifespan

    def __call__(
        self, environ: dict[str, Any], start_response: Callable[..., Any]
    ) -> Iterable[bytes]:
        if self._lifespan is None:
            result: Iterable[bytes] = self.app(environ, start_response)
            return result
        return self._serve(self._lifespan, environ, start_response)

    def _serve(
        self, lifespan: _Lifespan, environ: dict[str, Any], start_response: Callable[..., Any]
    ) -> Iterator[bytes]:
        """Serve one request; it counts as open until the server closes its response."""
        if not lifespan.enter():
            logger.warning("WSGI app refused a request: it is closed")
            yield from _refuse(start_response, CLOSED_PROBLEM)
            return
        try:
            yield from self.app(environ, start_response)
        finally:
            lifespan.leave()

    def close(self) -> None:
        """Refuse new requests, cancel open ones, end the lifespan and the loop; safe twice.

        Call it from outside a request, after the WSGI server stopped handing out requests.

        Raises:
            ToolUsageError: called from the app's own event loop thread.
            Exception: the app's lifespan shutdown failed (raised once the loop has stopped).
        """
        if self._lifespan is not None:
            self._lifespan.close()

    def __enter__(self) -> WsgiApp:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class LoopbackOnlyWsgi(WsgiApp):
    """WSGI wrapper: a request whose `REMOTE_ADDR` is not loopback gets a 403 problem."""

    def __call__(
        self, environ: dict[str, Any], start_response: Callable[..., Any]
    ) -> Iterable[bytes]:
        # a2wsgi copies REMOTE_ADDR into the ASGI scope only with REMOTE_PORT, so check it here
        peer = environ.get("REMOTE_ADDR")
        if peer is None or (isinstance(peer, str) and is_local_peer(peer)):
            return super().__call__(environ, start_response)
        logger.warning(
            "HTTP tool server refused a request from non-local client %s: anonymous and local-only",
            peer,
        )
        return _refuse(start_response, NON_LOCAL_PROBLEM)


def _refuse(start_response: Callable[..., Any], found: Problem) -> list[bytes]:
    """Answer a WSGI request with `found` as an RFC 9457 problem."""
    response = problem_response(found)
    body = encode_body(response)
    status = f"{response.status} {HTTPStatus(response.status).phrase}"
    start_response(status, _headers(response, body))
    return [body]


def _headers(response: ServiceResponse, body: bytes) -> list[tuple[str, str]]:
    return [
        ("Content-Type", response.media_type),
        ("Content-Length", str(len(body))),
        *response.headers.items(),
    ]


class _CloseAware:
    """ASGI wrapper: a request cancelled by `close()` answers 503, or aborts once it started."""

    # a2wsgi wakes its worker only when the ASGI task ends with a result or an error; a bare
    # cancellation would leave the worker waiting forever.

    def __init__(self, app: Any, lifespan: Any) -> None:
        self.app = app
        self.lifespan = lifespan

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        started = False

        async def tracked(message: Any) -> None:
            nonlocal started
            started = started or message.get("type") == "http.response.start"
            await send(message)

        try:
            await self.app(scope, receive, tracked)
        except asyncio.CancelledError:
            if not self.lifespan.closing or scope.get("type") != "http":
                raise
            if started:
                raise ConnectionAbortedError("the WSGI app closed mid-response") from None
            await _send_problem(send, SHUTTING_DOWN_PROBLEM)


async def _send_problem(send: Any, found: Problem) -> None:
    response = problem_response(found)
    body = encode_body(response)
    headers = [(k.encode("latin-1"), v.encode("latin-1")) for k, v in _headers(response, body)]
    await send({"type": "http.response.start", "status": response.status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class _Lifespan:
    """An ASGI app's lifespan held open on a dedicated event loop thread."""

    def __init__(self, app: Any) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, name="agent-adapter-wsgi-loop", daemon=True
        )
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._closed = False
        self._open = 0
        self._stop: asyncio.Event | None = None
        self._hold_task: asyncio.Task[None] | None = None
        self.thread.start()
        _LIVE.add(self)
        started: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._held = asyncio.run_coroutine_threadsafe(self._hold(app, started), self.loop)
        try:
            started.result(timeout=STARTUP_TIMEOUT_S)
        except BaseException:
            self._closed = True
            self._finish()  # startup failed: its error is what the caller sees
            raise

    @property
    def closing(self) -> bool:
        """True once `close()` has begun (or startup failed)."""
        return self._closed

    def enter(self) -> bool:
        """Count one open response; False once the app is closing or closed."""
        with self._lock:
            if self._closed:
                return False
            self._open += 1
            return True

    def leave(self) -> None:
        """The server closed one response."""
        with self._idle:
            self._open -= 1
            self._idle.notify_all()

    async def _hold(self, app: Any, started: concurrent.futures.Future[None]) -> None:
        self._stop = asyncio.Event()
        self._hold_task = asyncio.current_task()
        try:
            async with app.router.lifespan_context(app):
                started.set_result(None)
                await self._stop.wait()
        except BaseException as exc:
            if started.done():
                raise
            started.set_exception(exc)

    def close(self) -> None:
        if threading.current_thread() is self.thread:
            raise ToolUsageError("close() the WSGI app from outside its own requests")
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._drain()
        error = self._finish()
        if error is not None:
            log_failure(logger, logging.WARNING, error, "WSGI app lifespan shutdown failed")
            raise error

    def _drain(self) -> None:
        """Cancel request tasks before the lifespan ends; wait a bounded time for responses."""
        deadline = time.monotonic() + CLOSE_TIMEOUT_S
        while True:
            self._cancel_requests()
            with self._idle:
                if self._open == 0:
                    return
                left = deadline - time.monotonic()
                if left <= 0:
                    logger.warning(
                        "%d WSGI response(s) still open after %.1f s; closing anyway",
                        self._open,
                        CLOSE_TIMEOUT_S,
                    )
                    return
                self._idle.wait(min(DRAIN_POLL_S, left))

    def _finish(self) -> BaseException | None:
        """Stop the lifespan, then the loop; the shutdown error, if any."""
        stop = self._stop
        if stop is not None:
            self.loop.call_soon_threadsafe(stop.set)
        done, _ = concurrent.futures.wait([self._held], timeout=CLOSE_TIMEOUT_S)
        if not done:
            logger.warning("WSGI app lifespan did not end in %.0f s; cancelling", CLOSE_TIMEOUT_S)
            self._held.cancel()
            concurrent.futures.wait([self._held], timeout=CLOSE_TIMEOUT_S)
        error = None if self._held.cancelled() or not self._held.done() else self._held.exception()
        self._cancel_requests()  # anything the lifespan shutdown left behind
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=CLOSE_TIMEOUT_S)
        if self.thread.is_alive():
            logger.warning("WSGI app event loop thread did not stop in %.0f s", CLOSE_TIMEOUT_S)
        else:
            self.loop.close()
        _LIVE.discard(self)
        return error

    def _cancel_requests(self) -> None:
        """Cancel and await request tasks still running, so none is destroyed while pending."""
        # a2wsgi leaves a request's task waiting when its response is closed unread
        if not self.thread.is_alive():
            return
        future = asyncio.run_coroutine_threadsafe(_cancel_other_tasks(self._hold_task), self.loop)
        try:
            cancelled = future.result(timeout=CLOSE_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning(
                "WSGI app request tasks did not stop in %.0f s; closing anyway", CLOSE_TIMEOUT_S
            )
            return
        if cancelled:
            logger.info("WSGI app close cancelled %d unfinished request task(s)", cancelled)


async def _cancel_other_tasks(keep: asyncio.Task[None] | None = None) -> int:
    """Cancel the loop's other tasks except `keep`, await them; how many were newly cancelled."""
    current = asyncio.current_task()
    rest = [task for task in asyncio.all_tasks() if task is not current and task is not keep]
    fresh = [task for task in rest if not task.cancelling()]
    for task in fresh:
        task.cancel()
    results = await asyncio.gather(*rest, return_exceptions=True)
    for result in results:
        if isinstance(result, ConnectionAbortedError):
            logger.debug("a WSGI response was aborted by close()")
        elif isinstance(result, Exception):
            log_failure(logger, logging.WARNING, result, "WSGI request task failed while closing")
    return len(fresh)
