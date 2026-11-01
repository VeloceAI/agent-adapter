"""`agui_endpoint`: a Starlette (and FastAPI) `POST` handler that streams one AG-UI run."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.ag_ui._input import MAX_BODY_BYTES, AGUIInputInvalid, parse_input
from agent_adapter.integrations.ag_ui.bridge import Agent, AGUIBridge

logger = logging.getLogger(__name__)

# Returns the caller's principal id ("" for an anonymous caller), or None to refuse (401).
Authenticate = Callable[[Any], Awaitable[str | None]]
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})


async def anonymous(request: Any) -> str:
    """An `Authenticate` that lets everyone in as one anonymous principal (local use only)."""
    return ""


def agui_endpoint(
    bridge: AGUIBridge,
    agent: Agent,
    *,
    authenticate: Authenticate,
    max_body_bytes: int = MAX_BODY_BYTES,
    allowed_hosts: Iterable[str] | None = None,
) -> Callable[[Any], Awaitable[Any]]:
    """A Starlette endpoint: authenticate, read a bounded JSON body, stream the run as SSE.

    Mount it with `Route("/agui", endpoint, methods=["POST"])`, or on FastAPI with
    `app.add_route("/agui", endpoint, methods=["POST"])`. `authenticate` is required:
    pass `anonymous` only for a local, single-user app.

    `allowed_hosts` lists the `Host` names served (`"*"`: any). By default an `anonymous`
    endpoint serves local names only (DNS rebinding); any other endpoint serves every host.

    Responses: 421 (Host not served), 401 (refused), 413 (body too large), 415 (not JSON),
    422 (invalid input), else 200 `text/event-stream`.
    """

    hosts = _hosts(allowed_hosts, local=authenticate is anonymous)

    async def endpoint(request: Any) -> Any:
        from starlette.responses import JSONResponse, StreamingResponse

        if hosts is not None and _host_name(request.headers.get("host", "")) not in hosts:
            logger.warning("ag_ui: request refused: Host header not allowed")
            return JSONResponse({"error": "this host is not served"}, status_code=421)
        principal = await authenticate(request)
        if principal is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        content_type = request.headers.get("content-type", "")
        if content_type.split(";")[0].strip().lower() != "application/json":
            return JSONResponse({"error": "expected application/json"}, status_code=415)
        body = await _read(request, max_body_bytes)
        if body is None:
            return JSONResponse({"error": "request body too large"}, status_code=413)
        try:
            run_input = parse_input(body, max_bytes=max_body_bytes)
        except AGUIInputInvalid as exc:
            logger.info("ag_ui: invalid run input refused (%s)", exc.kind)
            return JSONResponse({"error": str(exc)}, status_code=422)
        return StreamingResponse(
            bridge.sse(run_input, agent, principal=principal),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    return endpoint


async def _read(request: Any, limit: int) -> bytes | None:
    """The body, read in chunks; None as soon as it passes `limit` bytes."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            logger.info("ag_ui: request body over %d bytes refused", limit)
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _hosts(given: Iterable[str] | None, *, local: bool) -> frozenset[str] | None:
    """The served host names, or None for any host."""
    if given is None:
        return LOCAL_HOSTS if local else None
    if isinstance(given, str):
        raise ConfigError("allowed_hosts must be a list of host names, not a string")
    names = list(given)
    if not names or not all(isinstance(name, str) and name.strip() for name in names):
        raise ConfigError("allowed_hosts must name at least one non-empty host")
    if any(name.strip() == "*" for name in names):
        return None
    return frozenset(_host_name(name) for name in names)


def _host_name(header: str) -> str:
    """`Host` without its port, lower case: `"[::1]:80"` -> `"[::1]"`."""
    value = header.strip().lower()
    if value.startswith("["):
        return value[: value.find("]") + 1]
    return value.rpartition(":")[0] if value.count(":") == 1 else value
