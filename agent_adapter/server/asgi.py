"""The Starlette app: `ToolService` routes under `prefix`, MCP at `prefix + mcp_path`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, Forbidden, MissingDependency
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations.mcp._peer import NON_LOCAL_CLIENT, LoopbackOnly
from agent_adapter.integrations.mcp._scope import GatedPrincipal
from agent_adapter.server._health import Readiness
from agent_adapter.server._hosts import HostPolicy
from agent_adapter.server._payloads import Refused
from agent_adapter.server._state import ServiceResponse, encode_body, problem_response
from agent_adapter.server.auth import Principal, RequestInfo
from agent_adapter.server.problems import problem, problem_for
from agent_adapter.server.service import SINGLE_HEADERS, ToolService

if TYPE_CHECKING:
    from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)

PRINCIPAL_STATE = "aa.principal"  # request scope state: the principal the gate authenticated
MCP_OPTIONS = frozenset({"host", "transport_security", "name", "instructions", "version"})
_HINT = 'pip install "agent-adapter[server]"'


def create_app(
    registry: Registry,
    *,
    mcp: bool = True,
    mcp_path: str = "/mcp",
    prefix: str = "",
    **service_options: Any,
) -> Any:
    """A Starlette app for `ToolService(registry, **service_options)`; MCP at `prefix + mcp_path`.

    Host and anonymous options: docs/cross-language/http-server.md. MCP needs the lifespan.
    `GET /readyz` answers from `app.state.readiness` (503 before startup with MCP, and
    once shutdown or `drain()` begins).

    Raises:
        ConfigError: bad paths, options or service config; anonymous on a public bind host.
        MissingDependency: `starlette` (or, with `mcp=True`, `mcp`) is not installed.
    """
    mcp_options = dict(service_options.pop("mcp_options", None) or {})
    unknown = sorted(set(mcp_options) - MCP_OPTIONS)
    if unknown:
        raise ConfigError(f"mcp_options accepts only {sorted(MCP_OPTIONS)}, got {unknown}")
    bind = service_options.pop("host", None)
    if bind is not None:  # MCP checks its own host: a public mount needs transport_security
        mcp_options.setdefault("host", bind)
    hosts = HostPolicy(
        service_options.pop("allowed_hosts", None),
        mcp_options,
        anonymous=bool(service_options.get("allow_anonymous", False)),
        bind=bind,
        anonymous_public=bool(service_options.pop("allow_anonymous_public", False)),
    )
    _check_path(prefix, "prefix", empty_ok=True)
    _check_path(mcp_path, "mcp_path", empty_ok=False)
    _require("starlette")
    from starlette.applications import Starlette
    from starlette.routing import Route

    service = ToolService(registry, **service_options)
    routes: list[Route] = []
    subapps: list[Any] = []
    if mcp:
        mcp_app = _mcp_app(service, prefix + mcp_path, mcp_options)
        routes.append(Route(prefix + mcp_path, endpoint=_McpGate(service, mcp_app)))
        subapps.append(mcp_app)
    readiness = Readiness(needs_lifespan=bool(subapps))
    routes.append(Route(prefix + "/readyz", endpoint=_Ready(readiness, hosts)))
    routes.append(Route(prefix + "/{path:path}", endpoint=_Endpoint(service, prefix, hosts)))
    app = Starlette(routes=routes, lifespan=_lifespan(subapps, readiness))
    if hosts.local_only:
        app.add_middleware(LoopbackOnly, refuse=refuse_non_local, name="HTTP tool server")
    app.state.service = service
    app.state.hosts = hosts
    app.state.readiness = readiness
    return app


def serve(app: Any, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run `app` with uvicorn (blocking); a `create_app` app learns its real bind host.

    Raises:
        ConfigError: an anonymous `create_app` app on a non-local `host` that was built
            without `allow_anonymous_public=True` (nothing starts).
        MissingDependency: `uvicorn` is not installed.
    """
    _require("uvicorn")
    import uvicorn

    hosts = getattr(getattr(app, "state", None), "hosts", None)
    if isinstance(hosts, HostPolicy):
        hosts.bind(host)

    uvicorn.run(app, host=host, port=port)


class _Endpoint:
    """ASGI app for every `ToolService` route: bounded body read, SSE streaming."""

    def __init__(self, service: ToolService, prefix: str, hosts: HostPolicy) -> None:
        self.service = service
        self.prefix = prefix
        self.hosts = hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from starlette.requests import Request

        request = Request(scope, receive)
        result = await self._handle(request, "/" + str(scope["path_params"].get("path", "")))
        await _respond(result, self.prefix)(scope, receive, send)

    async def _handle(self, request: Any, path: str) -> ServiceResponse:
        if not self.hosts.allows(request.headers.get("host")):
            logger.warning("request refused: Host header not allowed")
            return problem_response(problem("misdirected-request", "this host is not served"))
        duplicated = _duplicated(request)
        if duplicated is not None:
            return problem_response(problem("bad-request", f"send at most one {duplicated}"))
        limit = self.service.max_body_bytes
        try:
            async with asyncio.timeout(self.service.body_timeout_s):
                body = await _read_body(request, limit)
        except TimeoutError:
            return problem_response(problem("request-timeout", "the request body was too slow"))
        if body is None:
            found = problem("payload-too-large", f"request bodies are limited to {limit} B")
            return problem_response(found)
        return await self.service.handle(request_info(request, path), body)


class _Ready:
    """`GET /readyz` (no auth): 200 while ready, 503 while starting or draining."""

    def __init__(self, readiness: Readiness, hosts: HostPolicy) -> None:
        self.readiness = readiness
        self.hosts = hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from starlette.requests import Request

        request = Request(scope)
        if not self.hosts.allows(request.headers.get("host")):
            logger.warning("request refused: Host header not allowed")
            result = problem_response(problem("misdirected-request", "this host is not served"))
        elif request.method != "GET":
            result = problem_response(problem("method-not-allowed", "use GET"))
        else:
            result = self.readiness.response()
        await _respond(result, "")(scope, receive, send)


class _McpGate:
    """Authenticates `/mcp` requests with the service's authenticator before MCP sees them."""

    def __init__(self, service: ToolService, app: Any) -> None:
        self.service = service
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from starlette.requests import Request

        request = Request(scope)
        if _duplicated(request) is not None:
            found = problem("bad-request", "send at most one authorization header")
            await _respond(problem_response(found), "")(scope, receive, send)
            return
        try:
            principal = await self.service.authenticate(request_info(request, "/mcp"))
        except Refused as exc:
            await _respond(problem_response(exc.problem), "")(scope, receive, send)
            return
        except Exception as exc:  # Unauthorized / Forbidden from the authenticator
            await _respond(problem_response(problem_for(exc)), "")(scope, receive, send)
            return
        state = dict(scope.get("state") or {})
        state[PRINCIPAL_STATE] = principal
        await self.app({**scope, "state": state}, receive, send)


class _AuthorizedRunner:
    """The service's executor for MCP calls, after the service's `authorize` hook."""

    def __init__(self, service: ToolService) -> None:
        self.service = service

    @property
    def approval(self) -> Any:
        return self.service.executor.approval

    async def visible(self, spec: ToolSpec, request_ctx: Any) -> bool:
        """Whether `tools/list` shows `spec` to this caller (`authorize` with no arguments)."""
        principal = _gate_principal(request_ctx)
        if principal is None:
            return False
        try:
            return await self.service._state.allowed(principal, spec, None)
        except Refused:  # a raising hook hides the tool (fails closed)
            return False

    async def run_spec(
        self,
        spec: ToolSpec,
        args: Mapping[str, Any] | None = None,
        ctx: ToolContext | None = None,
        *,
        call_id: str | None = None,
    ) -> Any:
        principal = mcp_principal(ctx)
        if principal is None:
            raise Forbidden("no authenticated principal for this MCP call")
        if not await self.service._state.allowed(principal, spec, args):
            logger.warning("MCP call to %s refused by authorize", spec.name)
            raise Forbidden(f"not allowed to call tool {spec.name!r}")
        if ctx is not None:
            ctx.extras["principal"] = principal  # tools see a Principal on both surfaces
        return await self.service.executor.run_spec(spec, args, ctx, call_id=call_id)


def mcp_principal(ctx: ToolContext | None) -> Principal | None:
    """The principal `_McpGate` stored on the MCP request of a tool call's `ctx`, if any."""
    native = ctx.extras.get("mcp_request") if ctx is not None else None
    return _gate_principal(native)


def _principal_hook(request_ctx: Any) -> str | None:
    """MCP's principal hook: the gate's principal id; anonymous (id "") is None."""
    principal = _gate_principal(request_ctx)
    return (principal.id or None) if principal is not None else None


def _gate_principal(request_ctx: Any) -> Principal | None:
    scope = getattr(getattr(request_ctx, "request", None), "scope", None)
    state = scope.get("state") if isinstance(scope, Mapping) else None
    principal = state.get(PRINCIPAL_STATE) if isinstance(state, Mapping) else None
    return principal if isinstance(principal, Principal) else None


def _mcp_app(service: ToolService, path: str, options: Mapping[str, Any]) -> Any:
    from agent_adapter.integrations.mcp import streamable_http_app

    return streamable_http_app(
        service.registry,
        executor=_AuthorizedRunner(service),
        path=path,
        principal=GatedPrincipal(_principal_hook),  # behind _McpGate: already authenticated
        allow_anonymous=service.allow_anonymous,
        **options,
    )


NON_LOCAL_PROBLEM = problem(
    NON_LOCAL_CLIENT, "this server is anonymous and serves local clients only"
)


async def refuse_non_local(scope: Scope, receive: Receive, send: Send) -> None:
    """The 403 `non-local-client` problem for a request the loopback guard refused."""
    await _respond(problem_response(NON_LOCAL_PROBLEM), "")(scope, receive, send)


def request_info(request: Any, path: str) -> RequestInfo:
    """The framework-neutral view of a Starlette request (`path` relative to the prefix)."""
    client = request.client
    return RequestInfo(
        method=request.method,
        path=path,
        headers=dict(request.headers.items()),
        query=dict(request.query_params.items()),
        client=client.host if client is not None else None,
    )


async def _read_body(request: Any, limit: int) -> bytes | None:
    """The body, or None as soon as it exceeds `limit` bytes (never buffered past it)."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return None
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


def _duplicated(request: Any) -> str | None:
    """The first credential or key header sent more than once (ambiguous: refused)."""
    for name in SINGLE_HEADERS:
        if len(request.headers.getlist(name)) > 1:
            return name
    return None


def _respond(result: ServiceResponse, prefix: str) -> Any:
    from starlette.responses import Response, StreamingResponse

    headers = dict(result.headers)
    if prefix and headers.get("Location", "").startswith("/"):
        headers["Location"] = prefix + headers["Location"]
    if result.stream is not None:
        return StreamingResponse(
            result.stream, status_code=result.status, headers=headers, media_type=result.media_type
        )
    try:
        content = encode_body(result)
    except ValueError:
        logger.error("response body is not JSON (status %d)", result.status)
        return _respond(problem_response(problem("internal-error", "unencodable response")), "")
    media_type = result.media_type if result.body is not None else None
    return Response(content, status_code=result.status, headers=headers, media_type=media_type)


def _lifespan(
    subapps: Sequence[Any], readiness: Readiness
) -> Callable[[Any], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for sub in subapps:
                await stack.enter_async_context(sub.router.lifespan_context(sub))
            readiness.start()
            try:
                yield
            finally:
                readiness.drain()  # before the subapps stop: probes fail first

    return lifespan


def _check_path(path: str, name: str, *, empty_ok: bool) -> None:
    if path == "" and empty_ok:
        return
    if not path.startswith("/") or path.endswith("/") or "{" in path or "//" in path:
        raise ConfigError(f"{name} must start with '/' and not end with '/', got {path!r}")


def _require(module: str) -> None:
    try:
        if module == "uvicorn":
            import uvicorn  # noqa: F401
        else:
            import starlette  # noqa: F401
    except ImportError as exc:
        logger.debug("%s is not importable: %s", module, exc)
        raise MissingDependency(f"the HTTP server needs {module}: {_HINT}") from exc


__all__ = ["PRINCIPAL_STATE", "create_app", "mcp_principal", "request_info", "serve"]
