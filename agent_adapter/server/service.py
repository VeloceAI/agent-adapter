"""ToolService: every route of the HTTP tool server, with no web framework imports."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent_adapter._version import __version__
from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import ApprovalStore, MemoryApprovalStore
from agent_adapter.cache.keys import digest
from agent_adapter.cache.memory import MemoryCache
from agent_adapter.core.errors import (
    AdapterError,
    ConfigError,
    Forbidden,
    ToolUsageError,
    Unauthorized,
    log_failure,
)
from agent_adapter.core.ports import CachePort, LedgerPort
from agent_adapter.core.registry import Registry
from agent_adapter.exec.control import is_control_flow
from agent_adapter.exec.executor import Executor
from agent_adapter.server import _budget, _calls, _reads
from agent_adapter.server._approval import ServerApproval
from agent_adapter.server._idempotency import CacheIdempotency, Replays
from agent_adapter.server._payloads import Refused, csv, descriptor, refuse, renders, shape_of
from agent_adapter.server._runs import RunOwners
from agent_adapter.server._state import (
    ServiceResponse,
    State,
    encode_body,
    json_response,
    problem_response,
)
from agent_adapter.server.auth import (
    Authenticator,
    Authorizer,
    DecisionAuthorizer,
    Principal,
    RequestInfo,
    allow_authenticated,
    owner_or_approver,
)
from agent_adapter.server.events import EventStream, MemoryEventStream
from agent_adapter.server.idempotency import IdempotencyStore, MemoryIdempotencyStore
from agent_adapter.server.openapi import Dialect, openapi_document
from agent_adapter.server.problems import problem_for
from agent_adapter.tracing.sinks import FanoutSink

__all__ = ["ANONYMOUS", "ServiceResponse", "ToolService", "encode_body"]

logger = logging.getLogger(__name__)

ANONYMOUS = Principal(id="")  # every anonymous caller shares this one identity
SINGLE_HEADERS = ("authorization", "idempotency-key")
Handler = Callable[[Principal, RequestInfo, bytes, list[str]], Awaitable[ServiceResponse]]


class ToolService:
    """The HTTP tool server's behaviour behind `handle(request, body)`.

    Tools run through `(executor or Executor(registry)).replace(...)` with a
    `DeferredApproval` over `approvals` and a sink that also feeds `events`; the host's
    executor is never changed. Paths are relative to the mount point (no prefix).
    """

    max_body_bytes = 1 << 20  # larger request bodies are refused with 413
    sse_keep_alive_s = 15.0
    sse_idle_s = 300.0
    max_listed = 100  # approvals per listing
    max_streams_per_principal = 16  # open event streams; more -> 429
    max_streams = 1024
    idempotency_lease_s = 300.0  # the longest call an Idempotency-Key lease covers
    body_timeout_s = 30.0  # reading a request body (ASGI); slower -> 408

    def __init__(
        self,
        registry: Registry,
        *,
        executor: Executor | None = None,
        approvals: ApprovalStore | None = None,
        events: EventStream | None = None,
        ledger: LedgerPort | None = None,
        idempotency: IdempotencyStore | CachePort | None = None,
        authenticate: Authenticator | None = None,
        authorize: Authorizer | None = None,
        authorize_decision: DecisionAuthorizer | None = None,
        allow_anonymous: bool = False,
        idempotency_ttl: float = 86_400,
        title: str = "agent-adapter tools",
    ) -> None:
        """Validate the configuration and build the service's own executor view.

        Raises:
            ConfigError: no `authenticate` and `allow_anonymous=False` (fail fast), or a
                non-positive `idempotency_ttl`.
            ToolUsageError: `registry`, `executor`, `approvals` or `events` of the wrong type.
        """
        if not isinstance(registry, Registry):
            raise ToolUsageError(f"ToolService needs a Registry, got {type(registry).__name__}")
        if authenticate is None and not allow_anonymous:
            raise ConfigError("ToolService needs authenticate= (or allow_anonymous=True)")
        if idempotency_ttl <= 0:
            raise ConfigError("idempotency_ttl must be > 0 seconds")
        base = executor if executor is not None else Executor(registry)
        if not isinstance(base, Executor):
            raise ToolUsageError(f"executor must be an Executor, got {type(base).__name__}")
        store = approvals if approvals is not None else _store_of(base)
        if not isinstance(store, ApprovalStore):
            raise ToolUsageError(f"approvals must be an ApprovalStore, got {type(store).__name__}")
        stream = events if events is not None else MemoryEventStream()
        if not isinstance(stream, EventStream):
            raise ToolUsageError(f"events must be an EventStream, got {type(stream).__name__}")
        sink = FanoutSink(base.sink, stream) if base.sink is not None else FanoutSink(stream)
        changes: dict[str, Any] = {"approval": ServerApproval(store), "sink": sink}
        if ledger is not None:
            changes["ledger"] = ledger
        self.registry = registry
        self.title = title
        self.allow_anonymous = allow_anonymous
        self.executor = base.replace(**changes)
        self.approvals = store
        self.events = stream
        self.ledger = self.executor.ledger
        self._authenticate = authenticate
        self._state = State(
            registry=registry,
            executor=self.executor,
            store=store,
            events=stream,
            ledger=self.ledger,
            idempotency=Replays(
                _idempotency_store(idempotency),
                ttl=idempotency_ttl,
                lease=self.idempotency_lease_s,
                namespace=base.namespace,
            ),
            owners=RunOwners(),
            authorize=authorize or allow_authenticated,
            authorize_decision=authorize_decision or owner_or_approver,
            anonymous=allow_anonymous,
            max_listed=self.max_listed,
        )
        if allow_anonymous:
            logger.warning(
                "ToolService %r accepts anonymous requests: anonymous callers share one "
                "identity, so they can see and decide each other's runs",
                title,
            )

    async def handle(self, request: RequestInfo, body: bytes) -> ServiceResponse:
        """Route one request; errors become RFC 9457 problems, never exceptions."""
        method, route = request.method.upper(), "unknown"
        try:
            handler, params, route = self._route(method, _segments(request.path))
            if handler is None:
                return json_response(200, {"status": "ok", "version": __version__})
            if len(body) > self.max_body_bytes:
                refuse(
                    "payload-too-large", f"request bodies are limited to {self.max_body_bytes} B"
                )
            _check_headers(request, json_body=method == "POST")
            principal = await self.authenticate(request)
            return await handler(principal, request, body, params)
        except Refused as exc:
            return problem_response(exc.problem)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            return self._failed(method, route, exc)

    def openapi(self, *, server_url: str | None = None, dialect: Dialect = "3.1") -> dict[str, Any]:
        """The OpenAPI document of every tool exported to `http` (see `openapi_document`)."""
        return openapi_document(
            self.registry,
            title=self.title,
            version=__version__,
            server_url=server_url,
            dialect=dialect,
        )

    async def authenticate(self, request: RequestInfo) -> Principal:
        """The caller, `ANONYMOUS` when allowed, else `Unauthorized`.

        Raises:
            Unauthorized: no principal and anonymous requests are not allowed.
            Forbidden: the authenticator refused the caller outright.
        """
        if self._authenticate is None:
            return ANONYMOUS
        try:
            principal = await self._authenticate(request)
        except (Unauthorized, Forbidden):
            raise
        except Exception as exc:
            if is_control_flow(exc):
                raise
            logger.error("authenticate hook raised %s", type(exc).__name__)  # no message: tokens
            refuse("internal-error", "authentication failed")
        if principal is not None and not isinstance(principal, Principal):
            logger.error("authenticate hook returned %s, not Principal", type(principal).__name__)
            refuse("internal-error", "authentication failed")
        if principal is None or not principal.id:
            if self.allow_anonymous:
                return ANONYMOUS
            raise Unauthorized("authentication required")
        return principal

    def _route(self, method: str, parts: list[str]) -> tuple[Handler | None, list[str], str]:
        table: dict[tuple[str, ...], dict[str, Handler]] = {
            ("tools",): {"GET": self._catalogue},
            ("tools", "*"): {"GET": self._tool},
            ("tools", "*", "invoke"): {"POST": self._invoke},
            ("approvals",): {"GET": self._approvals},
            ("approvals", "*"): {"GET": self._approval},
            ("approvals", "*", "decide"): {"POST": self._decide},
            ("runs", "*", "events"): {"GET": self._events},
            ("runs", "*", "cost"): {"GET": self._cost},
            ("budget",): {"GET": self._budget},
            ("budget", "*"): {"GET": self._budget},
            ("openapi.json",): {"GET": self._openapi},
            ("openapi-3.0.json",): {"GET": self._openapi30},
        }
        if parts == ["healthz"]:
            if method != "GET":
                refuse("method-not-allowed", "use GET")
            return None, [], "healthz"
        for pattern, methods in table.items():
            if len(pattern) == len(parts) and all(
                p == "*" or p == part for p, part in zip(pattern, parts, strict=True)
            ):
                if method not in methods:
                    refuse("method-not-allowed", f"use {', '.join(methods)}")
                params = [part for p, part in zip(pattern, parts, strict=True) if p == "*"]
                route = "/".join("{}" if p == "*" else p for p in pattern)
                return methods[method], params, route
        refuse("not-found", "no such route")

    async def _catalogue(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        shape = shape_of(request.query)
        tools = self._state.tools().subset(
            tags=csv(request.query.get("tags")) or None,
            kinds=csv(request.query.get("kinds")) or None,
        )
        items = [
            descriptor(self._state.listed(spec), shape)
            for spec in tools
            if renders(spec, shape) and await self._state.allowed(principal, spec, None)
        ]
        catalogue = {"version": "1", "tools": items}
        etag = f'"{digest(catalogue)}"'
        if request.header("if-none-match") == etag:
            return ServiceResponse(status=304, headers={"ETag": etag})
        return json_response(200, catalogue, {"ETag": etag})

    async def _tool(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        shape = shape_of(request.query)
        spec = self._state.spec(params[0])
        if not await self._state.allowed(principal, spec, None):
            refuse("forbidden", "not allowed to see this tool", tool=spec.name)
        if not renders(spec, shape):
            refuse("tool-not-found", f"tool {spec.name!r} is not exported to {shape}")
        return json_response(200, descriptor(self._state.listed(spec), shape))

    async def _invoke(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        return await _calls.invoke(self._state, principal, request, body, params[0])

    async def _approvals(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        return await _reads.list_approvals(self._state, principal, request)

    async def _approval(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        return await _reads.get_approval(self._state, principal, params[0])

    async def _decide(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        return await _calls.decide(self._state, principal, body, params[0])

    async def _events(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        return await _reads.run_events(
            self._state,
            principal,
            request,
            params[0],
            keep_alive_s=self.sse_keep_alive_s,
            idle_s=self.sse_idle_s,
            per_principal=self.max_streams_per_principal,
            total=self.max_streams,
        )

    async def _cost(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        return await _reads.run_cost(self._state, principal, params[0])

    async def _budget(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        return await _budget.budgets(self._state, principal, request, params[0] if params else None)

    async def _openapi(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        return await self._visible_openapi(principal, "3.1")

    async def _openapi30(
        self, principal: Principal, request: RequestInfo, body: bytes, params: list[str]
    ) -> ServiceResponse:
        return await self._visible_openapi(principal, "3.0")

    async def _visible_openapi(self, principal: Principal, dialect: Dialect) -> ServiceResponse:
        """The document of the tools `principal` may call, in `dialect`."""
        tools = self._state.tools()
        visible = [s.name for s in tools if await self._state.allowed(principal, s, None)]
        document = openapi_document(
            tools.subset(names=visible), title=self.title, version=__version__, dialect=dialect
        )
        return json_response(200, document)

    def _failed(self, method: str, route: str, exc: Exception) -> ServiceResponse:
        found = problem_for(exc)  # logs name the route, never the client's path
        if isinstance(exc, AdapterError):
            logger.info("%s %s -> %d %s", method, route, found.status, exc.kind)
        else:
            log_failure(logger, logging.ERROR, exc, "%s %s failed", method, route)
        return problem_response(found)


def _check_headers(request: RequestInfo, *, json_body: bool) -> None:
    """Refuse ambiguous credentials or keys (400) and non-JSON POST bodies (415, no CSRF)."""
    for name in SINGLE_HEADERS:
        if sum(1 for key in request.headers if key.lower() == name) > 1:
            refuse("bad-request", f"send at most one {name} header")
    if json_body:
        media_type = (request.header("content-type") or "").split(";")[0].strip().lower()
        if media_type != "application/json":
            refuse("unsupported-media-type", "send the body as application/json")


def _segments(path: str) -> list[str]:
    return [part for part in path.split("/") if part]


def _idempotency_store(given: IdempotencyStore | CachePort | None) -> IdempotencyStore:
    if given is None:
        return MemoryIdempotencyStore()
    if isinstance(given, IdempotencyStore):
        return given
    if not isinstance(given, CachePort):
        raise ToolUsageError(
            f"idempotency must be an IdempotencyStore or a CachePort, got {type(given).__name__}"
        )
    if not isinstance(given, MemoryCache):
        logger.warning(
            "idempotency=%s is a plain CachePort: replays are best effort across workers "
            "(two workers can run one key); pass an IdempotencyStore such as "
            "RedisIdempotencyStore",
            type(given).__name__,
        )
    return CacheIdempotency(given)


def _store_of(executor: Executor) -> ApprovalStore:
    port = executor.approval
    return port.store if isinstance(port, DeferredApproval) else MemoryApprovalStore()
