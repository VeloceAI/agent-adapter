"""Serve registry tools as an A2A agent: a Starlette app with auth, host and body guards."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from agent_adapter._version import __version__
from agent_adapter.core.errors import ConfigError, Forbidden, ToolUsageError, log_failure
from agent_adapter.core.registry import Registry
from agent_adapter.integrations.a2a._task_store import bounded_task_store
from agent_adapter.integrations.a2a._wire import MAX_DATA_BYTES, require_a2a
from agent_adapter.integrations.a2a.agent import PRINCIPAL_STATE, Authorizer, a2a_executor
from agent_adapter.integrations.a2a.card import agent_card
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.mcp._peer import NON_LOCAL_CLIENT, LoopbackOnly

logger = logging.getLogger(__name__)

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_BODY_BYTES = 1024 * 1024
BODY_TIMEOUT_S = 30.0
CARD_PATH = "/.well-known/agent-card.json"
Authenticator = Callable[[Any], Awaitable[Any]]  # Starlette Request -> principal (has `.id`)


@dataclass(frozen=True)
class Caller:
    """A minimal principal: `id` scopes tasks, runs and approvals (`""` is anonymous)."""

    id: str
    scopes: frozenset[str] = field(default_factory=frozenset)


ANONYMOUS = Caller("")


def static_bearer(tokens: Mapping[str, str]) -> Authenticator:
    """An authenticator for `Authorization: Bearer <token>` over a `{token: caller id}` table.

    Raises:
        ToolUsageError: an empty token or caller id.
    """
    table: dict[str, Caller] = {}
    for token, caller in tokens.items():
        if not isinstance(token, str) or not token.strip() or not isinstance(caller, str):
            raise ToolUsageError("static_bearer maps non-empty token strings to caller ids")
        if not caller:
            raise ToolUsageError("static_bearer caller ids must be non-empty")
        table[_sha256(token.strip())] = Caller(caller)

    async def authenticate(request: Any) -> Caller | None:
        scheme, _, token = (request.headers.get("authorization") or "").strip().partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        wanted = _sha256(token.strip())
        for digest, caller in table.items():  # constant time per entry
            if hmac.compare_digest(digest, wanted):
                return caller
        return None

    return authenticate


def a2a_app(
    registry: Registry,
    *,
    url: str,
    executor: ToolRunner | None = None,
    authenticate: Authenticator | None = None,
    authorize: Authorizer | None = None,
    allow_anonymous: bool = False,
    host: str = "127.0.0.1",
    allowed_hosts: Iterable[str] | None = None,
    public_card: bool = False,
    client_approval: bool = False,
    name: str = "agent-adapter",
    description: str = "Tools served by agent-adapter.",
    version: str = __version__,
    max_body_bytes: int = MAX_BODY_BYTES,
    max_data_bytes: int = MAX_DATA_BYTES,
    task_store: Any = None,
) -> Any:
    """A Starlette app: the agent card at `CARD_PATH`, JSON-RPC A2A at `/`.

    `url` is the public JSON-RPC URL written in the card. Without `authenticate` any
    client that reaches the port can call every tool: a non-local `host` then needs
    `allow_anonymous=True`, and a local one serves loopback peers only. Tasks, runs and
    approvals are scoped by the principal's `id`. `task_store` (an SDK `TaskStore`) defaults
    to `bounded_task_store()`. Options: `docs/integrations/a2a.md`.

    Raises:
        ConfigError: a non-local `host` without `allowed_hosts`, or without auth or
            `allow_anonymous=True`; bad limits or `url`.
        MissingDependency: `a2a-sdk` (with its `http-server` extra) is not installed.
    """
    require_a2a()
    if not isinstance(registry, Registry):
        raise ToolUsageError(f"a2a_app needs a Registry, got {registry!r}")
    if authenticate is not None and not callable(authenticate):
        raise ToolUsageError(f"authenticate must be callable or None, got {authenticate!r}")
    if not (isinstance(max_body_bytes, int) and max_body_bytes > 0):
        raise ConfigError(f"max_body_bytes must be a positive int, got {max_body_bytes!r}")
    hosts = _allowed_hosts(host, allowed_hosts)
    local_only = _check_auth(name, host, authenticate, allow_anonymous)
    if client_approval:
        logger.warning(
            "A2A app %r: client_approval=True lets A2A clients approve confirm tools themselves",
            name,
        )
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
    from starlette.applications import Starlette

    card = agent_card(registry, url=url, name=name, description=description, version=version)
    agent = a2a_executor(
        registry,
        executor=executor,
        client_approval=client_approval,
        authorize=authorize,
        max_data_bytes=max_data_bytes,
    )
    store = bounded_task_store() if task_store is None else task_store
    handler = DefaultRequestHandler(agent_executor=agent, task_store=store, agent_card=card)

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await handler.aclose()

    routes = [
        *create_agent_card_routes(card, card_url=CARD_PATH),
        *create_jsonrpc_routes(handler, "/", context_builder=_builder_class()()),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.add_middleware(
        _Gate,
        authenticate=authenticate,
        allow_anonymous=allow_anonymous,
        hosts=hosts,
        public_card=public_card,
        max_body_bytes=max_body_bytes,
        name=name,
    )
    if local_only:
        app.add_middleware(LoopbackOnly, refuse=_refuse_non_local, name=f"A2A app {name!r}")
    app.state.a2a_agent = agent
    app.state.a2a_card = card
    app.state.a2a_task_store = store
    return app


def _allowed_hosts(host: str, allowed: Iterable[str] | None) -> frozenset[str]:
    if allowed is not None:
        names = frozenset(name.lower() for name in allowed)
        if not names or not all(isinstance(name, str) and name for name in names):
            raise ConfigError("allowed_hosts must name at least one host")
        return names
    if host not in LOCAL_HOSTS:
        raise ConfigError(f"host {host!r} is not local: pass allowed_hosts= (Host header names)")
    return LOCAL_HOSTS


def _check_auth(
    name: str, host: str, authenticate: Authenticator | None, allow_anonymous: bool
) -> bool:
    """True when the app is anonymous on a local host, so it must serve loopback only."""
    if authenticate is not None:
        if allow_anonymous:
            logger.warning("A2A app %r: requests without credentials run as anonymous", name)
        return False
    if host in LOCAL_HOSTS and not allow_anonymous:
        logger.warning(
            "A2A app %r has no auth: any local client can call every tool; non-loopback "
            "clients are refused",
            name,
        )
        return True
    if not allow_anonymous:
        raise ConfigError(
            f"host {host!r} is not local and the A2A app has no auth: pass authenticate=, "
            "or allow_anonymous=True"
        )
    logger.warning("A2A app %r on host %r has no auth: anyone can call every tool", name, host)
    return False


class _Gate:
    """ASGI middleware: Host allow-list, bounded body, authentication, principal in scope."""

    def __init__(
        self,
        app: Any,
        *,
        authenticate: Authenticator | None,
        allow_anonymous: bool,
        hosts: frozenset[str],
        public_card: bool,
        max_body_bytes: int,
        name: str,
    ) -> None:
        self.app = app
        self.authenticate = authenticate
        self.allow_anonymous = allow_anonymous
        self.hosts = hosts
        self.public_card = public_card
        self.max_body_bytes = max_body_bytes
        self.name = name

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        from starlette.requests import Request

        request = Request(scope)
        if _host_name(request.headers.get("host")) not in self.hosts:
            logger.warning("A2A app %r refused a request: Host header not allowed", self.name)
            await _error(send, 421, "misdirected-request", "this host is not served")
            return
        body = await self._body(scope, receive, send)
        if body is None:
            return
        replay = _replay(body)
        is_card = scope.get("method") == "GET" and scope.get("path") == CARD_PATH
        principal: Any = ANONYMOUS
        if self.authenticate is not None and not (is_card and self.public_card):
            principal = await self._principal(Request(scope, replay), send)
            if principal is None:
                return
        await self.app({**scope, PRINCIPAL_STATE: principal}, _replay(body), send)

    async def _body(self, scope: Any, receive: Any, send: Any) -> bytes | None:
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_body_bytes):
            await _error(send, 413, "payload-too-large", f"limit {self.max_body_bytes} B")
            return None
        chunks: list[bytes] = []
        size = 0
        try:
            async with asyncio.timeout(BODY_TIMEOUT_S):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return None
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > self.max_body_bytes:
                        await _error(
                            send, 413, "payload-too-large", f"limit {self.max_body_bytes} B"
                        )
                        return None
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        return b"".join(chunks)
        except TimeoutError:
            await _error(send, 408, "request-timeout", "the request body was too slow")
            return None

    async def _principal(self, request: Any, send: Any) -> Any:
        assert self.authenticate is not None
        try:
            principal = await self.authenticate(request)
        except Forbidden:
            await _error(send, 403, "forbidden", "not allowed")
            return None
        except Exception as exc:  # Unauthorized, or a failing verifier: refuse (fails closed)
            log_failure(
                logger, logging.WARNING, exc, "A2A app %r: authentication failed", self.name
            )
            await _error(send, 401, "unauthorized", "invalid credentials")
            return None
        if principal is None:
            if self.allow_anonymous:
                return ANONYMOUS
            await _error(send, 401, "unauthorized", "credentials required")
            return None
        if not isinstance(getattr(principal, "id", None), str):
            logger.error(
                "A2A app %r: authenticate returned %s", self.name, type(principal).__name__
            )
            await _error(send, 500, "internal-error", "authentication misconfigured")
            return None
        return principal


def _host_name(header: str | None) -> str:
    """The host of a Host header, without port or IPv6 brackets, lower-cased."""
    value = (header or "").strip().lower()
    if value.startswith("["):
        return value[1:].partition("]")[0]
    return value.rpartition(":")[0] if value.count(":") == 1 else value


def _replay(body: bytes) -> Callable[[], Awaitable[dict[str, Any]]]:
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            await asyncio.Event().wait()  # like a real server: block until disconnect
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _error(send: Any, status: int, kind: str, detail: str) -> None:
    body = json.dumps({"error": kind, "error_description": detail}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", b"%d" % len(body))]
    if status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _refuse_non_local(scope: Any, receive: Any, send: Any) -> None:
    await _error(send, 403, NON_LOCAL_CLIENT, "this A2A agent has no auth and serves local clients")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


@functools.cache
def _builder_class() -> type[Any]:
    from a2a.auth.user import User
    from a2a.extensions.common import HTTP_EXTENSION_HEADER, get_requested_extensions
    from a2a.server.context import ServerCallContext
    from a2a.server.routes import ServerCallContextBuilder
    from a2a.utils.constants import VERSION_HEADER

    class PrincipalUser(User):  # type: ignore[misc,unused-ignore]
        """The gate's principal as the SDK's user; `user_name` is the task-store owner."""

        def __init__(self, principal: Any) -> None:
            self.principal = principal

        @property
        def is_authenticated(self) -> bool:
            return bool(self.principal.id)

        @property
        def user_name(self) -> str:
            name: str = self.principal.id
            return name

    class PrincipalContextBuilder(ServerCallContextBuilder):  # type: ignore[misc,unused-ignore]
        """A call context holding the gate's principal and only the protocol headers."""

        def build(self, request: Any) -> ServerCallContext:
            principal = request.scope.get(PRINCIPAL_STATE, ANONYMOUS)
            # Never the credentials: the SDK logs call contexts at DEBUG.
            headers = {
                name: request.headers[name]
                for name in (VERSION_HEADER.lower(), HTTP_EXTENSION_HEADER.lower())
                if name in request.headers
            }
            return ServerCallContext(
                user=PrincipalUser(principal),
                state={PRINCIPAL_STATE: principal, "headers": headers},
                requested_extensions=get_requested_extensions(
                    request.headers.getlist(HTTP_EXTENSION_HEADER)
                ),
            )

    return PrincipalContextBuilder
