"""Serve registry tools over MCP: a v2 lowlevel `Server` and a stateless Streamable HTTP app."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from agent_adapter._version import __version__
from agent_adapter.approval.builtin import AutoApprove
from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import ApprovalStore, MemoryApprovalStore
from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.exec.defaults import runner_for
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.mcp._approval import ClientApproval, install_state_boundary
from agent_adapter.integrations.mcp._handlers import (
    TARGET,
    Handlers,
    mcp_tool,
    require_mcp,
)
from agent_adapter.integrations.mcp._peer import NON_LOCAL_CLIENT, LoopbackOnly
from agent_adapter.integrations.mcp._scope import (
    GatedPrincipal,
    PrincipalHook,
    RequiredPrincipal,
    principal_of,
)
from agent_adapter.integrations.mcp.tasks import TaskOptions

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server

__all__ = [
    "AUTHENTICATED",
    "TARGET",
    "mcp_server",
    "mcp_tool",
    "require_mcp",
    "streamable_http_app",
]

logger = logging.getLogger(__name__)


class _Authenticated:
    """Default `principal` of the HTTP app: the SDK's bearer-token identity."""

    def __repr__(self) -> str:
        return "AUTHENTICATED"


AUTHENTICATED: Any = _Authenticated()
DEFAULT_NAME = "agent-adapter"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_FIXED_HTTP_OPTIONS = frozenset({"streamable_http_path", "json_response", "stateless_http"})


def mcp_server(
    registry: Registry,
    *,
    executor: ToolRunner | None = None,
    name: str = "agent-adapter",
    version: str = __version__,
    instructions: str | None = None,
    client_approval: bool = False,
    request_state_keys: Sequence[bytes | str] | None = None,
    principal: PrincipalHook | None = None,
    allow_anonymous: bool | None = None,
    approval_store: ApprovalStore | None = None,
    tasks: TaskOptions | bool = False,
) -> Server:
    """A lowlevel MCP `Server` listing and calling `registry.for_target("mcp")`.

    Run and call ids are scoped by the caller's principal. Each option is described in
    `docs/integrations/mcp.md` (Security and Client approval); `tasks` (the 2026-07-28
    tasks extension) in `docs/integrations/mcp-tasks.md`.

    Raises:
        ToolUsageError: bad `registry` or `principal`, or keys without `client_approval`.
        ConfigError: bad `request_state_keys`, `approval_store` or `tasks` (see the docs).
        MissingDependency: `mcp` is not installed.
    """
    require_mcp()  # before the option checks, which log
    task_options = _task_options(tasks)
    if not isinstance(registry, Registry):
        raise ToolUsageError(f"mcp_server needs a Registry, got {registry!r}")
    if principal is not None and not callable(principal):
        raise ToolUsageError(f"principal must be callable or None, got {principal!r}")
    if request_state_keys is not None and not client_approval:
        raise ToolUsageError("request_state_keys only applies with client_approval=True")
    if request_state_keys is not None and name == DEFAULT_NAME:
        raise ConfigError("request_state_keys need a unique name= (the requestState audience)")
    _check_approval_store(approval_store, client_approval)
    shared = _shared_store(approval_store, executor)
    if request_state_keys is not None and shared is None:
        raise ConfigError(
            "request_state_keys need a shared ApprovalStore so each approval state is used once"
        )
    if request_state_keys is not None and isinstance(shared, MemoryApprovalStore):
        logger.warning(
            "MCP server %r: request_state_keys with a MemoryApprovalStore, which is per "
            "process: a state replayed on another worker is not refused",
            name,
        )
    from mcp.server.lowlevel import Server

    approval = ClientApproval(approval_store) if client_approval else None
    if client_approval:
        _warn_port_replaced(executor)
    service = None
    if task_options is not None:
        from agent_adapter.integrations.mcp._task_service import TaskService

        service = TaskService(
            task_options,
            principal=principal,
            allow_anonymous=allow_anonymous,
            store=task_options.store,
        )
    handlers = Handlers(
        registry,
        executor,
        approval=approval,
        principal=principal,
        allow_anonymous=allow_anonymous,
        tasks=service,
    )
    extra: dict[str, Any] = {} if service is None else {"lifespan": service.lifespan}
    server = Server(
        name,
        version=version,
        instructions=instructions,
        on_list_tools=_listing(handlers, principal),
        on_call_tool=handlers.call_tool,
        get_tool_input_schema=handlers.input_schema,
        **extra,
    )
    if service is not None:
        service.install(server)
    if client_approval:
        install_state_boundary(server, audience=name, keys=request_state_keys, principal=principal)
    return server


def streamable_http_app(
    registry: Registry,
    *,
    executor: ToolRunner | None = None,
    path: str = "/mcp",
    stateless: bool = True,
    json_response: bool = True,
    name: str = "agent-adapter",
    version: str = __version__,
    instructions: str | None = None,
    client_approval: bool = False,
    request_state_keys: Sequence[bytes | str] | None = None,
    principal: PrincipalHook | None = AUTHENTICATED,
    allow_anonymous: bool = False,
    approval_store: ApprovalStore | None = None,
    tasks: TaskOptions | bool = False,
    **http_options: Any,
) -> Any:
    """An ASGI (Starlette) app serving `mcp_server(registry, ...)` over Streamable HTTP at `path`.

    `principal` defaults to the SDK's bearer-token identity (client id, issuer, subject);
    without one, client keys are ignored unless `allow_anonymous=True`. `http_options` go to
    `Server.streamable_http_app` (`host=`, `transport_security=`, `auth=`,
    `token_verifier=`, ...). The app's lifespan must run (uvicorn does).

    Without auth (`auth=` + `token_verifier=`) any client that reaches the port can call
    every tool: a non-local `host` then needs `allow_anonymous=True`, and on a local `host`
    requests from a non-loopback peer are refused (403). A custom `principal` is not auth;
    unless `allow_anonymous=True`, a request it names no caller for is refused.

    Raises:
        ConfigError: an `http_options` key this function sets; a non-local `host` without
            `transport_security`, or without auth or `allow_anonymous=True`; `auth=` or
            `token_verifier=` alone.
        MissingDependency: `mcp` is not installed.
    """
    require_mcp()  # before the auth checks, which log
    clash = sorted(_FIXED_HTTP_OPTIONS & set(http_options))
    if clash:
        raise ConfigError(
            f"streamable_http_app sets {clash} itself; use path/stateless/json_response"
        )
    host = http_options.get("host", "127.0.0.1")
    if host not in LOCAL_HOSTS and http_options.get("transport_security") is None:
        raise ConfigError(
            f"host {host!r} is not local: pass transport_security= with its allowed hosts"
        )
    local_only = _check_http_auth(name, host, principal, allow_anonymous, http_options)
    hook = _default_principal() if principal is AUTHENTICATED else principal
    custom = principal is not AUTHENTICATED and not isinstance(principal, GatedPrincipal)
    if custom and hook is not None and not allow_anonymous:
        hook = RequiredPrincipal(hook)
    if allow_anonymous:
        logger.warning(
            "MCP HTTP app %r accepts anonymous run and call keys: clients can share runs", name
        )
    server = mcp_server(
        registry,
        executor=executor,
        name=name,
        version=version,
        instructions=instructions,
        client_approval=client_approval,
        request_state_keys=request_state_keys,
        principal=hook,
        allow_anonymous=allow_anonymous,
        approval_store=approval_store,
        tasks=tasks,
    )
    app = server.streamable_http_app(
        streamable_http_path=path,
        json_response=json_response,
        stateless_http=stateless,
        **http_options,
    )
    if local_only:
        app.add_middleware(LoopbackOnly, refuse=_refuse_non_local, name=f"MCP HTTP app {name!r}")
    return app


def _listing(handlers: Handlers, principal: PrincipalHook | None) -> Any:
    """`handlers.list_tools`, refusing first when a `RequiredPrincipal` names no caller."""
    if not isinstance(principal, RequiredPrincipal):
        return handlers.list_tools

    async def list_tools(ctx: Any, params: Any) -> Any:
        principal_of(ctx, principal)
        return await handlers.list_tools(ctx, params)

    return list_tools


def _check_http_auth(
    name: str,
    host: str,
    principal: PrincipalHook | None,
    allow_anonymous: bool,
    http_options: dict[str, Any],
) -> bool:
    """Refuse an unauthenticated app on a public host unless anonymous access is explicit.

    Returns:
        True when the app is unauthenticated on a local host, so it must serve loopback only.
    """
    auth, verifier = http_options.get("auth"), http_options.get("token_verifier")
    if (auth is None) != (verifier is None):
        # the SDK installs its bearer-token middleware only when both are given
        raise ConfigError("auth= and token_verifier= go together: either alone checks no token")
    if auth is not None or isinstance(principal, GatedPrincipal):
        return False
    if host in LOCAL_HOSTS:
        logger.warning(
            "MCP HTTP app %r has no auth: any client that reaches the port can call every "
            "tool; non-loopback clients are refused",
            name,
        )
        return True
    if not allow_anonymous:
        raise ConfigError(
            f"host {host!r} is not local and the app has no auth: pass auth= and "
            "token_verifier=, or allow_anonymous=True"
        )
    logger.warning(
        "MCP HTTP app %r on host %r has no auth: any client that reaches the port can call "
        "every tool",
        name,
        host,
    )
    return False


async def _refuse_non_local(scope: Any, receive: Any, send: Any) -> None:
    # the error shape of the SDK's own bearer-auth refusals, sent as raw ASGI messages
    body = json.dumps(
        {
            "error": NON_LOCAL_CLIENT,
            "error_description": "this MCP server has no auth and serves local clients only",
        }
    ).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", b"%d" % len(body))]
    await send({"type": "http.response.start", "status": 403, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _default_principal() -> Callable[..., str | None]:
    require_mcp()
    from mcp.server.request_state import authenticated_principal

    return cast("Callable[..., str | None]", authenticated_principal)


def _task_options(tasks: object) -> TaskOptions | None:
    if tasks is True:
        return TaskOptions()
    if tasks is False:
        return None
    if isinstance(tasks, TaskOptions):
        return tasks
    raise ConfigError(f"tasks must be a bool or TaskOptions, got {type(tasks).__name__}")


def _check_approval_store(store: Any, client_approval: bool) -> None:
    if store is None:
        return
    if not client_approval:
        raise ConfigError("approval_store only applies with client_approval=True")
    usable = isinstance(store, ApprovalStore) and all(
        callable(getattr(store, method)) for method in ("create", "find", "decide", "consume")
    )
    if not usable:
        raise ConfigError(f"approval_store must be an ApprovalStore, got {type(store).__name__}")


def _shared_store(
    approval_store: ApprovalStore | None, executor: ToolRunner | None
) -> ApprovalStore | None:
    if approval_store is not None:
        return approval_store
    port = getattr(runner_for(executor), "approval", None)
    return port.store if isinstance(port, DeferredApproval) else None


def _warn_port_replaced(executor: ToolRunner | None) -> None:
    port = getattr(runner_for(executor), "approval", None)
    if port is not None and not isinstance(port, AutoApprove):
        logger.warning(
            "client_approval=True: MCP clients that can elicit approve confirm tools "
            "themselves; the executor's %s is not asked for them",
            type(port).__name__,
        )
