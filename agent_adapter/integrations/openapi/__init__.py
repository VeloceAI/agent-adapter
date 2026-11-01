"""OpenAPI 3.x importer: one tool per operation, called over httpx."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agent_adapter.core.errors import ConfigError, IntegrationError
from agent_adapter.core.imported import spec_from_schema
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations.base import IntegrationBase, ToolRunner
from agent_adapter.integrations.openapi._document import load_document, server_url
from agent_adapter.integrations.openapi._http import (
    CONTEXT_PARAM,
    IDEMPOTENT_METHODS,
    OperationCall,
)
from agent_adapter.integrations.openapi._refs import check_document
from agent_adapter.integrations.openapi._schema import Operation, build_operations
from agent_adapter.integrations.openapi.auth import Auth, BearerAuth, HeaderAuth, QueryAuth
from agent_adapter.runtime.loop_local import LoopLocal

__all__ = [
    "INTEGRATION",
    "Auth",
    "BearerAuth",
    "HeaderAuth",
    "OpenAPIIntegration",
    "QueryAuth",
    "import_openapi",
]

TARGET = "openapi"
_READ_ONLY = frozenset({"get", "head"})


def import_openapi(
    registry: Registry,
    spec: Mapping[str, Any] | str | os.PathLike[str],
    *,
    base_url: str | None = None,
    auth: Auth | None = None,
    headers: Mapping[str, str] | None = None,
    operations: Iterable[str] | None = None,
    prefix: str = "",
    timeout: float = 30.0,
    client_factory: Callable[..., Any] | None = None,
    **policy: Any,
) -> list[ToolSpec]:
    """Add one tool per operation of an OpenAPI 3.x document (all or nothing).

    `spec` is a mapping, JSON/YAML text or a file path. `client_factory(base_url=,
    headers=, timeout=)` makes the `httpx.AsyncClient` (default: `httpx.AsyncClient`), once
    per event loop. 4xx answers (except 429) raise `ToolError`; 429 and 5xx raise
    `httpx.HTTPStatusError`, which the executor retries (`retries=`) before `ToolFailed`.

    Raises:
        ConfigError: bad document, options, server URL or unknown `operations`.
        DuplicateTool: a tool name is already registered.
        MissingDependency: httpx (or pyyaml for YAML) is not installed.
    """
    INTEGRATION.require()
    _check_options(auth, timeout, prefix)
    document = load_document(spec)
    url = server_url(document, base_url)
    factory = client_factory if client_factory is not None else _httpx_client
    clients: LoopLocal[Any] = LoopLocal(
        functools.partial(factory, base_url=url, headers=dict(headers or {}), timeout=timeout),
        close=_aclose,
    )
    values = check_document(document)  # the size the inlining budget is relative to
    found = build_operations(document, prefix=prefix, only=operations, values=values)
    calls = functools.partial(
        OperationCall, clients=clients, auth=auth, base_url=url, timeout=timeout
    )
    return registry.add_many(_spec(op, calls, policy) for op in found)


def _check_options(auth: Any, timeout: Any, prefix: Any) -> None:
    if auth is not None and not isinstance(auth, Auth):
        raise ConfigError(f"auth must be BearerAuth, HeaderAuth, QueryAuth or None, got {auth!r}")
    if isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0:
        raise ConfigError(f"timeout must be a positive number of seconds, got {timeout!r}")
    if not isinstance(prefix, str):
        raise ConfigError(f"prefix must be a string, got {prefix!r}")


def _spec(
    op: Operation, calls: Callable[..., OperationCall], policy: Mapping[str, Any]
) -> ToolSpec:
    options: dict[str, Any] = {
        "kind": "http",
        "tags": {"openapi", *op.tags},
        "effects": _effects(op.method),
        "idempotent": op.method in IDEMPOTENT_METHODS,
        **policy,
    }
    # `idempotent=True` in the policy also opts a POST/PATCH into retries of 5xx answers.
    call = calls(op, idempotent=options["idempotent"] is True)
    return spec_from_schema(
        op.name,
        op.description,
        op.input_schema,
        call,
        output_schema=op.output_schema,
        context_param=CONTEXT_PARAM,  # the executor's attempt info, for Retry-After
        **options,
    )


def _effects(method: str) -> str:
    if method in _READ_ONLY:
        return "read_only"
    return "destructive" if method == "delete" else "additive"


def _httpx_client(**options: Any) -> Any:
    import httpx

    return httpx.AsyncClient(**options)


async def _aclose(client: Any) -> None:
    await client.aclose()


class OpenAPIIntegration(IntegrationBase):
    """Imports tools (`import_openapi`); it has no tool export."""

    name = TARGET
    requires = ("httpx",)
    extra = "openapi"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """Not supported: OpenAPI is an import source.

        Raises:
            IntegrationError: always; the HTTP tool server publishes `/openapi.json`.
        """
        raise IntegrationError(
            "the openapi integration only imports tools; the HTTP tool server "
            "(agent_adapter.server) publishes registry tools as /openapi.json",
            integration=self.name,
        )


INTEGRATION = OpenAPIIntegration()
