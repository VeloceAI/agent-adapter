"""Import tools from an MCP server, and build an `mcp.Client` for a URL, command or server."""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, IntegrationError, ToolError, ToolUsageError
from agent_adapter.core.imported import spec_from_schema
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations.mcp._handlers import require_mcp
from agent_adapter.integrations.mcp.annotations import (
    APPROVAL_ID_META,
    CALL_ID_META,
    ERROR_KIND_META,
    RUN_ID_META,
    annotation_options,
    is_wrapper_schema,
    unwrap_output_schema,
    unwrap_structured,
)
from agent_adapter.integrations.mcp.tasks import TASK_ID_META

if TYPE_CHECKING:
    import httpx2
    import mcp.types as t
    from mcp import Client
    from mcp.types import RequestParamsMeta

logger = logging.getLogger(__name__)

CONTEXT_PARAM = "aa_ctx"  # keyword the imported tool's function receives its context under
MAX_RESUMES = 10_000  # pending remote approvals remembered per imported tool
MAX_PAGES = 1000  # tools/list pages read before giving up on a server that never ends
HTTP_TIMEOUT_S = 30.0
HTTP_READ_TIMEOUT_S = 300.0  # a server may hold a response stream open
DEFAULT_TASK_WAIT_S = 30.0  # how long a call follows a remote task before it can be resumed
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]")


async def import_mcp(
    registry: Registry,
    client: Client,
    *,
    prefix: str = "",
    names: Iterable[str] | None = None,
    send_ids: bool = False,
    **policy: Any,
) -> list[ToolSpec]:
    """Register the tools of an open `mcp.Client` (all of them, or the remote `names`).

    The tools call the server through `client`, so they work only while it stays open.

    Args:
        prefix: Prepended to each local name; characters other than letters, digits and
            `_` become `_`.
        send_ids: Send the local run id as the remote run key (servers you trust). The
            resume keys of a remote pending approval are always sent back on the retry.
        **policy: Execution options as for `build_spec`; they override the defaults
            `kind="mcp"`, `tags={"mcp"}` and the `effects` / `idempotent` the remote
            annotations imply.

    Raises:
        IntegrationError: a requested name is not on the server, or listing never ends.
        ToolDefinitionError, DuplicateTool, SchemaNotPortable: as for `registry.add_many`;
            nothing is registered then.
    """
    tools = await _list_tools(client)
    wanted = None if names is None else ({names} if isinstance(names, str) else set(names))
    if wanted is not None:
        missing = sorted(wanted - {tool.name for tool in tools})
        if missing:
            logger.warning("MCP server has no tools named %s", missing)
            raise IntegrationError(
                f"the MCP server has no tools named {missing}", integration="mcp"
            )
    specs = [
        _imported_spec(client, tool, prefix, policy, send_ids)
        for tool in tools
        if wanted is None or tool.name in wanted
    ]
    registry.add_many(specs)
    logger.info("imported %d MCP tools: %s", len(specs), [spec.name for spec in specs])
    return specs


async def _list_tools(client: Client) -> list[t.Tool]:
    tools: list[t.Tool] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(MAX_PAGES):
        page = await client.list_tools(cursor=cursor)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor:
            return tools
        if cursor in seen:
            break
        seen.add(cursor)
    logger.error("MCP tools/list did not end (last cursor repeated or %d pages)", MAX_PAGES)
    raise IntegrationError("the MCP server's tool listing never ends", integration="mcp")


def _imported_spec(
    client: Client, tool: t.Tool, prefix: str, policy: Mapping[str, Any], send_ids: bool
) -> ToolSpec:
    local = _INVALID_NAME_CHARS.sub("_", prefix + tool.name)
    output_schema = tool.output_schema
    if output_schema is not None and is_wrapper_schema(output_schema):
        output_schema = unwrap_output_schema(output_schema)
    properties = tool.input_schema.get("properties") or {}
    context_param = CONTEXT_PARAM if CONTEXT_PARAM not in properties else None
    options: dict[str, Any] = {
        "kind": "mcp",
        "tags": frozenset({"mcp"}),
        **annotation_options(tool.annotations),
        **policy,
    }
    return spec_from_schema(
        local,
        tool.description or tool.title or tool.name,
        tool.input_schema,
        _RemoteTool(client, tool.name, local, tool.output_schema, context_param, send_ids),
        output_schema=output_schema,
        context_param=context_param,
        **options,
    )


class _RemoteTool:
    """Calls one remote tool; a remote pending approval is resumed with the keys it named."""

    def __init__(
        self,
        client: Client,
        remote: str,
        local: str,
        output_schema: Mapping[str, Any] | None,
        context_param: str | None,
        send_ids: bool,
    ) -> None:
        self.client = client
        self.remote = remote
        self.local = local
        self.output_schema = output_schema
        self.context_param = context_param
        self.send_ids = send_ids
        self.__name__ = local
        # (local run id, local call id) -> the remote's resume `_meta` of a pending result
        self._resume: OrderedDict[tuple[str, str], dict[str, str]] = OrderedDict()
        self._lock = threading.Lock()

    async def __call__(self, **kwargs: Any) -> Any:
        ctx = kwargs.pop(self.context_param, None) if self.context_param else None
        local = (ctx.run_id, ctx.call_id) if ctx is not None and ctx.call_id else None
        task_id = self._task(local)
        if task_id is not None:
            result = await self._follow(task_id)
        else:
            result = await self.client.call_tool(self.remote, kwargs, meta=self._meta(ctx, local))
        self._remember(local, result)
        if result.is_error:
            raise ToolError(
                _text(result) or f"MCP tool {self.remote!r} failed",
                tool=self.local,
                details=_error_details(result.meta),
            )
        if result.structured_content is not None:
            return unwrap_structured(result.structured_content, self.output_schema)
        return _content_value(result)

    def _meta(
        self, ctx: ToolContext | None, local: tuple[str, str] | None
    ) -> RequestParamsMeta | None:
        with self._lock:
            meta = dict(self._resume.get(local, {})) if local is not None else {}
        if not meta and self.send_ids and ctx is not None and ctx.run_id:
            meta = {RUN_ID_META: ctx.run_id}
        return cast("RequestParamsMeta", meta) if meta else None

    def _task(self, local: tuple[str, str] | None) -> str | None:
        with self._lock:
            keys = self._resume.get(local, {}) if local is not None else {}
            return keys.get(TASK_ID_META)

    async def _follow(self, task_id: str) -> t.CallToolResult:
        from agent_adapter.integrations.mcp._tasks_client import follow, tasks_extension

        extension = tasks_extension(self.client)
        wait_s = extension.wait_s if extension is not None else DEFAULT_TASK_WAIT_S
        return await follow(self.client.session, task_id, wait_s)

    def _remember(self, local: tuple[str, str] | None, result: t.CallToolResult) -> None:
        if local is None:
            return
        meta = result.meta or {}
        keys = {k: meta[k] for k in (RUN_ID_META, CALL_ID_META) if isinstance(meta.get(k), str)}
        task = meta.get(TASK_ID_META)
        with self._lock:
            if result.is_error and isinstance(task, str) and task:
                self._resume[local] = {TASK_ID_META: task}
                if len(self._resume) > MAX_RESUMES:
                    self._resume.popitem(last=False)
            elif result.is_error and APPROVAL_ID_META in meta and len(keys) == 2:
                self._resume[local] = keys
                if len(self._resume) > MAX_RESUMES:
                    self._resume.popitem(last=False)
            else:
                self._resume.pop(local, None)


def _text(result: t.CallToolResult) -> str:
    from mcp.types import TextContent

    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _error_details(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    names = {
        APPROVAL_ID_META: "approval_id",
        ERROR_KIND_META: "remote_kind",
        TASK_ID_META: "task_id",
    }
    return {
        local: meta[key] for key, local in names.items() if meta and isinstance(meta.get(key), str)
    }


def _content_value(result: t.CallToolResult) -> Any:
    """Text blocks joined; any other content as JSON dicts; nothing -> None."""
    if not result.content:
        return None
    from mcp.types import TextContent

    if all(isinstance(block, TextContent) for block in result.content):
        return _text(result)
    return [
        block.model_dump(mode="json", by_alias=True, exclude_none=True) for block in result.content
    ]


def mcp_client(
    target: Any,
    *,
    headers: Mapping[str, str] | None = None,
    auth: httpx2.Auth | None = None,
    tasks: bool = False,
    task_wait_s: float = DEFAULT_TASK_WAIT_S,
) -> Client:
    """An unopened `mcp.Client` for an http(s) URL (Streamable HTTP), a `StdioServerParameters`,
    an in-process `Server` / `MCPServer`, or a client transport.

    Args:
        auth: An `httpx2.Auth` for a URL, e.g. `oauth_client(...)`; https only, except on a
            loopback host.
        tasks: Declare the 2026-07-28 tasks extension; a remote task is followed for up to
            `task_wait_s` seconds, then the call fails with its `task_id` and can resume it.

    Raises:
        ToolUsageError: a non-URL string, `headers` / `auth` for a non-URL target, an
            `Authorization` header with `auth`, or another type.
        ConfigError: `auth` over plain http to a non-loopback host; a bad `task_wait_s`.
        MissingDependency: `mcp` is not installed.
    """
    require_mcp()
    from mcp import Client, StdioServerParameters
    from mcp.server.lowlevel import Server
    from mcp.server.mcpserver import MCPServer

    options: dict[str, Any] = {}
    if tasks:
        from agent_adapter.integrations.mcp._tasks_client import tasks_extension_for

        options["extensions"] = [tasks_extension_for(task_wait_s)]
    if isinstance(target, str):
        if not target.startswith(("http://", "https://")):
            raise ToolUsageError(f"mcp_client needs an http(s) URL, got {target!r}")
        _check_auth(target, headers, auth)
        return Client(_HttpTransport(target, dict(headers or {}), auth), **options)
    if headers:
        raise ToolUsageError("mcp_client: headers= only applies to an http(s) URL")
    if auth is not None:
        raise ToolUsageError("mcp_client: auth= only applies to an http(s) URL")
    if isinstance(target, (Server, MCPServer, StdioServerParameters)) or (
        hasattr(target, "__aenter__") and hasattr(target, "__aexit__")
    ):
        return Client(target, **options)
    raise ToolUsageError(
        "mcp_client takes an http(s) URL, StdioServerParameters, an MCP server or a transport, "
        f"got {type(target).__name__}"
    )


def _check_auth(url: str, headers: Mapping[str, str] | None, auth: Any) -> None:
    if auth is None:
        return
    import httpx2

    if not isinstance(auth, httpx2.Auth):
        raise ToolUsageError(f"auth must be an httpx2.Auth, got {type(auth).__name__}")
    if any(name.lower() == "authorization" for name in headers or {}):
        raise ToolUsageError("pass either an Authorization header or auth=, not both")
    if url.startswith("http://") and urlsplit(url).hostname not in _LOOPBACK:
        raise ConfigError("auth= over plain http would expose tokens: use https")


def _new_http_client(headers: Mapping[str, str]) -> httpx2.AsyncClient:
    import httpx2

    return httpx2.AsyncClient(
        headers=dict(headers), timeout=httpx2.Timeout(HTTP_TIMEOUT_S, read=HTTP_READ_TIMEOUT_S)
    )


class _HttpTransport:
    """Streamable HTTP client transport owning an `httpx2.AsyncClient` with fixed headers."""

    def __init__(self, url: str, headers: Mapping[str, str], auth: Any = None) -> None:
        self.url = url
        self.headers = headers  # may hold credentials: never logged
        self.auth = auth
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> Any:
        from mcp.client.streamable_http import streamable_http_client

        stack = AsyncExitStack()
        try:
            http = await stack.enter_async_context(_new_http_client(self.headers))
            if self.auth is not None:
                http.auth = self.auth
            streams = await stack.enter_async_context(
                streamable_http_client(self.url, http_client=http)
            )
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        return streams

    async def __aexit__(self, *exc_info: Any) -> bool | None:
        stack, self._stack = self._stack, None
        if stack is None:
            return None
        return await stack.__aexit__(*exc_info)


__all__ = ["import_mcp", "mcp_client"]
