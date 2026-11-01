"""Claude Agent SDK: tools on an in-process SDK MCP server; approval through `can_use_tool`.

`claude_sdk_bridge` pairs a server with its own `can_use_tool` gate: an approval becomes a
one-use token (60 s, dropped at the end of the turn) of that bridge only, bound to the
`tool_use_id`, `mcp__<server>__<tool>` and the final input. The bridge's own MCP server
reads the `tool_use_id` Claude Code sends with each `tools/call` and takes only that call's
token. Nothing else approves a `confirm` tool: without a token (`allowed_tools`, permission
modes, other gates) the executor's own port decides. One bridge is one run.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_adapter._version import __version__ as _adapter_version
from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_gate import (
    TokenTable,
    approval_request,
    check_framework,
    dump_arguments,
    gate_decision,
    json_text,
    pending_message,
)
from agent_adapter.integrations.base import (
    MODEL_VISIBLE_ERRORS,
    IntegrationBase,
    ToolRunner,
    model_visible_error,
)
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

TARGET = "claude_agent"
_DEFAULT_SERVER = "agent_adapter"
_MCP_PREFIX = "mcp__"
_TOOL_USE_ID_META = "claudecode/toolUseId"  # Claude Code's tool-use id in tools/call `_meta`
_SUPPORTED_SDK = ((0, 2, 150), (0, 3, 0))  # the `claude-agent` extra's range
# Major-granular: mcp has no `__version__` and R8 rules out importlib.metadata here, so the
# major is probed; the `Server(on_call_tool=)` probe covers the API the bridge uses.
_SUPPORTED_MCP = ((2, 0), (3, 0))

CanUseTool = Callable[[str, dict[str, Any], Any], Awaitable[Any]]


@dataclass(frozen=True)
class ClaudeSdkBridge:
    """An SDK MCP server, the `can_use_tool` gate its tools trust, and the gate's hooks."""

    name: str  # the `mcp_servers` key; Claude sees `mcp__<name>__<tool>`
    server: dict[str, Any]  # McpSdkServerConfig
    can_use_tool: CanUseTool
    run_id: str  # one run: every request and call of this bridge, and its run guards
    hooks: dict[str, list[Any]]  # `ClaudeAgentOptions(hooks=…)`: drop tokens between turns


@dataclass(frozen=True)
class _Gate:
    """What a bridge's handlers share with its gate."""

    server: str
    tokens: TokenTable | None  # None: a permission-only gate (no tokens)
    run_id: str


class ClaudeAgentIntegration(IntegrationBase):
    """Exports SDK MCP tools; a bridge's `can_use_tool` asks an approval port for Claude."""

    name = TARGET
    requires = ("claude_agent_sdk",)
    extra = "claude-agent"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `SdkMcpTool` running `spec` through the executor (no gate of its own)."""
        return claude_sdk_tool(spec, executor=executor)

    def bind_approval(self, port: ApprovalPort) -> Any:
        """A `can_use_tool` callback asking `port` about every tool; it grants no tokens."""
        return _permission(port, None, None)


INTEGRATION = ClaudeAgentIntegration()


def claude_sdk_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """An `SdkMcpTool` for `spec`; a `confirm` tool is approved by the executor's own port."""
    INTEGRATION.require()
    return _tool(spec, lazy_runner(executor))


def claude_sdk_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """`SdkMcpTool`s for every registry tool exported to this target."""
    return INTEGRATION.export_tools(registry, executor=executor)


def claude_sdk_server(
    registry: Registry, *, executor: ToolRunner | None = None, name: str = _DEFAULT_SERVER
) -> dict[str, Any]:
    """An SDK MCP server config without a gate (see `claude_sdk_bridge` for approvals)."""
    INTEGRATION.require()
    from claude_agent_sdk import create_sdk_mcp_server

    tools = claude_sdk_tools(registry, executor=executor)
    return dict(create_sdk_mcp_server(name, _adapter_version, tools))


def claude_sdk_bridge(
    registry: Registry,
    port: ApprovalPort,
    *,
    executor: ToolRunner | None = None,
    name: str = _DEFAULT_SERVER,
) -> ClaudeSdkBridge:
    """A server named `name` with its gate, for one client session: `ClaudeAgentOptions(
    mcp_servers={bridge.name: bridge.server}, can_use_tool=bridge.can_use_tool,
    hooks=bridge.hooks)`.

    The gate asks `port` about this server's registry tools when their `confirm` rule says
    so (edits become `updated_input`), and about every other tool under its Claude name.
    Its `Stop` and `UserPromptSubmit` hooks drop unused tokens, so an approval never
    outlives its turn. Everything using one bridge shares its run and its tokens.

    Raises:
        IntegrationError: the SDK is outside the extra's range, or `mcp` is older than 2.
    """
    INTEGRATION.require()
    _check_frameworks()
    from claude_agent_sdk import HookMatcher

    tokens = TokenTable()
    gate = _Gate(name, tokens, new_id("run_"))
    specs = {spec.name: spec for spec in registry.for_target(TARGET)}
    server = _bridge_server(name, specs, lazy_runner(executor), gate)

    async def drop_tokens(event: Any, tool_use_id: str | None, context: Any) -> Any:
        tokens.clear()
        return {}

    hooks = {event: [HookMatcher(hooks=[drop_tokens])] for event in ("UserPromptSubmit", "Stop")}
    permission = _permission(port, registry, gate)
    return ClaudeSdkBridge(name, server, permission, gate.run_id, hooks)


def can_use_tool(
    port: ApprovalPort, registry: Registry, *, server: str = _DEFAULT_SERVER
) -> CanUseTool:
    """A permission-only `can_use_tool`: it grants no tokens, so a `confirm` tool is asked
    again by the executor's port when it runs (use `claude_sdk_bridge` to ask once)."""
    return _permission(port, registry, _Gate(server, None, new_id("run_")))


def _registry_name(tool_name: str, registry: Registry, server: str) -> str:
    """The registry tool behind `mcp__<server>__<tool>` for exactly `server`, or ""."""
    rest = tool_name.removeprefix(f"{_MCP_PREFIX}{server}__")
    return rest if rest != tool_name and rest in registry else ""


def _tool(spec: ToolSpec, runner: ToolRunner) -> Any:
    """A gate-less `SdkMcpTool` (the SDK's own server gives handlers no tool-use id)."""
    from claude_agent_sdk import tool

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        return await _run(spec, runner, arguments, ToolContext(extras={TARGET: arguments}))

    return tool(spec.name, spec.description, spec.json_schema())(handler)


async def _run(
    spec: ToolSpec, runner: ToolRunner, arguments: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    """One call: MCP-style text content, `is_error` for every failure (as the SDK reports)."""
    try:
        value = await runner.run_spec(spec, arguments, ctx, call_id=ctx.call_id or None)
    except MODEL_VISIBLE_ERRORS as exc:
        return {"content": [_text(model_visible_error(exc))], "is_error": True}
    except Exception as exc:
        if is_control_flow(exc):
            raise
        logger.warning(
            "claude_agent: %s (call %s) failed: %s", spec.name, ctx.call_id, type(exc).__name__
        )
        return {"content": [_text(model_visible_error(exc))], "is_error": True}
    return {"content": [_text(value)]}


def _bridge_server(
    name: str, specs: Mapping[str, ToolSpec], runner: ToolRunner, gate: _Gate
) -> dict[str, Any]:
    """The bridge's own MCP server: like the SDK's, but it reads the call's tool-use id."""
    from mcp.server.lowlevel import Server
    from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

    tools = [
        Tool(name=spec.name, description=spec.description, inputSchema=spec.json_schema())
        for spec in specs.values()
    ]

    async def on_list_tools(ctx: Any, params: Any) -> Any:
        return ListToolsResult(tools=tools)

    async def on_call_tool(ctx: Any, params: Any) -> Any:
        spec = specs.get(params.name)
        if spec is None:
            text = f"Tool {params.name!r} not found"
            return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)
        use_id = (params.meta or {}).get(_TOOL_USE_ID_META)
        arguments = dict(params.arguments or {})
        ctx_ = _context(spec, arguments, gate, use_id if isinstance(use_id, str) else "")
        result = await _run(spec, runner, arguments, ctx_)
        content = [TextContent(type="text", text=block["text"]) for block in result["content"]]
        return CallToolResult(content=content, isError=bool(result.get("is_error")))

    server = Server(
        name, version=_adapter_version, on_list_tools=on_list_tools, on_call_tool=on_call_tool
    )
    return {"type": "sdk", "name": name, "instance": server}


def _context(
    spec: ToolSpec, arguments: dict[str, Any], gate: _Gate, tool_use_id: str
) -> ToolContext:
    """The call's context: the bridge's run, its tool-use id, and the token of exactly it."""
    ctx = ToolContext(run_id=gate.run_id, call_id=tool_use_id, extras={TARGET: arguments})
    dumped = dump_arguments(spec, arguments)
    if gate.tokens is None or not tool_use_id or dumped is None:
        return ctx  # no id, no token: the executor asks its own port
    qualified = f"{_MCP_PREFIX}{gate.server}__{spec.name}"
    token = gate.tokens.take(qualified, dumped, call_id=tool_use_id, run_id=gate.run_id)
    ctx.approval = token.decision if token is not None else None
    return ctx


@functools.cache
def _check_frameworks() -> None:
    """Refuse an SDK outside the extra's range, or an `mcp` older than 2 (other server API)."""
    import claude_agent_sdk
    import mcp.types
    from mcp.server.lowlevel import Server

    check_framework(
        TARGET,
        str(getattr(claude_agent_sdk, "__version__", "0")),
        _SUPPORTED_SDK,
        {"ToolPermissionContext.tool_use_id": _has_tool_use_id(claude_agent_sdk)},
    )
    major = 2 if hasattr(mcp.types, "jsonrpc_message_adapter") else 1
    check_framework(
        "mcp",
        f"{major}.0",
        _SUPPORTED_MCP,
        {"Server(on_call_tool=)": "on_call_tool" in inspect.signature(Server).parameters},
    )


def _has_tool_use_id(sdk: Any) -> bool:
    fields = getattr(sdk.ToolPermissionContext, "__dataclass_fields__", {})
    return "tool_use_id" in fields


def _permission(port: ApprovalPort, registry: Registry | None, gate: _Gate | None) -> CanUseTool:
    """The permission callback behind the bridge, `can_use_tool` and `bind_approval`."""

    async def permission(tool_name: str, tool_input: dict[str, Any], context: Any) -> Any:
        call_id = str(getattr(context, "tool_use_id", None) or "")
        if registry is not None and gate is not None:
            name = _registry_name(tool_name, registry, gate.server)
            if name:
                spec = registry.get(name)
                return await _ask_for_spec(port, spec, tool_name, tool_input, call_id, gate)
        run_id = gate.run_id if gate is not None else ""
        return await _ask_for_other(port, tool_name, tool_input, call_id, run_id)

    return permission


async def _ask_for_spec(
    port: ApprovalPort,
    spec: ToolSpec,
    qualified: str,
    tool_input: dict[str, Any],
    call_id: str,
    gate: _Gate,
) -> Any:
    """One of this server's tools: its confirm rule decides; an approval issues a token."""
    from claude_agent_sdk import PermissionResultAllow

    try:
        decision = await gate_decision(
            spec,
            tool_input,
            port=port,
            integration=TARGET,
            call_id=call_id,
            run_id=gate.run_id,
            edits=True,
        )
    except ApprovalPending as exc:
        return _deny_pending(spec.name, call_id, exc)
    if decision is None:
        return PermissionResultAllow()
    if not decision.approved:
        return _deny(spec.name, call_id, decision)
    edited = decision.edited_args
    final = dict(edited) if edited is not None else dict(tool_input)
    dumped = dump_arguments(spec, final)
    if gate.tokens is not None and dumped is not None:
        granted = decision.model_copy(update={"edited_args": None})  # already the final input
        gate.tokens.issue(qualified, dumped, granted, call_id=call_id, run_id=gate.run_id)
    return PermissionResultAllow(updated_input=final if edited is not None else None)


async def _ask_for_other(
    port: ApprovalPort, tool_name: str, tool_input: dict[str, Any], call_id: str, run_id: str
) -> Any:
    """Any other tool: asked under its Claude name; edits pass through; never a token."""
    from claude_agent_sdk import PermissionResultAllow

    request = approval_request(tool_name, tool_input, run_id=run_id, call_id=call_id)
    try:
        decision = await port.request(request)
    except ApprovalPending as exc:
        return _deny_pending(tool_name, call_id, exc)
    if not decision.approved:
        return _deny(tool_name, call_id, decision)
    edited = decision.edited_args
    return PermissionResultAllow(updated_input=dict(edited) if edited is not None else None)


def _deny(name: str, call_id: str, decision: ApprovalDecision) -> Any:
    from claude_agent_sdk import PermissionResultDeny

    denied = ApprovalDenied(name, call_id=call_id, reason=decision.reason)
    return PermissionResultDeny(message=json.dumps(model_visible_error(denied)))


def _deny_pending(name: str, call_id: str, exc: ApprovalPending) -> Any:
    from claude_agent_sdk import PermissionResultDeny

    logger.info("claude_agent: %s (call %s) waits for approval", name, call_id)
    return PermissionResultDeny(message=pending_message(exc))


def _text(value: Any) -> dict[str, Any]:
    text = json_text(value)
    return {"type": "text", "text": text if isinstance(text, str) else str(text)}


__all__ = [
    "INTEGRATION",
    "ClaudeAgentIntegration",
    "ClaudeSdkBridge",
    "can_use_tool",
    "claude_sdk_bridge",
    "claude_sdk_server",
    "claude_sdk_tool",
    "claude_sdk_tools",
]
