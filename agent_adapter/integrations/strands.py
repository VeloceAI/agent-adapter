"""Strands Agents: each ToolSpec as an `AgentTool`; approval through a before-tool-call hook."""

from __future__ import annotations

import functools
import json
import logging
import weakref
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_gate import (
    TokenTable,
    dump_arguments,
    gate_decision,
    pending_message,
    take_approval,
)
from agent_adapter.integrations.base import (
    MODEL_VISIBLE_ERRORS,
    IntegrationBase,
    ToolRunner,
    model_visible_error,
)
from agent_adapter.integrations.strands_compress import (
    strands_compress_middleware,  # registers the format
)
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

TARGET = "strands"
# One run id per Strands agent (its conversation), shared by the gate and the tools.
_RUN_IDS: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()


class StrandsIntegration(IntegrationBase):
    """Exports `AgentTool`s; `strands_approval_hook` asks an approval port before a call."""

    name = TARGET
    requires = ("strands",)
    extra = "strands"
    call_id_fields = ("toolUseId",)

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `AgentTool` running `spec` through the executor."""
        return strands_tool(spec, executor=executor)

    def bind_approval(self, port: ApprovalPort) -> Any:
        """A `BeforeToolCallEvent` callback asking `port` about this module's tools."""
        return _approval_hook(port, None)


INTEGRATION = StrandsIntegration()


def strands_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """An `AgentTool` for `spec`; results are Strands tool results (`status` error or success)."""
    INTEGRATION.require()
    return _tool_class()(spec, lazy_runner(executor))


def strands_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """`AgentTool`s for every registry tool exported to this target."""
    return INTEGRATION.export_tools(registry, executor=executor)


def strands_approval_hook(port: ApprovalPort, registry: Registry) -> Any:
    """A `BeforeToolCallEvent` callback: asks `port` about `registry` tools whose rule says so.

    Register it with `agent.hooks.add_callback(BeforeToolCallEvent, hook)`. A denial cancels
    the call. An approval swaps in a one-call view of the tool holding a one-use token
    (this hook's, bound to the agent's run, the `toolUseId`, the tool and its input), so the
    executor does not ask again. Edits are not supported: an edit is a denial.
    """
    return _approval_hook(port, registry)


def _run_id(agent: Any) -> str:
    """The run id of a Strands agent (created on first use); "" when it cannot be tracked."""
    if agent is None:
        return ""
    try:
        run_id = _RUN_IDS.get(agent)
        if run_id is None:
            run_id = _RUN_IDS[agent] = new_id("run_")
    except TypeError:  # not weak-referenceable or not hashable
        return ""
    return run_id


def _approval_hook(port: ApprovalPort, registry: Registry | None) -> Any:
    """The async callback (Strands awaits only coroutine functions) behind both factories."""
    tokens = TokenTable()

    async def before_tool_call(event: Any) -> None:
        tool_use: Mapping[str, Any] = event.tool_use
        selected = event.selected_tool
        spec = _spec(registry, str(tool_use.get("name", "")), selected)
        if spec is None:
            return
        call_id = str(tool_use.get("toolUseId") or "")
        arguments = tool_use.get("input")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        run_id = _run_id(getattr(event, "agent", None))
        try:
            decision = await gate_decision(
                spec, arguments, port=port, integration=TARGET, call_id=call_id, run_id=run_id
            )
        except ApprovalPending as exc:
            logger.info("strands: %s (call %s) waits for approval", spec.name, call_id)
            event.cancel_tool = pending_message(exc)
            return
        if decision is None:
            return
        if not decision.approved:
            denied = ApprovalDenied(spec.name, call_id=call_id, reason=decision.reason)
            event.cancel_tool = json.dumps(model_visible_error(denied))
            return
        dumped = dump_arguments(spec, arguments) or {}
        if isinstance(selected, _tool_class()) and selected.spec.name == spec.name:
            tokens.issue(spec.name, dumped, decision, call_id=call_id, run_id=run_id)
            event.selected_tool = selected.approved_view(tokens)

    return before_tool_call


def _spec(registry: Registry | None, name: str, selected: Any) -> ToolSpec | None:
    """The registry's spec for `name`; without a registry, the spec of one of our tools."""
    if registry is not None:
        return registry.get(name) if name in registry else None
    spec = getattr(selected, "spec", None) if isinstance(selected, _tool_class()) else None
    return spec if isinstance(spec, ToolSpec) else None


@functools.cache
def _tool_class() -> type[Any]:
    """The `AgentTool` subclass, built on first use (the framework is imported lazily)."""
    from strands.types.tools import AgentTool
    from strands.types.tools import ToolSpec as StrandsToolSpec

    class AdapterAgentTool(AgentTool):  # type: ignore[misc, unused-ignore]
        """An `AgentTool` that runs a `ToolSpec` through a `ToolRunner`."""

        def __init__(
            self, spec: ToolSpec, runner: ToolRunner, tokens: TokenTable | None = None
        ) -> None:
            super().__init__()
            self.spec = spec
            self.runner = runner
            self.tokens = tokens  # set only on the one-call view an approval swaps in

        def approved_view(self, tokens: TokenTable) -> Any:
            """This tool for one approved call: it may take its token from `tokens`."""
            return type(self)(self.spec, self.runner, tokens)

        @property
        def tool_name(self) -> str:
            return self.spec.name

        @property
        def tool_spec(self) -> StrandsToolSpec:
            spec = self.spec
            return {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": {"json": spec.json_schema()},
            }

        @property
        def tool_type(self) -> str:
            return "python"

        async def stream(
            self, tool_use: Any, invocation_state: dict[str, Any], **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            """Run the call; one result event (control flow propagates)."""
            yield await _run(self, tool_use, invocation_state)

    return AdapterAgentTool


async def _run(tool: Any, tool_use: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    spec: ToolSpec = tool.spec
    arguments = tool_use.get("input")
    arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
    native = INTEGRATION.bind_context(tool_use)
    ctx = ToolContext(
        run_id=_run_id(state.get("agent")), call_id=native.call_id, extras=native.extras
    )
    if tool.tokens is not None:
        ctx.approval = take_approval(tool.tokens, spec, arguments, ctx)
    use_id = str(tool_use.get("toolUseId") or "")
    try:
        value = await tool.runner.run_spec(spec, arguments, ctx, call_id=ctx.call_id or None)
    except MODEL_VISIBLE_ERRORS as exc:
        return {
            "toolUseId": use_id,
            "status": "error",
            "content": [_block(model_visible_error(exc))],
        }
    return {"toolUseId": use_id, "status": "success", "content": [_block(value)]}


def _block(value: Any) -> dict[str, Any]:
    """A Strands content block: text for strings, JSON for JSON values, else `str()` text."""
    if isinstance(value, str):
        return {"text": value}
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return {"text": str(value)}
    return {"json": value}


__all__ = [
    "INTEGRATION",
    "StrandsIntegration",
    "strands_approval_hook",
    "strands_compress_middleware",
    "strands_tool",
    "strands_tools",
]
