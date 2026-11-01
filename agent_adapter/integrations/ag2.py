"""AG2 (the `ag2` package): each ToolSpec as a `FunctionTool`; approval as tool middleware.

AG2's own `approval_required` answer ("y", "always") is never taken as an approval. The
gate `approval=port` installs asks that port and, on approval, issues a one-use token
bound to the conversation stream, the tool call id, the tool and its validated
arguments; without a matching token the executor asks its own port.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending, ToolArgsInvalid
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_gate import (
    TokenTable,
    check_framework,
    dump_arguments,
    gate_decision,
    pending_message,
    take_approval,
)
from agent_adapter.integrations.ag2_compress import ag2_compress_policy  # registers the format
from agent_adapter.integrations.base import IntegrationBase, ToolRunner, model_visible_error

logger = logging.getLogger(__name__)

TARGET = "ag2"
_SUPPORTED = ((1, 2, 2), (2, 0))  # the `ag2` extra's range
_SOURCE_PREFIX = "agent_adapter"
_NOT_AN_OBJECT = "arguments are not a JSON object"


class AG2Integration(IntegrationBase):
    """Exports `FunctionTool`s; `bind_approval` gives no gate (use `ag2_tool(approval=)`)."""

    name = TARGET
    requires = ("ag2",)
    extra = "ag2"
    call_id_fields = ("id",)

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `FunctionTool` running `spec` through the executor."""
        return ag2_tool(spec, executor=executor)

    def bind_context(self, native: Any) -> ToolContext:
        """Ids from a `ToolCallEvent` (`id`); the stream id as run and thread when given.

        `native` is the event, or an `(event, context)` pair from inside a run.
        """
        event, context = native if isinstance(native, tuple) else (native, None)
        base = super().bind_context(event)
        run_id = _run_id(context)
        return ToolContext(
            run_id=run_id, call_id=base.call_id, thread_id=run_id, extras={TARGET: event}
        )


INTEGRATION = AG2Integration()


def ag2_tool(
    spec: ToolSpec,
    *,
    executor: ToolRunner | None = None,
    approval: ApprovalPort | None = None,
    middleware: Iterable[Any] = (),
) -> Any:
    """An AG2 `FunctionTool` for `spec`.

    Args:
        spec: The tool.
        executor: The runner (default: the default executor at call time).
        approval: A port asked before a call whose `confirm` rule says so (innermost
            tool middleware, so it sees the arguments that run). Optional: without it
            the executor's own port decides.
        middleware: Extra AG2 tool middleware, outermost first.

    Raises:
        MissingDependency: AG2 is not installed.
        IntegrationError: the installed AG2 is outside the extra's range or lacks what
            the bridge reads (`ConversationContext.tool_call_id`, `ToolCallEvent.handled_by`).
    """
    INTEGRATION.require()
    tokens = TokenTable() if approval is not None else None
    hooks = [*middleware]
    if approval is not None and tokens is not None:
        hooks.append(_gate(spec, approval, tokens))
    return _tool_class()(spec, lazy_runner(executor), tokens, hooks)


def ag2_tools(
    registry: Registry,
    *,
    executor: ToolRunner | None = None,
    approval: ApprovalPort | None = None,
    middleware: Iterable[Any] = (),
) -> list[Any]:
    """AG2 `FunctionTool`s for every registry tool exported to this target."""
    INTEGRATION.require()
    hooks = tuple(middleware)
    return [
        ag2_tool(spec, executor=executor, approval=approval, middleware=hooks)
        for spec in registry.for_target(TARGET)
    ]


def _run_id(context: Any) -> str:
    """The conversation's stream id: one per `Agent.ask` stream; "" without a context."""
    stream_id = getattr(getattr(context, "stream", None), "id", None)
    return str(stream_id) if stream_id is not None else ""


def _arguments(event: Any) -> dict[str, Any] | None:
    """The call's decoded arguments; None when they are not a JSON object."""
    try:
        arguments = event.serialized_arguments
    except ValueError:
        return None
    return dict(arguments) if isinstance(arguments, Mapping) else None


def _gate(spec: ToolSpec, port: ApprovalPort, tokens: TokenTable) -> Any:
    """Tool middleware: ask `port`, then let the call through holding a one-use token."""
    from ag2.events import ToolResultEvent

    async def approval_gate(call_next: Any, event: Any, context: Any) -> Any:
        arguments = _arguments(event)
        if arguments is None:
            return await call_next(event, context)  # the tool reports the bad arguments
        call_id, run_id = str(event.id or ""), _run_id(context)
        try:
            decision = await gate_decision(
                spec, arguments, port=port, integration=TARGET, call_id=call_id, run_id=run_id
            )
        except ApprovalPending as exc:
            logger.info("ag2: %s (call %s) waits for approval", spec.name, call_id)
            return ToolResultEvent.from_call(event, result=pending_message(exc))
        if decision is None:
            return await call_next(event, context)
        if not decision.approved:
            denied = ApprovalDenied(spec.name, call_id=call_id, reason=decision.reason)
            return ToolResultEvent.from_call(event, result=json.dumps(model_visible_error(denied)))
        dumped = dump_arguments(spec, arguments) or {}
        tokens.issue(spec.name, dumped, decision, call_id=call_id, run_id=run_id)
        try:
            return await call_next(event, context)
        finally:  # a token the call did not take (its arguments changed) never outlives it
            tokens.take(spec.name, dumped, call_id=call_id, run_id=run_id)

    return approval_gate


@functools.cache
def _tool_class() -> type[Any]:
    """The `FunctionTool` subclass, built on first use (the framework is imported lazily)."""
    import ag2
    from ag2.context import ConversationContext
    from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResult, ToolResultEvent
    from ag2.tools.types import FunctionDefinition, FunctionTool, FunctionToolSchema

    check_framework(
        TARGET,
        str(getattr(ag2, "__version__", "0")),
        _SUPPORTED,
        {
            "ConversationContext.tool_call_id": "tool_call_id"
            in getattr(ConversationContext, "__dataclass_fields__", {}),
            "ToolCallEvent.handled_by": callable(getattr(ToolCallEvent, "handled_by", None)),
        },
    )

    class AdapterFunctionTool(FunctionTool):  # type: ignore[misc, unused-ignore]
        """A `FunctionTool` that runs a `ToolSpec` through a `ToolRunner`."""

        def __init__(
            self,
            spec: ToolSpec,
            runner: ToolRunner,
            tokens: TokenTable | None,
            middleware: Iterable[Any],
        ) -> None:
            # Not FunctionTool.__init__: it needs a fast_depends model of a Python function.
            self.model = None  # type: ignore[assignment, unused-ignore]  # `source` is overridden
            self.timeout = None
            self._middleware = tuple(middleware)
            self.name = spec.name
            self.schema = FunctionToolSchema(
                function=FunctionDefinition(
                    name=spec.name, description=spec.description, parameters=spec.json_schema()
                )
            )
            self.spec = spec
            self.runner = runner
            self.tokens = tokens  # shared with this tool's approval gate

        @property
        def source(self) -> str:
            """The implementation AG2 attributes calls to (its "always" grants key on it)."""
            return f"{_SOURCE_PREFIX}:{self.spec.name}:{id(self.spec):x}"

        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            """AG2 deep-copies a tool on registration; share the spec, runner and tokens."""
            clone = copy.copy(self)
            clone._middleware = tuple(self._middleware)
            return clone

        async def _execute(self, event: Any, context: Any) -> Any:
            return await _run(self, event, context, ToolResultEvent, ToolErrorEvent, ToolResult)

    return AdapterFunctionTool


async def _run(
    tool: Any, event: Any, context: Any, result_event: Any, error_event: Any, result: Any
) -> Any:
    """Run one call; a failure becomes a `ToolErrorEvent` showing `model_visible_error` text.

    AG2 would show the model the full traceback; the event still carries the exception.
    Control flow (`asyncio.CancelledError`) propagates.
    """
    spec: ToolSpec = tool.spec
    ctx = INTEGRATION.bind_context((event, context))
    arguments = _arguments(event)
    try:
        if arguments is None:
            raise ToolArgsInvalid(spec.name, [{"type": "dict_type", "msg": _NOT_AN_OBJECT}])
        if tool.tokens is not None:
            ctx.approval = take_approval(tool.tokens, spec, arguments, ctx)
        value = await tool.runner.run_spec(spec, arguments, ctx, call_id=ctx.call_id or None)
    except Exception as exc:
        return error_event(
            parent_id=event.id,
            name=event.name,
            error=exc,
            result=result(json.dumps(model_visible_error(exc))),
        )
    return result_event.from_call(event, result=value)


__all__ = [
    "INTEGRATION",
    "AG2Integration",
    "ag2_compress_policy",
    "ag2_tool",
    "ag2_tools",
]
