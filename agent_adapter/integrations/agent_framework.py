"""Microsoft Agent Framework: ToolSpecs as `FunctionTool`s, approval, a compression format.

MAF's approval responses are never taken as an approval. `resolve_approvals` asks the
approval port and issues a one-use token per approved call; without a matching token the
executor asks its own port when the call runs.
"""

from __future__ import annotations

import copy
import functools
import inspect
import json
from collections.abc import Mapping
from typing import Any, ClassVar

from agent_adapter.context.formats import register_format
from agent_adapter.context.formats._items import Call, Item, ItemsConverter, Result, Text, as_text
from agent_adapter.context.formats._shared import arguments_of
from agent_adapter.core.context import ToolContext
from agent_adapter.core.messages import Message
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_gate import (
    TokenTable,
    check_framework,
    decide_calls,
    run_bound,
    take_approval,
)
from agent_adapter.integrations.base import IntegrationBase, ToolRunner

TARGET = "agent_framework"
_SUPPORTED = ((1, 23), (2, 0))  # the `agent-framework` extra's range
_CALL_ID_KEY = "call_id"  # MAF's own FunctionInvocationContext.metadata key
_CONTEXT_PARAM = "_aa_invocation"  # the injected FunctionInvocationContext parameter
# One-use approvals `resolve_approvals` issues, bound to session, call id, tool and arguments.
_TOKENS = TokenTable()


class AgentFrameworkIntegration(IntegrationBase):
    """Exports `FunctionTool`s; MAF's function loop pauses `confirm` tools for approval."""

    name = TARGET
    requires = ("agent_framework",)
    extra = "agent-framework"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `FunctionTool` running `spec` through the executor."""
        return agent_framework_tool(spec, executor=executor)

    def bind_context(self, native: Any) -> ToolContext:
        """Ids from a `FunctionInvocationContext`: call id; the session as run and thread.

        Never an approval: MAF's approval flow only adds a gate in front of the executor's.
        """
        metadata: Mapping[str, Any] = getattr(native, "metadata", None) or {}
        call_id = metadata.get(_CALL_ID_KEY)
        session_id = _session_id((getattr(native, "kwargs", None) or {}).get("session"))
        return ToolContext(
            run_id=session_id,
            call_id=call_id if isinstance(call_id, str) else "",
            thread_id=session_id,
            extras={TARGET: native},
        )


INTEGRATION = AgentFrameworkIntegration()


def agent_framework_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """A `FunctionTool` for `spec`; any `confirm` rule sets `approval_mode="always_require"`.

    Raises:
        IntegrationError: the installed MAF is outside the extra's range or lacks what the
            bridge reads (`FunctionTool.invoke(context=, tool_call_id=)`, context `metadata`).
    """
    INTEGRATION.require()
    return _tool_class()(spec, lazy_runner(executor))


def agent_framework_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """`FunctionTool`s for every registry tool exported to this target."""
    return INTEGRATION.export_tools(registry, executor=executor)


async def resolve_approvals(
    response: Any, port: ApprovalPort, registry: Registry, *, session: Any = None
) -> list[Any]:
    """Answer each `function_approval_request` of `response` from `port`, in order.

    Send the returned `function_approval_response`s back with `agent.run(..., session=)`.
    Pass the run's `session`: its id is the run id the executor checks when the approved
    call runs. Registry tools are asked with validated arguments (approved unasked when
    their `confirm` rule says no); other tools with their raw arguments.

    Raises:
        ApprovalPending: a decision is still open (every request was asked first).
    """
    requests = [
        item
        for item in getattr(response, "user_input_requests", None) or ()
        if getattr(item, "type", "") == "function_approval_request"
    ]
    calls = [
        (str(c.name or ""), dict(c.parse_arguments() or {}), str(c.call_id or ""))
        for c in (item.function_call for item in requests)
    ]
    decisions = await decide_calls(
        port, calls, registry, run_id=_session_id(session), integration=TARGET, tokens=_TOKENS
    )
    return [
        item.to_function_approval_response(approved=decision.approved)
        for item, decision in zip(requests, decisions, strict=True)
    ]


def _session_id(session: Any) -> str:
    session_id = getattr(session, "session_id", None)
    return session_id if isinstance(session_id, str) else ""


@functools.cache
def _tool_class() -> type[Any]:
    """The `FunctionTool` subclass, built on first use (the framework is imported lazily)."""
    import agent_framework
    from agent_framework import FunctionInvocationContext, FunctionTool

    invoke = set(inspect.signature(FunctionTool.invoke).parameters)
    context = set(inspect.signature(FunctionInvocationContext.__init__).parameters)
    check_framework(
        TARGET,
        str(getattr(agent_framework, "__version__", "0")),
        _SUPPORTED,
        {
            "FunctionTool.invoke(context=, tool_call_id=)": {"context", "tool_call_id"} <= invoke,
            "FunctionInvocationContext.metadata": "metadata" in context,
        },
    )

    class AdapterFunctionTool(FunctionTool):  # type: ignore[misc, unused-ignore]
        """A `FunctionTool` that runs a `ToolSpec` through a `ToolRunner`."""

        DEFAULT_EXCLUDE: ClassVar[set[str]] = {*FunctionTool.DEFAULT_EXCLUDE, "spec", "runner"}

        def __init__(self, spec: ToolSpec, runner: ToolRunner) -> None:
            async def call(**arguments: Any) -> Any:
                ctx = INTEGRATION.bind_context(arguments.pop(_CONTEXT_PARAM, None))
                ctx.approval = take_approval(_TOKENS, spec, arguments, ctx)
                return await run_bound(runner, spec, arguments, ctx)

            # MAF injects its context into the parameter annotated with this class.
            call.__annotations__ = {_CONTEXT_PARAM: FunctionInvocationContext}
            call.__signature__ = _call_signature()  # type: ignore[attr-defined]
            super().__init__(
                name=spec.name,
                description=spec.description,
                func=call,
                input_model=spec.json_schema(),
                approval_mode="never_require" if spec.confirm is False else "always_require",
            )
            self.spec = spec
            self.runner = runner

        async def invoke(
            self,
            *,
            arguments: Any = None,
            context: Any = None,
            tool_call_id: str | None = None,
            **kwargs: Any,
        ) -> Any:
            """Record the call id on the context (MAF's direct path leaves it out)."""
            if context is not None and tool_call_id:
                context.metadata.setdefault(_CALL_ID_KEY, tool_call_id)
            return await super().invoke(
                arguments=arguments, context=context, tool_call_id=tool_call_id, **kwargs
            )

    return AdapterFunctionTool


def _call_signature() -> inspect.Signature:
    """`(_aa_invocation=None, **arguments)`: one injected context, then the tool's arguments."""
    return inspect.Signature(
        [
            inspect.Parameter(_CONTEXT_PARAM, inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("arguments", inspect.Parameter.VAR_KEYWORD),
        ]
    )


FORMAT = "agent_framework"
_ROLES = {"system": "system", "developer": "system", "user": "user", "assistant": "assistant"}
_ROLES["tool"] = "tool"
_APPROVALS = frozenset({"function_approval_request", "function_approval_response"})


def is_agent_framework_message(item: Any) -> bool:
    """True for an Agent Framework `Message`, found by class, without importing MAF."""
    return any(
        cls.__name__ == "Message" and cls.__module__.startswith("agent_framework.")
        for cls in type(item).__mro__
    )


class AgentFrameworkConverter(ItemsConverter):
    """`Message` lists: text, `function_call` and `function_result` contents.

    Messages holding an approval request or response are kept verbatim.
    """

    name = FORMAT

    def role_of(self, native: Any) -> str | None:
        if not is_agent_framework_message(native):
            return None
        return _ROLES.get(str(getattr(native.role, "value", native.role)))

    def items_of(self, native: Any) -> list[Any]:
        return list(native.contents)

    def read(self, item: Any) -> Item:
        kind = getattr(item, "type", None)
        if kind == "text":
            return Text(item.text or "")
        if kind == "function_call":
            raw = item.arguments
            arguments = arguments_of(dict(raw) if isinstance(raw, Mapping) else raw)
            name = str(item.name or "")
            return Call(ToolCall(id=str(item.call_id or ""), name=name, arguments=arguments))
        if kind == "function_result":
            return Result(str(item.call_id or ""), "", as_text(item.result))
        return None

    def keep_verbatim(self, native: Any) -> bool:
        return any(getattr(item, "type", None) in _APPROVALS for item in native.contents)

    def with_items(self, native: Any, items: list[Any]) -> Any:
        changed = copy.copy(native)
        changed.contents = items
        return changed

    def text_item(self, text: str, like: Any | None) -> Any:
        if like is None:
            return _content().from_text(text)
        changed = copy.copy(like)
        changed.text = text
        return changed

    def with_result(self, item: Any, text: str) -> Any:
        changed = copy.copy(item)
        changed.result = text
        others = [part for part in item.items or () if getattr(part, "type", None) != "text"]
        changed.items = [_content().from_text(text), *others]
        return changed

    def new(self, message: Message) -> Any:
        content = _content()
        contents: list[Any] = [content.from_text(message.content)] if message.content else []
        if message.role == "tool":
            contents = [content.from_function_result(message.tool_call_id, result=message.content)]
        contents += [
            content.from_function_call(c.id, c.name, arguments=json.dumps(c.arguments))
            for c in message.tool_calls
        ]
        return _message_class()(message.role, contents)


def _content() -> Any:
    """MAF's `Content` class, imported on first use.

    Raises:
        MissingDependency: `agent-framework-core` is not installed.
    """
    INTEGRATION.require()
    from agent_framework import Content

    return Content


def _message_class() -> Any:
    """MAF's `Message` class (see `_content`)."""
    INTEGRATION.require()
    from agent_framework import Message as MafMessage

    return MafMessage


register_format(FORMAT, AgentFrameworkConverter, detect=is_agent_framework_message)


__all__ = [
    "INTEGRATION",
    "AgentFrameworkIntegration",
    "agent_framework_tool",
    "agent_framework_tools",
    "resolve_approvals",
]
