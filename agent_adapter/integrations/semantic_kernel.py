"""Semantic Kernel: ToolSpecs as kernel functions, an approval filter, a compression format."""

from __future__ import annotations

import contextvars
import copy
import functools
import json
import logging
import weakref
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_adapter.context.formats import register_format
from agent_adapter.context.formats._items import Call, Item, ItemsConverter, Result, Text, as_text
from agent_adapter.context.formats._shared import arguments_of
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending, ToolDefinitionError
from agent_adapter.core.messages import Message
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_gate import (
    TokenTable,
    dump_arguments,
    gate_decision,
    json_text,
    pending_message,
    run_bound,
)
from agent_adapter.integrations.base import IntegrationBase, ToolRunner, model_visible_error
from agent_adapter.runtime.ids import new_id
from agent_adapter.schema.transforms import inline_refs

logger = logging.getLogger(__name__)

TARGET = "semantic_kernel"
DEFAULT_PLUGIN = "agent_adapter"
_SPEC_ATTR = "__aa_spec__"  # the ToolSpec behind an exported function's method
# Names Semantic Kernel injects into a method instead of taking them from the arguments.
_RESERVED = frozenset({"kernel", "service", "execution_settings", "arguments"})


@dataclass(frozen=True)
class _Gate:
    """What the filter hands one invocation of our function: its run and its tokens."""

    run_id: str
    tokens: TokenTable


# Set by the filter around `next(context)` (one invocation), and by our function's `invoke`
# (its call id and qualified `<plugin>-<function>` name).
_GATE: contextvars.ContextVar[_Gate | None] = contextvars.ContextVar("aa_sk_gate", default=None)
_CALL: contextvars.ContextVar[tuple[str, str]] = contextvars.ContextVar(
    "aa_sk_call", default=("", "")
)
# One run id per conversation (chat history), shared by the filter's requests and the calls.
_RUN_IDS: dict[int, tuple[weakref.ref[Any], str]] = {}

Filter = Callable[[Any, Callable[[Any], Awaitable[None]]], Awaitable[None]]


class SemanticKernelIntegration(IntegrationBase):
    """Exports kernel functions; `approval_filter` asks an approval port before auto-calls."""

    name = TARGET
    requires = ("semantic_kernel",)
    extra = "semantic-kernel"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One kernel function running `spec` through the executor."""
        return kernel_function(spec, executor=executor)

    def bind_context(self, native: Any) -> ToolContext:
        """Context for one call; `native` is `(spec, arguments)`.

        Call id and qualified name come from the function call being invoked; run id and
        the approval token from `approval_filter`, for exactly this call and its arguments.
        """
        spec, arguments = native
        call_id, qualified = _CALL.get()
        gate = _GATE.get()
        ctx = ToolContext(
            run_id=gate.run_id if gate is not None else "",
            call_id=call_id,
            extras={TARGET: arguments},
        )
        dumped = dump_arguments(spec, arguments)
        if gate is not None and call_id and dumped is not None:
            token = gate.tokens.take(qualified, dumped, call_id=call_id, run_id=gate.run_id)
            ctx.approval = token.decision if token is not None else None
        return ctx

    def bind_approval(self, port: ApprovalPort) -> Any:
        """An auto-function-invocation filter asking `port` about this module's functions."""
        return _filter(port, None, None)


INTEGRATION = SemanticKernelIntegration()


def kernel_function(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """A `KernelFunctionFromMethod` for `spec`, its parameters taken from the JSON schema.

    Results reach the kernel as JSON text for non-string values (the model sees `str()`).

    Raises:
        ToolDefinitionError: a parameter is named `kernel`, `service`, `execution_settings`
            or `arguments` (Semantic Kernel injects those).
    """
    INTEGRATION.require()
    schema = inline_refs(spec.json_schema())
    properties: Mapping[str, Any] = schema.get("properties", {})
    clash = sorted(_RESERVED & set(properties))
    if clash:
        raise ToolDefinitionError(
            f"tool {spec.name!r}: Semantic Kernel reserves the argument name(s) {clash}"
        )
    from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata

    required = set(schema.get("required", []))
    parameters = [
        KernelParameterMetadata(
            name=name,
            description=prop.get("description"),
            type_=prop["type"] if isinstance(prop.get("type"), str) else "object",
            is_required=name in required,
            schema_data=copy.deepcopy(dict(prop)),
        )
        for name, prop in properties.items()
    ]
    method = _method(spec, lazy_runner(executor))
    return _function_class()(method=method, parameters=parameters)


def kernel_plugin(
    registry: Registry, *, name: str = DEFAULT_PLUGIN, executor: ToolRunner | None = None
) -> Any:
    """A `KernelPlugin` of the registry's tools for this target; the model sees `<name>-<tool>`."""
    INTEGRATION.require()
    from semantic_kernel.functions import KernelPlugin

    return KernelPlugin(name=name, functions=INTEGRATION.export_tools(registry, executor=executor))


def approval_filter(
    port: ApprovalPort, registry: Registry, *, plugin: str | None = DEFAULT_PLUGIN
) -> Filter:
    """An `AUTO_FUNCTION_INVOCATION` filter backed by `port`.

    Register it with `kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, f)`.
    `<plugin>-<function>` maps back to the registry name, for `plugin` only (the name given
    to `kernel_plugin`; `None` accepts any plugin), and the tool's `confirm` rule decides
    whether to ask. A denial (or a pending approval) becomes the function result without
    calling it. An approval is a one-use token for that invocation (run, call id, qualified
    name, arguments), so the executor does not ask again. Edits are not supported: an edit
    is a denial.
    """
    return _filter(port, registry, plugin)


def _registry_name(qualified: str, registry: Registry, plugin: str | None) -> str:
    """The registry tool behind `<plugin>-<function>` (or a bare function name), or ""."""
    plugin_name, dash, function = qualified.partition("-")
    if not dash:
        plugin_name, function = "", qualified
    if plugin is not None and plugin_name != plugin:
        return ""
    return function if function in registry else ""


def _method(spec: ToolSpec, runner: ToolRunner) -> Callable[..., Awaitable[Any]]:
    """The async method behind the kernel function, marked the way `@kernel_function` does."""

    async def method(**arguments: Any) -> Any:
        ctx = INTEGRATION.bind_context((spec, arguments))
        return json_text(await run_bound(runner, spec, arguments, ctx))

    marks = {
        "__kernel_function__": True,
        "__kernel_function_name__": spec.name,
        "__kernel_function_description__": spec.description,
        "__kernel_function_parameters__": [],
        "__kernel_function_return_description__": "",
        "__kernel_function_return_type__": "str",
        "__kernel_function_return_type_object__": str,
        "__kernel_function_return_required__": True,
        _SPEC_ATTR: spec,
    }
    for attribute, value in marks.items():
        setattr(method, attribute, value)
    return method


@functools.cache
def _function_class() -> type[Any]:
    """The `KernelFunctionFromMethod` subclass, built on first use (lazy import)."""
    from semantic_kernel.functions import KernelFunctionFromMethod

    class AdapterKernelFunction(KernelFunctionFromMethod):  # type: ignore[misc, unused-ignore]
        """A kernel function that hands its function call id to the wrapped `ToolSpec`."""

        async def invoke(
            self,
            kernel: Any,
            arguments: Any = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            call_id = (metadata or {}).get("id")
            call = (call_id if isinstance(call_id, str) else "", str(self.fully_qualified_name))
            token = _CALL.set(call)
            try:
                return await super().invoke(kernel, arguments, metadata, **kwargs)
            finally:
                _CALL.reset(token)

    return AdapterKernelFunction


def _filter(port: ApprovalPort, registry: Registry | None, plugin: str | None) -> Filter:
    """The filter behind `approval_filter` and `bind_approval`."""
    fallback_run = new_id("run_")  # for invocations without a chat history

    # Semantic Kernel passes the rest of the chain as the keyword `next`.
    async def approval(context: Any, next: Callable[[Any], Awaitable[None]]) -> None:
        function = context.function
        spec = _spec(registry, plugin, function)
        if spec is None:
            await next(context)
            return
        call_id = str(getattr(context.function_call_content, "id", None) or "")
        run_id = _run_id(context.chat_history, fallback_run)
        arguments = {
            p.name: context.arguments[p.name]
            for p in function.parameters
            if p.name in context.arguments
        }
        try:
            decision = await gate_decision(
                spec, arguments, port=port, integration=TARGET, call_id=call_id, run_id=run_id
            )
        except ApprovalPending as exc:
            logger.info("semantic_kernel: %s (call %s) waits for approval", spec.name, call_id)
            _set_result(context, pending_message(exc))
            return
        if decision is not None and not decision.approved:
            denied = ApprovalDenied(spec.name, call_id=call_id, reason=decision.reason)
            _set_result(context, json.dumps(model_visible_error(denied)))
            return
        gate = _Gate(run_id, TokenTable())
        if decision is not None:
            dumped = dump_arguments(spec, arguments) or {}
            qualified = str(function.fully_qualified_name)
            gate.tokens.issue(qualified, dumped, decision, call_id=call_id, run_id=run_id)
        token = _GATE.set(gate)
        try:
            await next(context)
        finally:
            _GATE.reset(token)  # unused tokens end with the invocation

    return approval


def _run_id(history: Any, fallback: str) -> str:
    """The run id of a conversation (created on first use), else the filter's own."""
    if history is None:
        return fallback
    key = id(history)  # chat histories are unhashable pydantic models: key by identity
    entry = _RUN_IDS.get(key)
    if entry is not None and entry[0]() is history:
        return entry[1]
    try:
        ref = weakref.ref(history, lambda _: _RUN_IDS.pop(key, None))
    except TypeError:  # not weak-referenceable
        return fallback
    run_id = new_id("run_")
    _RUN_IDS[key] = (ref, run_id)
    return run_id


def _spec(registry: Registry | None, plugin: str | None, function: Any) -> ToolSpec | None:
    """The registry's spec for the function; without a registry, the spec behind our method."""
    if registry is not None:
        name = _registry_name(function.fully_qualified_name, registry, plugin)
        return registry.get(name) if name else None
    spec = getattr(getattr(function, "method", None), _SPEC_ATTR, None)
    return spec if isinstance(spec, ToolSpec) else None


def _set_result(context: Any, value: str) -> None:
    from semantic_kernel.functions import FunctionResult

    context.function_result = FunctionResult(function=context.function.metadata, value=value)


FORMAT = "semantic_kernel"
_ROLES = {"system": "system", "developer": "system", "user": "user", "assistant": "assistant"}
_ROLES["tool"] = "tool"


def is_semantic_kernel_message(item: Any) -> bool:
    """True for a `ChatMessageContent`, found by class, without importing Semantic Kernel."""
    return any(
        cls.__name__ == "ChatMessageContent" and cls.__module__.startswith("semantic_kernel.")
        for cls in type(item).__mro__
    )


class SemanticKernelConverter(ItemsConverter):
    """`ChatMessageContent` lists (`ChatHistory.messages`): text, function calls and results."""

    name = FORMAT

    def role_of(self, native: Any) -> str | None:
        if not is_semantic_kernel_message(native):
            return None
        return _ROLES.get(str(getattr(native.role, "value", native.role)))

    def items_of(self, native: Any) -> list[Any]:
        return list(native.items)

    def read(self, item: Any) -> Item:
        kind = getattr(item, "content_type", None)
        if kind == "text":
            return Text(item.text)
        name = str(getattr(item, "name", None) or getattr(item, "function_name", "") or "")
        if kind == "function_call":
            raw = item.arguments
            arguments = arguments_of(dict(raw) if isinstance(raw, Mapping) else raw)
            return Call(ToolCall(id=str(item.id or ""), name=name, arguments=arguments))
        if kind == "function_result":
            return Result(str(item.id or ""), name, as_text(item.result))
        return None

    def with_items(self, native: Any, items: list[Any]) -> Any:
        return native.model_copy(update={"items": items})

    def text_item(self, text: str, like: Any | None) -> Any:
        if like is not None:
            return like.model_copy(update={"text": text})
        return _contents().TextContent(text=text)

    def with_result(self, item: Any, text: str) -> Any:
        return item.model_copy(update={"result": text})

    def new(self, message: Message) -> Any:
        contents = _contents()
        items: list[Any] = [contents.TextContent(text=message.content)] if message.content else []
        if message.role == "tool":
            items = [
                contents.FunctionResultContent(
                    id=message.tool_call_id, name=message.name or None, result=message.content
                )
            ]
        items += [
            contents.FunctionCallContent(id=c.id, name=c.name, arguments=json.dumps(c.arguments))
            for c in message.tool_calls
        ]
        role = contents.AuthorRole(message.role)
        return contents.ChatMessageContent(role=role, items=items)


def _contents() -> Any:
    """`semantic_kernel.contents`, imported on first use.

    Raises:
        MissingDependency: `semantic-kernel` is not installed.
    """
    INTEGRATION.require()
    import semantic_kernel.contents as contents

    return contents


register_format(FORMAT, SemanticKernelConverter, detect=is_semantic_kernel_message)


__all__ = [
    "INTEGRATION",
    "SemanticKernelIntegration",
    "approval_filter",
    "kernel_function",
    "kernel_plugin",
]
