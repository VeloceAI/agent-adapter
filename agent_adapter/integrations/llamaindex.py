"""LlamaIndex: registry tools as `FunctionTool`s, and the `llamaindex` compression format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_adapter.context.formats import register_format
from agent_adapter.context.formats._items import Call, Item, ItemsConverter, Text
from agent_adapter.context.formats._shared import arguments_of
from agent_adapter.core.messages import Message
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations.base import IntegrationBase, ToolRunner
from agent_adapter.runtime.sync import run_sync

if TYPE_CHECKING:
    from llama_index.core.tools import FunctionTool

TARGET = "llamaindex"


def llamaindex_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> FunctionTool:
    """A `FunctionTool` with `fn_schema=spec.args_model`; `acall` is async, `call` uses `run_sync`.

    Approval comes from the executor's port (LlamaIndex has no gate of its own). Invalid
    arguments, tool errors, denials, guardrails and run limits come back as the tool's
    `raw_output` (`{"error": {...}}`); other errors propagate.

    Raises:
        MissingDependency: `llama-index-core` is not installed.
        ToolUsageError: `spec.targets` excludes `llamaindex`.
    """
    INTEGRATION.require()
    check_target(spec, TARGET)
    from llama_index.core.tools import FunctionTool

    runner = lazy_runner(executor)
    native: list[Any] = []  # the FunctionTool, once built: bind_context's native object

    async def acall(**arguments: Any) -> Any:
        return await INTEGRATION._run_bound(native[0], spec, arguments, runner)

    def call(**arguments: Any) -> Any:
        return run_sync(lambda: acall(**arguments))

    tool = FunctionTool.from_defaults(
        fn=call,
        async_fn=acall,
        name=spec.name,
        description=spec.description,
        fn_schema=spec.args_model,
    )
    native.append(tool)
    return tool


def llamaindex_tools(
    registry: Registry, *, executor: ToolRunner | None = None
) -> list[FunctionTool]:
    """`llamaindex_tool` for every registry tool exported to `llamaindex`."""
    return [llamaindex_tool(spec, executor=executor) for spec in registry.for_target(TARGET)]


class LlamaIndexIntegration(IntegrationBase):
    """Exports `FunctionTool`s; LlamaIndex has no call id at the tool boundary."""

    name = TARGET
    requires = ("llama_index.core",)
    extra = "llamaindex"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> FunctionTool:
        """`llamaindex_tool(spec, executor=executor)`."""
        return llamaindex_tool(spec, executor=executor)


INTEGRATION = LlamaIndexIntegration()
FORMAT = "llamaindex"
_ROLES = {"system": "system", "developer": "system", "user": "user", "assistant": "assistant"}
_ROLES |= {"chatbot": "assistant", "model": "assistant", "tool": "tool", "function": "tool"}


def is_llamaindex_message(item: Any) -> bool:
    """True for a LlamaIndex `ChatMessage`, found by class, without importing LlamaIndex."""
    return any(
        cls.__name__ == "ChatMessage" and cls.__module__.startswith("llama_index.core.")
        for cls in type(item).__mro__
    )


class LlamaIndexConverter(ItemsConverter):
    """`ChatMessage` lists: text and `ToolCallBlock` blocks, and tool messages.

    Older LLM classes keep tool calls in `additional_kwargs["tool_calls"]`; those are read too.
    """

    name = FORMAT

    def role_of(self, native: Any) -> str | None:
        if not is_llamaindex_message(native):
            return None
        return _ROLES.get(str(getattr(native.role, "value", native.role)))

    def items_of(self, native: Any) -> list[Any]:
        return list(native.blocks)

    def read(self, item: Any) -> Item:
        kind = getattr(item, "block_type", None)
        if kind == "text":
            return Text(item.text)
        if kind == "tool_call":
            call = ToolCall(
                id=str(item.tool_call_id or ""),
                name=str(item.tool_name),
                arguments=arguments_of(item.tool_kwargs),
            )
            return Call(call)
        return None

    def tool_result_of(self, native: Any) -> tuple[str, str] | None:
        if self.role_of(native) != "tool":
            return None
        extra = native.additional_kwargs
        return str(extra.get("tool_call_id") or ""), str(extra.get("name") or "")

    def extra_calls(self, native: Any) -> tuple[ToolCall, ...]:
        calls = native.additional_kwargs.get("tool_calls") or ()
        return tuple(_kwargs_call(call) for call in calls)

    def with_items(self, native: Any, items: list[Any]) -> Any:
        return native.model_copy(update={"blocks": items})

    def text_item(self, text: str, like: Any | None) -> Any:
        if like is not None:
            return like.model_copy(update={"text": text})
        return _types().TextBlock(text=text)

    def with_result(self, item: Any, text: str) -> Any:
        return item  # tool results are whole messages here, never items

    def new(self, message: Message) -> Any:
        types = _types()
        blocks: list[Any] = [types.TextBlock(text=message.content)] if message.content else []
        extra: dict[str, Any] = {}
        if message.role == "tool":
            extra["tool_call_id"] = message.tool_call_id
            blocks = [types.TextBlock(text=message.content)]
        blocks += [
            types.ToolCallBlock(tool_call_id=c.id, tool_name=c.name, tool_kwargs=dict(c.arguments))
            for c in message.tool_calls
        ]
        return types.ChatMessage(role=message.role, blocks=blocks, additional_kwargs=extra)


def _kwargs_call(call: Any) -> ToolCall:
    """A tool call from `additional_kwargs["tool_calls"]`: an OpenAI object or its dict."""
    function = _field(call, "function") or {}
    arguments = arguments_of(_field(function, "arguments"))
    name = str(_field(function, "name") or "")
    return ToolCall(id=str(_field(call, "id") or ""), name=name, arguments=arguments)


def _field(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def _types() -> Any:
    """`llama_index.core.base.llms.types`, imported on first use.

    Raises:
        MissingDependency: `llama-index-core` is not installed.
    """
    INTEGRATION.require()
    from llama_index.core.base.llms import types

    return types


register_format(FORMAT, LlamaIndexConverter, detect=is_llamaindex_message)


__all__ = [
    "INTEGRATION",
    "LlamaIndexIntegration",
    "llamaindex_tool",
    "llamaindex_tools",
]
