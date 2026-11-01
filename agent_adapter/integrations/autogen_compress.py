"""AutoGen `LLMMessage` lists as a compression format, and a compressing model context.

Importing this module registers the "autogen" format. `autogen_core` is imported lazily,
only to build messages a compression step created and the model context class.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import Any

from agent_adapter.context.formats import Pinned, register_format
from agent_adapter.context.formats._shared import arguments_of, not_a_message
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.core.errors import MissingDependency
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations._compress_hooks import Shrink
from agent_adapter.integrations._compress_parts import PartsConverter, Piece

NAME = "autogen"
_TYPES = {
    "SystemMessage": "system",
    "UserMessage": "user",
    "AssistantMessage": "assistant",
    "FunctionExecutionResultMessage": "tool",
}
_SOURCE = "agent_adapter"  # `source` of a message a compression step created


def is_autogen_message(item: Any) -> bool:
    """True for an `autogen_core.models` LLM message, found by class, without importing it."""
    return _kind(item) is not None


def _kind(item: Any) -> str | None:
    for cls in type(item).__mro__:
        if cls.__name__ in _TYPES and cls.__module__.startswith("autogen_core."):
            return _TYPES[cls.__name__]
    return None


def _text(content: Any) -> str:
    """A str, or the str items of a multimodal list (images are not text)."""
    if isinstance(content, str):
        return content
    return "\n".join(item for item in content if isinstance(item, str))


def _with_text(content: Any, text: str) -> Any:
    if isinstance(content, str):
        return text
    others = [item for item in content if not isinstance(item, str)]
    return [text, *others]


class AutoGenConverter(PartsConverter):
    """System, user, assistant (text or `FunctionCall`s) and function-result messages.

    Each `FunctionExecutionResult` is one tool message; an assistant `thought` is kept but
    is not text a step changes.
    """

    name = NAME

    def pieces(self, index: int, item: Any) -> list[Piece]:
        kind = _kind(item)
        if kind is None:
            raise not_a_message(NAME, index, item)
        if kind == "tool":
            return [
                (
                    Message(
                        role="tool",
                        tool_call_id=str(result.call_id),
                        name=str(result.name),
                        content=str(result.content),
                    ),
                    (position,),
                )
                for position, result in enumerate(item.content)
            ]
        if kind == "assistant" and not isinstance(item.content, str):
            calls = tuple(
                ToolCall(id=str(c.id), name=str(c.name), arguments=arguments_of(c.arguments))
                for c in item.content
            )
            return [(Message(role="assistant", content="", tool_calls=calls), None)]
        return [(Message(role=kind, content=_text(item.content)), None)]

    def retext(self, item: Any, message: Message) -> Any:
        if _kind(item) == "assistant" and not isinstance(item.content, str):
            return item  # function calls carry no text
        return item.model_copy(update={"content": _with_text(item.content, message.content)})

    def parts(self, item: Any) -> Sequence[Any]:
        return list(item.content)

    def with_parts(self, item: Any, parts: list[Any]) -> Any:
        return item.model_copy(update={"content": parts})

    def retext_parts(self, item: Any, parts: list[Any], message: Message) -> list[Any]:
        return [parts[0].model_copy(update={"content": message.content})]

    def render(self, messages: list[Message]) -> list[Any]:
        m = _models()
        out: list[Any] = []
        for message in messages:
            if message.role == "tool":
                result = m.FunctionExecutionResult(
                    content=message.content, name=message.name, call_id=message.tool_call_id
                )
                out.append(m.FunctionExecutionResultMessage(content=[result]))
            elif message.role == "assistant" and message.tool_calls:
                calls = [_function_call(c) for c in message.tool_calls]
                out.append(m.AssistantMessage(content=calls, source=_SOURCE))
            elif message.role == "assistant":
                out.append(m.AssistantMessage(content=message.content, source=_SOURCE))
            elif message.role == "system":
                out.append(m.SystemMessage(content=message.content))
            else:
                out.append(m.UserMessage(content=message.content, source=_SOURCE))
        return out


def _function_call(call: ToolCall) -> Any:
    from autogen_core import FunctionCall

    return FunctionCall(id=call.id, name=call.name, arguments=json.dumps(dict(call.arguments)))


def _models() -> Any:
    """`autogen_core.models`.

    Raises:
        MissingDependency: autogen-core is not installed.
    """
    try:
        from autogen_core import models
    except ImportError as exc:
        raise MissingDependency(
            'the autogen format needs autogen-core: pip install "agent-adapter[autogen]"'
        ) from exc
    return models


def autogen_model_context(
    policy: CompressionPolicy,
    *,
    initial_messages: Sequence[Any] | None = None,
    pinned: Pinned | None = None,
    model_max_tokens: int | None = None,
) -> Any:
    """A `ChatCompletionContext` that keeps every message and returns them compressed.

    Use it as `AssistantAgent(..., model_context=autogen_model_context(policy))`. It holds a
    callable policy, so `dump_component()` cannot save it.

    Raises:
        ConfigError: `policy` is not a `CompressionPolicy` or `pinned` is not callable.
        MissingDependency: autogen-core is not installed.
    """
    shrink = Shrink(policy, NAME, pinned, model_max_tokens)
    _models()  # fail fast with MissingDependency
    return _context_class()(shrink, list(initial_messages or ()))


@functools.cache
def _context_class() -> type:
    """The model context class, built once: its base comes from `autogen_core`."""
    from autogen_core.model_context import ChatCompletionContext

    class CompressingChatCompletionContext(ChatCompletionContext):  # type: ignore[misc]
        """Every message is kept; `get_messages` returns them compressed."""

        def __init__(self, shrink: Shrink, initial_messages: list[Any]) -> None:
            super().__init__(initial_messages=initial_messages)
            self._shrink = shrink

        async def get_messages(self) -> list[Any]:
            return await self._shrink(list(self._messages))

    return CompressingChatCompletionContext


register_format(NAME, AutoGenConverter, detect=is_autogen_message)
