"""Agno `Message` lists as a compression format, and a model wrapper that compresses input.

Importing this module registers the "agno" format. `agno` is imported lazily, only to
build messages a compression step created.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent_adapter.context.formats import Pinned, register_format
from agent_adapter.context.formats._shared import arguments_of, not_a_message, text_of, with_text
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.core.errors import MissingDependency
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations._compress_hooks import Kind, Shrink, wrap_model
from agent_adapter.integrations._compress_parts import PartsConverter, Piece

NAME = "agno"
TEXT = frozenset({"text"})
_ROLES = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "assistant": "assistant",
    "model": "assistant",
    "tool": "tool",
}
# Agno's Model calls each of these with messages=; the stream ones are generators.
_METHODS: dict[str, Kind] = {
    "invoke": "call",
    "ainvoke": "acall",
    "invoke_stream": "iter",
    "ainvoke_stream": "aiter",
}


def is_agno_message(item: Any) -> bool:
    """True for an `agno.models.message.Message`, found by class, without importing Agno."""
    return any(
        cls.__name__ == "Message" and cls.__module__.startswith("agno.")
        for cls in type(item).__mro__
    )


def _call(call: Any) -> ToolCall:
    function = call.get("function") if isinstance(call, Mapping) else None
    function = function if isinstance(function, Mapping) else {}
    return ToolCall(
        id=str(call.get("id") or ""),
        name=str(function.get("name") or ""),
        arguments=arguments_of(function.get("arguments")),
    )


class AgnoConverter(PartsConverter):
    """System, user, assistant (OpenAI-style `tool_calls`) and tool messages."""

    name = NAME

    def pieces(self, index: int, item: Any) -> list[Piece]:
        role = _ROLES.get(str(getattr(item, "role", ""))) if is_agno_message(item) else None
        if role is None:
            raise not_a_message(NAME, index, item)
        text = text_of(item.content, TEXT)
        if role == "assistant":
            calls = tuple(_call(c) for c in item.tool_calls or () if isinstance(c, Mapping))
            message = Message(role="assistant", content=text, tool_calls=calls)
        elif role == "tool":
            message = Message(
                role="tool",
                tool_call_id=str(item.tool_call_id or ""),
                name=str(item.tool_name or item.name or ""),
                content=text,
            )
        else:
            message = Message(role=role, content=text)
        return [(message, None)]

    def retext(self, item: Any, message: Message) -> Any:
        return item.model_copy(update={"content": with_text(item.content, message.content, TEXT)})

    def render(self, messages: list[Message]) -> list[Any]:
        agno_message = _message_class()
        out: list[Any] = []
        for message in messages:
            if message.role == "tool":
                out.append(
                    agno_message(
                        role="tool",
                        content=message.content,
                        tool_call_id=message.tool_call_id,
                        tool_name=message.name or None,
                    )
                )
            elif message.role == "assistant" and message.tool_calls:
                calls = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": _json(c.arguments)},
                    }
                    for c in message.tool_calls
                ]
                out.append(
                    agno_message(role="assistant", content=message.content, tool_calls=calls)
                )
            else:
                out.append(agno_message(role=message.role, content=message.content))
        return out


def _json(arguments: Mapping[str, Any]) -> str:
    return json.dumps(dict(arguments), ensure_ascii=False, default=str)


def _message_class() -> Any:
    """`agno.models.message.Message`.

    Raises:
        MissingDependency: agno is not installed.
    """
    try:
        from agno.models.message import Message as AgnoMessage
    except ImportError as exc:
        raise MissingDependency(
            'the agno format needs agno: pip install "agent-adapter[agno]"'
        ) from exc
    return AgnoMessage


def agno_compressed_model(
    model: Any,
    policy: CompressionPolicy,
    *,
    pinned: Pinned | None = None,
    model_max_tokens: int | None = None,
) -> Any:
    """A copy of an Agno model that compresses the messages of each provider call.

    Use it as `Agent(model=agno_compressed_model(OpenAIChat(...), policy))`. The run's own
    message list (and the stored session) keeps the full history.

    Raises:
        ConfigError: `model` has no `invoke` methods, `policy` is not a `CompressionPolicy`,
            or `pinned` is not callable.
    """
    return wrap_model(model, Shrink(policy, NAME, pinned, model_max_tokens), _METHODS)


register_format(NAME, AgnoConverter, detect=is_agno_message)
