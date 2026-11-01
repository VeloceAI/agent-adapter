"""Pydantic AI message history (`ModelRequest`, `ModelResponse`) as a compression format.

Importing this module registers the "pydantic_ai" format. `pydantic_ai` is imported
lazily, only to build messages a compression step created.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from agent_adapter.context.formats import Pinned, register_format
from agent_adapter.context.formats._shared import not_a_message
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.core.errors import MissingDependency
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations._compress_hooks import Shrink
from agent_adapter.integrations._compress_parts import PartsConverter, Piece

NAME = "pydantic_ai"
_KINDS = {"ModelRequest": "request", "ModelResponse": "response"}
_USER_KINDS = frozenset({"user-prompt", "retry-feedback"})


def is_pydantic_ai_message(item: Any) -> bool:
    """True for a `ModelRequest` or `ModelResponse`, found by class, without importing it."""
    return _kind(item) is not None


def _kind(item: Any) -> str | None:
    for cls in type(item).__mro__:
        if cls.__name__ in _KINDS and cls.__module__.startswith("pydantic_ai."):
            return _KINDS[cls.__name__]
    return None


def _role(part: Any) -> str | None:
    """The role a request part speaks in; None for parts the model reads as state."""
    kind = getattr(part, "part_kind", None)
    if kind == "system-prompt":
        return "system"
    if kind in _USER_KINDS:
        return "user"
    if kind == "tool-return":
        return "tool"
    if kind == "retry-prompt":  # legacy: a tool retry when it names its tool
        return "tool" if getattr(part, "tool_name", None) else "user"
    return None


def _text(value: Any) -> str:
    """Text of a part's content: a str, the str items of a sequence, else JSON."""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "\n".join(item for item in value if isinstance(item, str))
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _tool_text(part: Any) -> str:
    """What the model reads for a tool return (or a legacy retry naming its tool)."""
    for name in ("model_response_str", "model_response"):
        render = getattr(part, name, None)
        if callable(render):
            return str(render())
    return _text(part.content)


class PydanticAIConverter(PartsConverter):
    """A response is one assistant message; a request splits into system, user and tool parts.

    Parts the model reads as state (instruction deltas, compaction) ride with the nearest
    message part. A tool return with outcome "denied" is pinned (approval state).
    """

    name = NAME

    def pieces(self, index: int, item: Any) -> list[Piece]:
        kind = _kind(item)
        if kind is None:
            raise not_a_message(NAME, index, item)
        if kind == "response":
            return [(_assistant(item.parts), None)]
        return _request_pieces(item.parts)

    def retext(self, item: Any, message: Message) -> Any:
        return dataclasses.replace(item, parts=_with_response_text(item.parts, message.content))

    def parts(self, item: Any) -> Sequence[Any]:
        return list(item.parts)

    def with_parts(self, item: Any, parts: list[Any]) -> Any:
        return dataclasses.replace(item, parts=parts)

    def retext_parts(self, item: Any, parts: list[Any], message: Message) -> list[Any]:
        out: list[Any] = []
        placed = False
        for part in parts:
            role = _role(part)
            if role is None:
                out.append(part)
            elif role == "tool":
                typed = getattr(part, "tool_kind", None) is not None  # framework-typed content
                out.append(part if typed else dataclasses.replace(part, content=message.content))
            elif not placed:
                out.append(_with_text(part, message.content))
                placed = True
            elif _media(part):
                out.append(dataclasses.replace(part, content=_media(part)))
        return out

    def render(self, messages: list[Message]) -> list[Any]:
        m = _messages_module()
        out: list[Any] = []
        for message in messages:
            if message.role == "assistant":
                calls = [
                    m.ToolCallPart(c.name, dict(c.arguments), c.id) for c in message.tool_calls
                ]
                text = [m.TextPart(message.content)] if message.content else []
                out.append(m.ModelResponse(parts=[*text, *calls]))
            elif message.role == "tool":
                part = m.ToolReturnPart(message.name, message.content, message.tool_call_id)
                out.append(m.ModelRequest(parts=[part]))
            elif message.role == "system":
                out.append(m.ModelRequest(parts=[m.SystemPromptPart(message.content)]))
            else:
                out.append(m.ModelRequest(parts=[m.UserPromptPart(message.content)]))
        return out


def _assistant(parts: Sequence[Any]) -> Message:
    texts = [p.content for p in parts if getattr(p, "part_kind", None) == "text"]
    calls = tuple(
        ToolCall(id=str(p.tool_call_id), name=str(p.tool_name), arguments=p.args_as_dict())
        for p in parts
        if getattr(p, "part_kind", None) == "tool-call"
    )
    return Message(role="assistant", content="\n".join(t for t in texts if t), tool_calls=calls)


def _request_pieces(parts: Sequence[Any]) -> list[Piece]:
    """One Message per system or tool part, one per run of user parts; state parts ride."""
    entries: list[tuple[Message, list[int]]] = []
    waiting: list[int] = []  # state parts before the next message part
    for position, part in enumerate(parts):
        role = _role(part)
        if role is None:
            waiting.append(position)
            continue
        message = _part_message(role, part)
        if role == "user" and entries and entries[-1][0].role == "user" and not waiting:
            joined = "\n".join(t for t in (entries[-1][0].content, message.content) if t)
            entries[-1] = (entries[-1][0].model_copy(update={"content": joined}), entries[-1][1])
            entries[-1][1].append(position)
            continue
        entries.append((message, [*waiting, position]))
        waiting = []
    if entries and waiting:
        entries[-1][1].extend(waiting)
    return [(message, tuple(owned)) for message, owned in entries]


def _part_message(role: str, part: Any) -> Message:
    if role == "tool":
        return Message(
            role="tool",
            tool_call_id=str(part.tool_call_id),
            name=str(part.tool_name or ""),
            content=_tool_text(part),
            pinned=getattr(part, "outcome", None) == "denied",
        )
    if getattr(part, "part_kind", None) == "user-prompt" or role == "system":
        return Message(role=role, content=_text(part.content))
    return Message(role="user", content=part.model_response())  # feedback, legacy retry


def _media(part: Any) -> list[Any]:
    """The non-text items of a user prompt part (images, files); none for other parts."""
    if getattr(part, "part_kind", None) != "user-prompt" or isinstance(part.content, str):
        return []
    return [value for value in part.content if not isinstance(value, str)]


def _with_text(part: Any, text: str) -> Any:
    """A system, user or feedback part holding `text`; media items of a user part stay."""
    media = _media(part)
    return dataclasses.replace(part, content=[text, *media] if media else text)


def _with_response_text(parts: Sequence[Any], text: str) -> list[Any]:
    """Response parts with the first text part holding `text`; later text parts go."""
    out: list[Any] = []
    placed = False
    for part in parts:
        if getattr(part, "part_kind", None) != "text":
            out.append(part)
        elif not placed:
            out.append(dataclasses.replace(part, content=text))
            placed = True
    if not placed and text:
        out.insert(0, _messages_module().TextPart(text))
    return out


def _messages_module() -> Any:
    """`pydantic_ai.messages`.

    Raises:
        MissingDependency: pydantic-ai is not installed.
    """
    try:
        from pydantic_ai import messages
    except ImportError as exc:
        raise MissingDependency(
            'the pydantic_ai format needs pydantic-ai: pip install "agent-adapter[pydantic-ai]"'
        ) from exc
    return messages


def pydantic_ai_history_processor(
    policy: CompressionPolicy,
    *,
    pinned: Pinned | None = None,
    model_max_tokens: int | None = None,
) -> Callable[[list[Any]], Awaitable[list[Any]]]:
    """A history processor for `Agent(capabilities=[ProcessHistory(...)])`.

    Pydantic AI keeps the processed history for the rest of the run.

    Raises:
        ConfigError: `policy` is not a `CompressionPolicy` or `pinned` is not callable.
    """
    shrink = Shrink(policy, NAME, pinned, model_max_tokens)

    async def compress_history(messages: list[Any]) -> list[Any]:
        return await shrink(messages)

    return compress_history


register_format(NAME, PydanticAIConverter, detect=is_pydantic_ai_message)
