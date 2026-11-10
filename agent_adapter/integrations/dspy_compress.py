"""DSPy prompts as a compression format: lm15 `Message`s (DSPy 3.5+) or chat dicts (3.4).

Importing this module registers the "dspy" format for the lm15 messages in a DSPy 3.5
`Prompt`. DSPy 3.4 adapters return OpenAI chat dicts, which the "openai" format covers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from agent_adapter.context.compress import CompressionReport
from agent_adapter.context.formats import Pinned, compress_for_sync, register_format
from agent_adapter.context.formats._shared import not_a_message
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.core.errors import MissingDependency
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations._compress_parts import PartsConverter, Piece

NAME = "dspy"
_ROLES = {"user": "user", "assistant": "assistant", "developer": "system"}


def is_dspy_message(item: Any) -> bool:
    """True for an lm15 `Message` (bundled as `dspy.lm15`), found by class, without importing."""
    cls = type(item)
    return cls.__name__ == "Message" and cls.__module__.startswith(("dspy.", "lm15."))


def _texts(parts: Sequence[Any]) -> str:
    return "\n".join(p.text for p in parts if getattr(p, "type", "") == "text" and p.text)


def _with_text(parts: Sequence[Any], text: str, text_part: Any) -> tuple[Any, ...]:
    """`parts` with the first text part holding `text`; later text parts go."""
    out: list[Any] = []
    placed = False
    for part in parts:
        if getattr(part, "type", "") != "text":
            out.append(part)
        elif not placed:
            out.append(dataclasses.replace(part, text=text))
            placed = True
    if not placed:
        out.insert(0, text_part(text))
    return tuple(out)


class DSPyConverter(PartsConverter):
    """User, assistant and developer messages whole; each tool result is its own tool message."""

    name = NAME

    def pieces(self, index: int, item: Any) -> list[Piece]:
        role = getattr(item, "role", "") if is_dspy_message(item) else ""
        if role == "tool":
            return [
                (
                    Message(
                        role="tool",
                        tool_call_id=part.id,
                        name=part.name or "",
                        content=_texts(part.content),
                    ),
                    (position,),
                )
                for position, part in enumerate(item.parts)
            ]
        if role not in _ROLES:
            raise not_a_message(NAME, index, item)
        calls = tuple(
            ToolCall(id=p.id, name=p.name, arguments=dict(p.input))
            for p in item.parts
            if getattr(p, "type", "") == "tool_call"
        )
        text = _texts(item.parts)
        return [(Message(role=_ROLES[role], content=text, tool_calls=calls), None)]

    def retext(self, item: Any, message: Message) -> Any:
        return dataclasses.replace(
            item, parts=_with_text(item.parts, message.content, _lm15().TextPart)
        )

    def parts(self, item: Any) -> Sequence[Any]:
        return list(item.parts)

    def with_parts(self, item: Any, parts: list[Any]) -> Any:
        return dataclasses.replace(item, parts=tuple(parts))

    def retext_parts(self, item: Any, parts: list[Any], message: Message) -> list[Any]:
        result = parts[0]
        content = _with_text(result.content, message.content, _lm15().TextPart)
        return [dataclasses.replace(result, content=content)]

    def render(self, messages: list[Message]) -> list[Any]:
        lm15 = _lm15()
        out: list[Any] = []
        for message in messages:
            if message.role == "tool":
                out.append(lm15.Message.tool(message.tool_call_id, message.content))
            elif message.role == "assistant":
                calls = [
                    lm15.ToolCallPart(id=c.id, name=c.name, input=dict(c.arguments))
                    for c in message.tool_calls
                ]
                text = [lm15.TextPart(message.content)] if message.content or not calls else []
                out.append(lm15.Message(role="assistant", parts=[*text, *calls]))
            elif message.role == "system":
                out.append(lm15.Message.developer(message.content))
            else:
                out.append(lm15.Message.user(message.content))
        return out


def _lm15() -> Any:
    """`dspy.lm15`, the lm15 types DSPy 3.5+ bundles.

    Raises:
        MissingDependency: dspy 3.5 or newer is not installed.
    """
    try:
        from dspy import lm15
    except ImportError as exc:
        raise MissingDependency(
            'the dspy format needs dspy>=3.5: pip install "agent-adapter[dspy]"'
        ) from exc
    return lm15


def dspy_compress_prompt(
    prompt: Any,
    policy: CompressionPolicy,
    *,
    pinned: Pinned | None = None,
    model_max_tokens: int | None = None,
) -> tuple[Any, CompressionReport]:
    """What a DSPy `Adapter.format` returned, compressed, and a report.

    A DSPy 3.5+ `Prompt` comes back as a `Prompt`; its system text is counted and kept.
    DSPy 3.4's OpenAI chat dicts come back as chat dicts.

    Raises:
        ConfigError: as `compress_for`.
    """
    if isinstance(prompt, list):
        return compress_for_sync(
            prompt, policy, format="openai", pinned=pinned, model_max_tokens=model_max_tokens
        )
    system = _lm15().Message.developer(prompt.system) if prompt.system else None

    def keep(native: Any) -> bool:
        return native is system or (pinned is not None and pinned(native))

    smaller, report = compress_for_sync(
        [system, *prompt.messages] if system else list(prompt.messages),
        policy,
        format=NAME,
        pinned=keep,
        model_max_tokens=model_max_tokens,
    )
    messages = tuple(m for m in smaller if m is not system)
    return dataclasses.replace(prompt, messages=messages), report


register_format(NAME, DSPyConverter, detect=is_dspy_message)
