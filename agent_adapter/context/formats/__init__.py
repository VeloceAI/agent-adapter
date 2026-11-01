"""Compress any framework's message list: convert it to `Message`, compress, convert back.

| Format | Covers |
|---|---|
| `agent_adapter` | `Message` lists (`run_tool_loop`, `CallableModel`, your own code) |
| `langchain` | LangChain and LangGraph `BaseMessage` lists |
| `openai` | chat dicts and Responses items (OpenAI Agents SDK, any OpenAI-compatible API) |
| `anthropic` | Anthropic Messages dicts (Anthropic, Claude Agent SDK) |
| `gemini` | Gemini `contents`, dicts or `google.genai` objects (Gemini, Google ADK) |

Framework integrations add their own formats with `register_format` when imported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeVar, cast

from agent_adapter.context.compress import CompressionReport, compress
from agent_adapter.context.formats._registry import (
    detect_registered,
    register_format,
    registered,
    registered_names,
)
from agent_adapter.context.formats._shared import Conversion, Converter, Pinned
from agent_adapter.context.formats.anthropic import AnthropicConverter
from agent_adapter.context.formats.gemini import GeminiConverter, is_content
from agent_adapter.context.formats.langchain import LangChainConverter, is_langchain_message
from agent_adapter.context.formats.openai import OpenAIConverter, is_responses_item
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from agent_adapter.runtime.sync import run_sync

__all__ = [
    "FORMATS",
    "Conversion",
    "FormatName",
    "Pinned",
    "compress_for",
    "compress_for_sync",
    "convert",
    "detect_format",
    "register_format",
    "registered_names",
]

T = TypeVar("T")
FormatName = Literal["auto", "agent_adapter", "langchain", "openai", "anthropic", "gemini"]
_CONVERTERS: dict[str, type[Converter]] = {
    "langchain": LangChainConverter,
    "openai": OpenAIConverter,
    "anthropic": AnthropicConverter,
    "gemini": GeminiConverter,
}
FORMATS = ("agent_adapter", *_CONVERTERS)
_ANTHROPIC_BLOCKS = frozenset({"tool_use", "tool_result", "thinking", "redacted_thinking"})


def detect_format(messages: Sequence[Any]) -> str:
    """The format of a message list; plain `{"role", "content": str}` dicts count as openai.

    Raises:
        ConfigError: the items mix formats or are not messages of any known format.
    """
    items = list(messages)
    if all(isinstance(item, Message) for item in items):
        return "agent_adapter"
    if all(is_langchain_message(item) for item in items):
        return "langchain"
    if all(is_content(item) for item in items):
        return "gemini"
    framework = detect_registered(items)
    if framework is not None:
        return framework
    if all(isinstance(item, Mapping) and "parts" not in item for item in items):
        return "anthropic" if any(_anthropic(item) for item in items) else "openai"
    kinds = sorted({f"{type(item).__module__}.{type(item).__qualname__}" for item in items})
    raise ConfigError(
        f"cannot tell the message format of {kinds}; pass format= explicitly, and for a"
        " framework's own types import its agent_adapter.integrations module first"
    )


def convert(
    messages: Sequence[T], format: FormatName | str = "auto", *, pinned: Pinned | None = None
) -> Conversion[T]:
    """A `Conversion`: `.messages` to compress, `.back(compressed)` for the native list.

    `pinned(native)` marks messages no step may change or remove.

    Raises:
        ConfigError: unknown format, `agent_adapter` (no conversion needed), or an item
            that is not a message of the format.
        MissingDependency: a framework format needs its library to build new messages.
    """
    name = detect_format(messages) if format == "auto" else format
    spec = registered(name)
    converter = _CONVERTERS.get(name) or (spec.converter if spec else None)
    if converter is None:
        if name == "agent_adapter":
            raise ConfigError("Message lists need no conversion; call compress() directly")
        raise ConfigError(
            f"unknown message format {name!r}; use one of {(*FORMATS, *registered_names())}"
            f" (a framework format registers when agent_adapter.integrations.{name} is imported)"
        )
    return Conversion(converter(messages), pinned)


async def compress_for(
    messages: Sequence[T],
    policy: CompressionPolicy,
    *,
    format: FormatName | str = "auto",
    pinned: Pinned | None = None,
    model_max_tokens: int | None = None,
    ctx: ToolContext | None = None,
) -> tuple[list[T], CompressionReport]:
    """`compress()` for any message format: the compressed list in that format, and a report.

    Messages kept as they were come back as the same objects; nothing calls the main model.

    Raises:
        ConfigError: as `compress` and `convert`.
        MissingDependency: as `convert`.
    """
    items = list(messages)
    name = detect_format(items) if format == "auto" else format
    if name == "agent_adapter":
        own = await _compress_own(items, policy, pinned, model_max_tokens, ctx)
        return cast("tuple[list[T], CompressionReport]", own)
    conversion = convert(items, name, pinned=pinned)
    smaller, report = await compress(
        conversion.messages, policy, model_max_tokens=model_max_tokens, ctx=ctx
    )
    return conversion.back(smaller), report


def compress_for_sync(
    messages: Sequence[T],
    policy: CompressionPolicy,
    *,
    format: FormatName | str = "auto",
    pinned: Pinned | None = None,
    model_max_tokens: int | None = None,
    ctx: ToolContext | None = None,
) -> tuple[list[T], CompressionReport]:
    """`compress_for` for sync code (Flask, Celery, scripts, a sync LangGraph node).

    Raises:
        ConfigError: as `compress_for`.
        MissingDependency: as `compress_for`.
    """
    return run_sync(
        lambda: compress_for(
            messages,
            policy,
            format=format,
            pinned=pinned,
            model_max_tokens=model_max_tokens,
            ctx=ctx,
        )
    )


async def _compress_own(
    messages: list[Any],
    policy: CompressionPolicy,
    pinned: Pinned | None,
    model_max_tokens: int | None,
    ctx: ToolContext | None,
) -> tuple[list[Message], CompressionReport]:
    """`compress` on Messages; `pinned` pins copies, and kept copies map back to the input."""
    if pinned is None:
        return await compress(messages, policy, model_max_tokens=model_max_tokens, ctx=ctx)
    if not callable(pinned):
        raise ConfigError("pinned must be a callable(native message) -> bool")
    copies = [m.model_copy(update={"pinned": True}) if pinned(m) else m for m in messages]
    originals = {id(copy): message for copy, message in zip(copies, messages, strict=True)}
    smaller, report = await compress(copies, policy, model_max_tokens=model_max_tokens, ctx=ctx)
    return [originals.get(id(m), m) for m in smaller], report


def _anthropic(item: Mapping[str, Any]) -> bool:
    """True for a dict only Anthropic writes: tool_use, tool_result or thinking blocks."""
    if is_responses_item(item):
        return False
    content = item.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, Mapping) and b.get("type") in _ANTHROPIC_BLOCKS for b in content)
