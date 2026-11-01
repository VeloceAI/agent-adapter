"""AG2 stream history (events) as a compression format, and an assembly policy.

Importing this module registers the "ag2" format. Events the model reads (requests,
responses, tool results, compaction summaries) become messages; every other event
(telemetry, provider replay state, call announcements) rides with its turn. `ag2` is
imported lazily, only to build events a compression step created.
"""

from __future__ import annotations

import copy
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

NAME = "ag2"
_KINDS = {
    "ModelRequest": "user",
    "ModelResponse": "assistant",
    "ToolResultsEvent": "tool",
    "CompactionSummary": "summary",
}
_TEXT_INPUTS = frozenset({"TextInput", "DataInput"})


def _names(value: Any) -> set[str]:
    return {cls.__name__ for cls in type(value).__mro__ if cls.__module__.startswith("ag2.")}


def is_ag2_event(item: Any) -> bool:
    """True for an AG2 event (`ag2.events.BaseEvent`), found by class, without importing it."""
    return "BaseEvent" in _names(item)


def _kind(item: Any) -> str | None:
    """How the model reads an event; None for events it never sees."""
    names = _names(item)
    return next((kind for name, kind in _KINDS.items() if name in names), None)


def _input_text(part: Any) -> str | None:
    """The text of a text or data input; None for files, images and other media."""
    names = _names(part)
    if "TextInput" in names:
        return str(part.content)
    if "DataInput" in names:
        return json.dumps(part.data, ensure_ascii=False, sort_keys=True, default=str)
    return None


def _inputs_text(parts: Sequence[Any]) -> str:
    texts = [_input_text(part) for part in parts]
    return "\n".join(t for t in texts if t)


def _with_text(parts: Sequence[Any], text: str) -> list[Any]:
    """Inputs with one text input holding `text` in place of every text and data input."""
    out = [part for part in parts if _input_text(part) is None]
    return [_ag2().events.TextInput(text), *out]


class AG2Converter(PartsConverter):
    """Requests are user turns, responses assistant turns with their calls, and each result in
    a `ToolResultsEvent` one tool message; a `CompactionSummary` is a user turn."""

    name = NAME

    def pieces(self, index: int, item: Any) -> list[Piece]:
        if not is_ag2_event(item):
            raise not_a_message(NAME, index, item)
        kind = _kind(item)
        if kind is None:
            return []  # rides with its turn
        if kind == "assistant":
            text = item.message.content if item.message is not None else ""
            calls = tuple(
                ToolCall(id=str(c.id), name=str(c.name), arguments=arguments_of(c.arguments))
                for c in item.tool_calls.calls
            )
            return [(Message(role="assistant", content=text or "", tool_calls=calls), None)]
        if kind == "tool":
            return [
                (
                    Message(
                        role="tool",
                        tool_call_id=str(result.parent_id),
                        name=str(result.name or ""),
                        content=_inputs_text(result.result.parts),
                    ),
                    (position,),
                )
                for position, result in enumerate(item.results)
            ]
        text = item.summary if kind == "summary" else _inputs_text(item.parts)
        return [(Message(role="user", content=text), None)]

    def rider_target(self, rider: int, before: int | None, after: int | None) -> int | None:
        """A rider joins the model response of its span, else the next turn, else the last."""
        for side in (after, before):
            if side is not None and _kind(self.items[side]) == "assistant":
                return side
        return after if after is not None else before

    def retext(self, item: Any, message: Message) -> Any:
        new = copy.copy(item)
        kind = _kind(item)
        if kind == "assistant":
            if item.message is None:
                return item  # no text to change
            new.message = copy.copy(item.message)
            new.message.content = message.content
        elif kind == "summary":
            new.summary = message.content
        else:
            new.parts = _with_text(item.parts, message.content)
        return new

    def parts(self, item: Any) -> Sequence[Any]:
        return list(item.results)

    def with_parts(self, item: Any, parts: list[Any]) -> Any:
        new = copy.copy(item)
        new.results = parts
        return new

    def retext_parts(self, item: Any, parts: list[Any], message: Message) -> list[Any]:
        result = copy.copy(parts[0])
        result.result = copy.copy(parts[0].result)
        result.result.parts = _with_text(parts[0].result.parts, message.content)
        return [result]

    def render(self, messages: list[Message]) -> list[Any]:
        events = _ag2().events
        out: list[Any] = []
        for message in messages:
            if message.role == "assistant":
                calls = [
                    events.ToolCallEvent(c.name, id=c.id, arguments=json.dumps(dict(c.arguments)))
                    for c in message.tool_calls
                ]
                out.append(
                    events.ModelResponse(
                        events.ModelMessage(message.content),
                        tool_calls=events.ToolCallsEvent(calls),
                    )
                )
            elif message.role == "tool":
                result = events.ToolResultEvent(
                    parent_id=message.tool_call_id,
                    name=message.name or None,
                    result=events.ToolResult(message.content),
                )
                out.append(events.ToolResultsEvent([result]))
            else:
                out.append(events.ModelRequest([events.TextInput(message.content)]))
        return out


def _ag2() -> Any:
    """The `ag2` package, with `ag2.events` loaded.

    Raises:
        MissingDependency: ag2 is not installed.
    """
    try:
        import ag2
        import ag2.events
    except ImportError as exc:
        raise MissingDependency(
            'the ag2 format needs ag2: pip install "agent-adapter[ag2]"'
        ) from exc
    return ag2


class CompressAssemblyPolicy:
    """An AG2 assembly policy: the events of each model call, compressed.

    The stream history is never changed; only what this call sends is shorter.
    """

    name = "context_compression"

    def __init__(self, shrink: Shrink) -> None:
        self._shrink = shrink

    async def apply(
        self, prompts: list[str], events: list[Any], context: Any
    ) -> tuple[list[str], list[Any]]:
        """`(prompts, events)` with the events compressed; prompts stay as they are."""
        if not any(_kind(event) is not None for event in events):
            return prompts, events  # nothing the model reads
        return prompts, await self._shrink(events)


def ag2_compress_policy(
    policy: CompressionPolicy,
    *,
    pinned: Pinned | None = None,
    model_max_tokens: int | None = None,
) -> CompressAssemblyPolicy:
    """An assembly policy for `Agent(..., assembly=[ag2_compress_policy(policy)])`.

    Raises:
        ConfigError: `policy` is not a `CompressionPolicy` or `pinned` is not callable.
    """
    return CompressAssemblyPolicy(Shrink(policy, NAME, pinned, model_max_tokens))


register_format(NAME, AG2Converter, detect=is_ag2_event)
