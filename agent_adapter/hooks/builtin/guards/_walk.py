"""Find and rewrite the text inside a guarded payload, each text by its place in the payload."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.messages import LLMRequest, LLMResponse, Message

MAX_DEPTH = 32  # deeper values are scanned as their text, never skipped
PAIR_NAMES = ("name", "key", "header", "field")  # `{"name": X, "value": Y}` names Y by X

Path = tuple[Any, ...]
Swap = Callable[[Path, str], str]  # (where the text sits, the text) -> its replacement


@dataclass(frozen=True)
class Field:
    """One text in a payload: where it sits and the names of every key above it."""

    text: str
    path: Path
    keys: tuple[str, ...] = ()


def fields(stage: str, payload: Any) -> list[Field]:
    """Every text a stage's payload carries (keys included), in walk order."""
    if stage == "llm_request" and isinstance(payload, LLMRequest):
        found = [
            item
            for index, message in enumerate(payload.messages)
            for item in _message(message, ("messages", index))
        ]
    elif stage == "llm_response" and isinstance(payload, LLMResponse):
        found = [
            Field(payload.text, ("text",)),
            *_leaves(payload.parsed, ("parsed",), (), 0),
            *_calls_leaves(payload.tool_calls, ("tool_calls",)),
            *_message(payload.message, ("message",)),
        ]
    else:
        found = list(_leaves(payload, (), (), 0))
    return [item for item in found if item.text]


def rewrite(stage: str, payload: Any, swap: Swap) -> Any:
    """The payload with each text replaced by `swap(path, text)`; keys never collapse."""
    if stage == "llm_response" and isinstance(payload, LLMResponse):
        return payload.model_copy(
            update={
                "text": swap(("text",), payload.text),
                "parsed": _map(payload.parsed, swap, ("parsed",), 0),
                "tool_calls": _calls(payload.tool_calls, swap, ("tool_calls",)),
                "message": _rewrite_message(payload.message, swap, ("message",)),
            }
        )
    return _map(payload, swap, (), 0)


def _message(message: Message, path: Path) -> Iterator[Field]:
    """A message's text, its tool-call arguments and the raw items a port replays."""
    yield Field(message.content, (*path, "content"))
    yield from _calls_leaves(message.tool_calls, (*path, "tool_calls"))
    if message.raw is not None:
        # Raw items are the provider's own shapes: their key names (`signature`) are
        # metadata, not the caller's secrets, so only their text is checked.
        for field in _leaves(message.raw.items, (*path, "raw"), (), 0):
            yield Field(field.text, field.path)


def _rewrite_message(message: Message, swap: Swap, path: Path) -> Message:
    # Raw items are rewritten in place, not dropped: ports replay them in place of the
    # content, and untouched items (thinking blocks, thought signatures) stay valid.
    raw = message.raw
    if raw is not None:
        raw = raw.model_copy(update={"items": _map(raw.items, swap, (*path, "raw"), 0)})
    return message.model_copy(
        update={
            "content": swap((*path, "content"), message.content),
            "tool_calls": _calls(message.tool_calls, swap, (*path, "tool_calls")),
            "raw": raw,
        }
    )


def _calls_leaves(calls: tuple[Any, ...], path: Path) -> Iterator[Field]:
    for index, call in enumerate(calls):
        yield from _leaves(call.arguments, (*path, index), (), 0)


def _calls(calls: tuple[Any, ...], swap: Swap, path: Path) -> tuple[Any, ...]:
    return tuple(
        call.model_copy(update={"arguments": _map(call.arguments, swap, (*path, index), 0)})
        for index, call in enumerate(calls)
    )


def _pair_name(value: Any) -> str | None:
    """The name a `[name, value]` pair or a `{"name": X, "value": Y}` entry gives its value."""
    if isinstance(value, list | tuple) and len(value) == 2 and isinstance(value[0], str):
        return value[0]
    if isinstance(value, Mapping) and "value" in value:
        for name in PAIR_NAMES:
            if isinstance(value.get(name), str):
                return str(value[name])
    return None


def _leaves(value: Any, path: Path, keys: tuple[str, ...], depth: int) -> Iterator[Field]:
    if value is None or isinstance(value, bool | float):
        return
    if isinstance(value, str):
        yield Field(value, path, keys)
    elif isinstance(value, int) or depth > MAX_DEPTH:
        yield Field(str(value), path, keys)
    elif isinstance(value, Mapping):
        pair = _pair_name(value)
        for index, (name, item) in enumerate(value.items()):
            if isinstance(name, str):
                yield Field(name, (*path, "key", index))  # a key can be personal data too
            inner = (*keys, name) if isinstance(name, str) else keys
            if pair is not None and name == "value":
                inner = (*inner, pair)
            yield from _leaves(item, (*path, "value", index), inner, depth + 1)
    elif isinstance(value, list | tuple | set | frozenset):
        pair = _pair_name(value) if isinstance(value, list | tuple) else None
        for index, item in enumerate(value):
            inner = (*keys, pair) if pair is not None and index == 1 else keys
            yield from _leaves(item, (*path, index), inner, depth + 1)
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _leaves(getattr(value, name), (*path, name), (*keys, name), depth + 1)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            name = field.name
            yield from _leaves(getattr(value, name), (*path, name), (*keys, name), depth + 1)
    else:
        yield Field(str(value), path, keys)  # what the model would see of an unknown object


def _map(value: Any, swap: Swap, path: Path, depth: int) -> Any:
    if value is None or isinstance(value, bool | float):
        return value
    if isinstance(value, str):
        return swap(path, value)
    if isinstance(value, int) or depth > MAX_DEPTH:
        text = str(value)
        changed = swap(path, text)
        return value if changed == text else changed
    if isinstance(value, Mapping):
        return _map_mapping(value, swap, path, depth)
    if isinstance(value, list | tuple):
        items = [_map(item, swap, (*path, i), depth + 1) for i, item in enumerate(value)]
        if hasattr(value, "_fields"):  # a named tuple takes its fields positionally
            return type(value)(*items)
        return type(value)(items)
    if isinstance(value, set | frozenset):
        return type(value)(_map(item, swap, (*path, i), depth + 1) for i, item in enumerate(value))
    if isinstance(value, BaseModel):
        names = type(value).model_fields
        return value.model_copy(
            update={n: _map(getattr(value, n), swap, (*path, n), depth + 1) for n in names}
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        updates = {
            f.name: _map(getattr(value, f.name), swap, (*path, f.name), depth + 1)
            for f in dataclasses.fields(value)
            if f.init
        }
        return dataclasses.replace(value, **updates)
    text = str(value)
    changed = swap(path, text)
    return value if changed == text else changed


def _map_mapping(value: Mapping[Any, Any], swap: Swap, path: Path, depth: int) -> dict[Any, Any]:
    """A dict with keys and values rewritten; a rewritten key never replaces another entry."""
    out: dict[Any, Any] = {}
    for index, (name, item) in enumerate(value.items()):
        new = swap((*path, "key", index), name) if isinstance(name, str) else name
        if new != name:
            base, count = new, 1
            while new in out or (new in value and new != name):
                count += 1
                new = f"{base}#{count}"
        out[new] = _map(item, swap, (*path, "value", index), depth + 1)
    return out
