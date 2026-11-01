"""The converter interface and `Conversion`: native messages to `Message` and back, losslessly."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, PrivateAttr

from agent_adapter.context.conversation import has_approval_marker
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message

T = TypeVar("T")
Pinned = Callable[[Any], bool]  # native message -> keep it verbatim
# Native blocks replayed to the provider but absent from `Message.content`: thinking,
# reasoning and Gemini thought parts. They are counted, so token estimates stay honest.
_HIDDEN_TYPES = frozenset({"thinking", "redacted_thinking", "reasoning"})
_HIDDEN_KEYS = frozenset({"thought_signature", "thoughtSignature", "encrypted_content"})
_MAX_DEPTH = 6


class TaggedMessage(Message):
    """A `Message` that remembers the native it came from, also in the copies steps make.

    Equal to a plain `Message` with the same fields; the origin is never sent or dumped.
    """

    _origin: tuple[object, int, int] | None = PrivateAttr(default=None)  # (token, group, part)
    _hidden: int = PrivateAttr(default=0)  # chars of replayed native blocks not in content

    @property
    def hidden_chars(self) -> int:
        """Characters of native blocks (thinking, reasoning) sent with this message."""
        return self._hidden

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Message):
            return NotImplemented
        return self.__dict__ == other.__dict__

    __hash__ = Message.__hash__


@dataclass(frozen=True)
class Group:
    """Native items that become one or more `Message`s together (e.g. tool results)."""

    natives: tuple[int, ...]  # indices into the input list, in order
    messages: tuple[Message, ...]  # untagged; one per part
    layout: tuple[Any, ...] = field(default=())  # converter data per part (e.g. block indices)


@dataclass(frozen=True)
class Present:
    """A part of a group that is still in the compressed list."""

    part: int
    message: Message
    unchanged: bool  # the very message the conversion made


class Converter(ABC):
    """One message format: `groups` reads the natives, `rebuild` and `render` write them."""

    name: str = ""

    def __init__(self, items: Sequence[Any]) -> None:
        self.items = list(items)

    @abstractmethod
    def groups(self) -> list[Group]:
        """The input as groups, in order; every native index appears once.

        Raises:
            ConfigError: an item is not a message of this format.
        """

    @abstractmethod
    def rebuild(self, group: Group, present: list[Present]) -> list[Any]:
        """Natives for the parts of `group` still present, some with changed text."""

    @abstractmethod
    def render(self, messages: list[Message]) -> list[Any]:
        """Natives for messages a compression step created (summaries)."""

    def finish(self, natives: list[Any]) -> list[Any]:
        """The rebuilt list, last; a format whose natives carry an order key fixes it here."""
        return natives

    def carries_approval(self, native: Any) -> bool:
        """True when `native` holds approval state; its group is then kept verbatim."""
        return has_approval_marker(native)


class Conversion(Generic[T]):
    """Native messages as `messages`; `back()` turns a compressed list into natives again.

    Messages the compression kept as they were come back as the very same native objects;
    changed ones keep every field of their native except the changed text.
    """

    def __init__(self, converter: Converter, pinned: Pinned | None = None) -> None:
        if pinned is not None and not callable(pinned):
            raise ConfigError("pinned must be a callable(native message) -> bool")
        self.format = converter.name
        self._token = object()  # tells this conversion's messages from another's
        self.items: list[T] = list(converter.items)
        self._converter = converter
        self.groups: list[Group] = []
        self.messages: list[Message] = []
        self._at: dict[int, tuple[int, int]] = {}  # id(message) -> (group, part)
        for number, group in enumerate(converter.groups()):
            pin = any(converter.carries_approval(self.items[n]) for n in group.natives) or (
                pinned is not None and any(pinned(self.items[n]) for n in group.natives)
            )
            tagged = tuple(
                self._tagged(message, number, part, pin)
                for part, message in enumerate(group.messages)
            )
            if tagged:
                tagged[0]._hidden = sum(hidden_chars(self.items[n]) for n in group.natives)
            self.groups.append(Group(group.natives, tagged, group.layout))
            for part, message in enumerate(tagged):
                self._at[id(message)] = (number, part)
            self.messages.extend(tagged)

    def back(self, messages: Sequence[Message]) -> list[T]:
        """Natives for `messages` (this conversion's messages, compressed or not)."""
        out: list[Any] = []
        index = 0
        while index < len(messages):
            whole = self._whole_group(messages, index)
            if whole is not None:
                out.extend(self.items[n] for n in self.groups[whole].natives)
                index += len(self.groups[whole].messages)
                continue
            stop = index + 1
            while stop < len(messages) and self._whole_group(messages, stop) is None:
                stop += 1
            out.extend(self._run(list(messages[index:stop])))
            index = stop
        return self._converter.finish(out)

    def _whole_group(self, messages: Sequence[Message], index: int) -> int | None:
        """The group starting at `index` when all its parts follow unchanged, else None."""
        where = self._at.get(id(messages[index]))
        if where is None or where[1] != 0:
            return None
        group = self.groups[where[0]]
        span = messages[index : index + len(group.messages)]
        same = len(span) == len(group.messages)
        if same and all(a is b for a, b in zip(span, group.messages, strict=True)):
            return where[0]
        return None

    def _run(self, run: list[Message]) -> list[Any]:
        """Natives for messages not forming whole unchanged groups."""
        out: list[Any] = []
        fresh: list[Message] = []
        index = 0
        while index < len(run):
            tag = self._tag(run[index])
            if tag is None:
                fresh.append(run[index])
                index += 1
                continue
            if fresh:
                out.extend(self._converter.render(fresh))
                fresh = []
            number = tag[0]
            group = self.groups[number]
            present: list[Present] = []
            while index < len(run):
                tag = self._tag(run[index])
                if tag is None or tag[0] != number or (present and tag[1] <= present[-1].part):
                    break
                message = run[index]
                present.append(Present(tag[1], message, message is group.messages[tag[1]]))
                index += 1
            out.extend(self._converter.rebuild(group, present))
        if fresh:
            out.extend(self._converter.render(fresh))
        return out

    def _tagged(self, message: Message, group: int, part: int, pin: bool) -> TaggedMessage:
        tagged = TaggedMessage(**{**message.__dict__, "pinned": message.pinned or pin})
        tagged._origin = (self._token, group, part)
        return tagged

    def _tag(self, message: Message) -> tuple[int, int] | None:
        """(group, part) of a message from this conversion, also after a step changed it."""
        where = self._at.get(id(message))
        if where is not None:
            return where
        origin = message._origin if isinstance(message, TaggedMessage) else None
        if origin is None or origin[0] is not self._token:
            return None
        if _kind(message) != _kind(self.groups[origin[1]].messages[origin[2]]):
            return None  # a step made it another kind of message: it is rendered as new
        return origin[1], origin[2]


def _kind(message: Message) -> tuple[Any, ...]:
    """What a step may not change and still have its native rebuilt: only the text may differ."""
    return message.role, message.tool_call_id, message.name, message.tool_calls


def hidden_chars(native: Any, depth: int = 0) -> int:
    """Characters of the thinking, reasoning and thought blocks inside one native message."""
    if depth > _MAX_DEPTH:
        return 0
    if isinstance(native, BaseModel):
        native = native.model_dump(exclude_none=True)
    if isinstance(native, Mapping):
        hidden = native.get("type") in _HIDDEN_TYPES or native.get("thought") is True
        if hidden or any(key in native for key in _HIDDEN_KEYS):
            return len(json.dumps(native, ensure_ascii=False, default=str))
        return sum(hidden_chars(value, depth + 1) for value in native.values())
    if isinstance(native, list | tuple):
        return sum(hidden_chars(value, depth + 1) for value in native)
    return 0


def text_of(value: Any, text_types: frozenset[str]) -> str:
    """The text of a str, or of the text blocks of a list (others are skipped)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = [
            block if isinstance(block, str) else _block_text(block, text_types) for block in value
        ]
        return "\n".join(text for text in texts if text)
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _block_text(block: Any, text_types: frozenset[str]) -> str:
    if isinstance(block, dict) and block.get("type") in text_types:
        text = block.get("text")
        return text if isinstance(text, str) else ""
    return ""


def with_text(value: Any, text: str, text_types: frozenset[str], default_type: str = "text") -> Any:
    """`value` with its text replaced: a str becomes `text`; in a block list the first text
    block holds `text` and the other text blocks go; other blocks stay."""
    if not isinstance(value, list):
        return text
    out: list[Any] = []
    placed = False
    for block in value:
        is_text = isinstance(block, str) or (
            isinstance(block, dict) and block.get("type") in text_types
        )
        if not is_text:
            out.append(block)
        elif not placed:
            out.append(text if isinstance(block, str) else {**block, "text": text})
            placed = True
    if not placed:
        out.insert(0, {"type": default_type, "text": text})
    return out


def arguments_of(value: Any) -> dict[str, Any]:
    """Tool-call arguments as a dict: a dict as is, a JSON object string parsed, else wrapped."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except ValueError:
            return {"_arguments": value}
        return parsed if isinstance(parsed, dict) else {"_arguments": value}
    return {} if value is None else {"_arguments": value}


def not_a_message(fmt: str, index: int, item: Any) -> ConfigError:
    """The error for an item that is not a message of `fmt`."""
    return ConfigError(f"item {index} is not a {fmt} message: {type(item).__name__}")
