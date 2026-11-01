"""A converter base for messages that hold a role and a list of items (SK, LlamaIndex, MAF)."""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_adapter.context.formats._shared import Converter, Group, Present, not_a_message
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall

ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(frozen=True)
class Text:
    """A text item."""

    text: str


@dataclass(frozen=True)
class Call:
    """A tool-call item."""

    call: ToolCall


@dataclass(frozen=True)
class Result:
    """A tool-result item; each becomes its own tool `Message`."""

    call_id: str
    name: str
    text: str


Item = Text | Call | Result | None  # None: any other item (image, reasoning), kept as is


def as_text(value: Any) -> str:
    """A tool result as text: a str as is, anything else as sorted JSON."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


class ItemsConverter(Converter):
    """Reads and writes natives through a few per-format hooks; the item logic is shared.

    One native becomes one `Message`, or one tool `Message` per tool-result item. A changed
    message keeps every item but its text: the first text item holds the new text.
    """

    @abstractmethod
    def role_of(self, native: Any) -> str | None:
        """ "system", "user", "assistant" or "tool"; None when `native` is not a message."""

    @abstractmethod
    def items_of(self, native: Any) -> Sequence[Any]:
        """The native's items, in order."""

    @abstractmethod
    def read(self, item: Any) -> Item:
        """What one item is."""

    @abstractmethod
    def with_items(self, native: Any, items: list[Any]) -> Any:
        """A copy of `native` holding `items`; every other field stays."""

    @abstractmethod
    def text_item(self, text: str, like: Any | None) -> Any:
        """A text item; `like` is the text item it replaces, if any."""

    @abstractmethod
    def with_result(self, item: Any, text: str) -> Any:
        """A copy of a tool-result item whose result is `text`."""

    @abstractmethod
    def new(self, message: Message) -> Any:
        """A native for a message a compression step created."""

    def tool_result_of(self, native: Any) -> tuple[str, str] | None:
        """(call id, tool name) when the whole native is one tool result, else None."""
        return None

    def extra_calls(self, native: Any) -> tuple[ToolCall, ...]:
        """Tool calls an assistant native keeps outside its items, used when its items have none."""
        return ()

    def keep_verbatim(self, native: Any) -> bool:
        """True for natives no step may change or remove (e.g. approval requests)."""
        return False

    def groups(self) -> list[Group]:
        groups: list[Group] = []
        for index, native in enumerate(self.items):
            role = self.role_of(native)
            if role not in ROLES:
                raise not_a_message(self.name, index, native)
            read = [self.read(item) for item in self.items_of(native)]
            pin = self.keep_verbatim(native)
            results = [(n, r) for n, r in enumerate(read) if isinstance(r, Result)]
            if results:
                messages = tuple(
                    Message(
                        role="tool", tool_call_id=r.call_id, name=r.name, content=r.text, pinned=pin
                    )
                    for _, r in results
                )
                groups.append(Group((index,), messages, tuple(n for n, _ in results)))
                continue
            text = "\n".join(r.text for r in read if isinstance(r, Text) and r.text)
            groups.append(Group((index,), (self._message(native, role, text, read, pin),)))
        return groups

    def _message(self, native: Any, role: str, text: str, read: list[Item], pin: bool) -> Message:
        whole = self.tool_result_of(native)
        if whole is not None or role == "tool":
            call_id, name = whole or ("", "")
            return Message(role="tool", tool_call_id=call_id, name=name, content=text, pinned=pin)
        if role == "assistant":
            calls = tuple(r.call for r in read if isinstance(r, Call)) or self.extra_calls(native)
            return Message(role="assistant", content=text, tool_calls=calls, pinned=pin)
        return Message(role=role, content=text, pinned=pin)

    def rebuild(self, group: Group, present: list[Present]) -> list[Any]:
        native = self.items[group.natives[0]]
        if len(present) == len(group.messages) and all(p.unchanged for p in present):
            return [native]
        items = list(self.items_of(native))
        if not group.layout:
            return [self.with_items(native, self._with_text(items, present[0].message.content))]
        kept = {group.layout[p.part]: p for p in present}
        out: list[Any] = []
        for number, item in enumerate(items):
            if number not in group.layout:
                out.append(item)
            elif number in kept:
                part = kept[number]
                out.append(item if part.unchanged else self.with_result(item, part.message.content))
        return [self.with_items(native, out)]

    def _with_text(self, items: list[Any], text: str) -> list[Any]:
        """The items with the first text item holding `text` and the other text items gone."""
        out: list[Any] = []
        placed = False
        for item in items:
            if not isinstance(self.read(item), Text):
                out.append(item)
            elif not placed:
                out.append(self.text_item(text, item))
                placed = True
        if not placed:
            out.insert(0, self.text_item(text, None))
        return out

    def render(self, messages: list[Message]) -> list[Any]:
        return [self.new(message) for message in messages]
