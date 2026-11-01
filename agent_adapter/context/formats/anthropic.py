"""Anthropic Messages API dicts (Anthropic, Claude Agent SDK): text, tool_use and tool_result."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_adapter.context.formats._shared import (
    Converter,
    Group,
    Present,
    arguments_of,
    not_a_message,
    text_of,
    with_text,
)
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall

NAME = "anthropic"
TEXT = frozenset({"text"})
_WHOLE = "whole"  # layout of a part that is the message's plain-string content


class AnthropicConverter(Converter):
    """A user message's tool_result blocks become tool messages; the rest one user message."""

    name = NAME

    def __init__(self, items: Any) -> None:
        super().__init__(items)
        for index, item in enumerate(self.items):
            if not isinstance(item, Mapping) or item.get("role") not in (
                "user",
                "assistant",
                "system",
            ):
                raise not_a_message(NAME, index, item)

    def groups(self) -> list[Group]:
        return [self._group(index, item) for index, item in enumerate(self.items)]

    def _group(self, index: int, item: Mapping[str, Any]) -> Group:
        role = item["role"]
        content = item.get("content")
        if role == "assistant":
            return Group((index,), (_assistant(content),))
        if not isinstance(content, list):
            return Group((index,), (Message(role=role, content=text_of(content, TEXT)),), (_WHOLE,))
        messages: list[Message] = []
        layout: list[Any] = []
        rest: list[int] = []
        for position, block in enumerate(content):
            if isinstance(block, Mapping) and block.get("type") == "tool_result":
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=str(block.get("tool_use_id", "")),
                        content=text_of(block.get("content"), TEXT),
                    )
                )
                layout.append((position,))
            else:
                rest.append(position)
        if rest or not messages:
            text = text_of([content[p] for p in rest], TEXT)
            messages.append(Message(role="system" if role == "system" else "user", content=text))
            layout.append(tuple(rest))
        return Group((index,), tuple(messages), tuple(layout))

    def rebuild(self, group: Group, present: list[Present]) -> list[Any]:
        item = self.items[group.natives[0]]
        content = item.get("content")
        if item["role"] == "assistant" or group.layout == (_WHOLE,):
            text = present[0].message.content
            return [
                item
                if present[0].unchanged
                else {**item, "content": with_text(content, text, TEXT)}
            ]
        blocks: list[Any] = []
        for part in present:
            mine = [content[p] for p in group.layout[part.part]]
            if part.unchanged:
                blocks.extend(mine)
            elif part.message.role == "tool":
                block = mine[0]
                text = part.message.content
                blocks.append({**block, "content": with_text(block.get("content"), text, TEXT)})
            else:
                blocks.extend(with_text(mine, part.message.content, TEXT))
        return [{**item, "content": blocks}] if blocks else []

    def render(self, messages: list[Message]) -> list[Any]:
        out: list[Any] = []
        results: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
                continue
            if results:
                out.append({"role": "user", "content": results})
                results = []
            out.append(_render(message))
        if results:
            out.append({"role": "user", "content": results})
        return out


def _assistant(content: Any) -> Message:
    text = text_of(content, TEXT)
    calls: list[ToolCall] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        arguments=arguments_of(block.get("input")),
                    )
                )
    return Message(role="assistant", content=text, tool_calls=tuple(calls))


def _render(message: Message) -> dict[str, Any]:
    if message.role != "assistant" or not message.tool_calls:
        return {"role": message.role, "content": message.content}
    blocks: list[dict[str, Any]] = (
        [{"type": "text", "text": message.content}] if message.content else []
    )
    blocks.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": dict(call.arguments)}
        for call in message.tool_calls
    )
    return {"role": "assistant", "content": blocks}
