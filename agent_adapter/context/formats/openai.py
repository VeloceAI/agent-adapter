"""OpenAI messages: chat-completions dicts and Responses input items (OpenAI Agents SDK)."""

from __future__ import annotations

import json
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

NAME = "openai"
CHAT_TEXT = frozenset({"text"})
RESPONSES_TEXT = frozenset({"input_text", "output_text", "text"})
_SYSTEM_ROLES = frozenset({"system", "developer"})
_TOOL_ROLES = frozenset({"tool", "function"})
_LEGACY = "legacy-"  # id prefix for a legacy `function_call` and the `function` reply after it


def is_responses_item(item: Mapping[str, Any]) -> bool:
    """True for a Responses item (it has a `type`); chat dicts have none."""
    return isinstance(item.get("type"), str)


class OpenAIConverter(Converter):
    """Chat-completions dicts, or Responses items when any item has a `type`."""

    name = NAME

    def __init__(self, items: Any) -> None:
        super().__init__(items)
        for index, item in enumerate(self.items):
            if not isinstance(item, Mapping):
                raise not_a_message(NAME, index, item)
        self.responses = any(is_responses_item(item) for item in self.items)
        self.text_types = RESPONSES_TEXT if self.responses else CHAT_TEXT
        self.legacy_ids = _legacy_ids(self.items)

    def groups(self) -> list[Group]:
        groups: list[Group] = []
        index = 0
        while index < len(self.items):
            item = self.items[index]
            if self.responses and _assistant_side(item):
                stop = index + 1
                while stop < len(self.items) and _assistant_side(self.items[stop]):
                    stop += 1
                natives = tuple(range(index, stop))
                groups.append(Group(natives, (self._assistant_group(natives),)))
                index = stop
                continue
            groups.append(Group((index,), (self._message(item, index),)))
            index += 1
        return groups

    def _message(self, item: Mapping[str, Any], index: int) -> Message:
        if _is_output(item):
            content = text_of(item.get("output"), self.text_types)
            return Message(role="tool", tool_call_id=str(item.get("call_id", "")), content=content)
        role = item.get("role")
        text = text_of(item.get("content"), self.text_types)
        if role in _SYSTEM_ROLES:
            return Message(role="system", content=text)
        if role == "user":
            return Message(role="user", content=text)
        if role in _TOOL_ROLES:
            return Message(
                role="tool",
                tool_call_id=str(item.get("tool_call_id") or self.legacy_ids.get(index, "")),
                name=str(item.get("name", "")),
                content=text,
            )
        if role == "assistant":
            calls = tuple(_chat_call(call) for call in item.get("tool_calls") or ())
            legacy = item.get("function_call")
            if not calls and isinstance(legacy, Mapping):
                calls = (_legacy_call(legacy, self.legacy_ids[index]),)
            return Message(role="assistant", content=text, tool_calls=calls)
        raise not_a_message(NAME, index, item)

    def _assistant_group(self, natives: tuple[int, ...]) -> Message:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for n in natives:
            item = self.items[n]
            if item.get("role") == "assistant":
                texts.append(text_of(item.get("content"), self.text_types))
            elif "call_id" in item:
                name = item.get("name") or item.get("type") or ""
                arguments = arguments_of(item.get("arguments")) if "arguments" in item else {}
                calls.append(ToolCall(id=str(item["call_id"]), name=str(name), arguments=arguments))
        text = "\n".join(t for t in texts if t)
        return Message(role="assistant", content=text, tool_calls=tuple(calls))

    def rebuild(self, group: Group, present: list[Present]) -> list[Any]:
        out: list[Any] = []
        for part in present:
            if part.unchanged:
                out.extend(self.items[n] for n in group.natives)
            elif len(group.natives) > 1 or (part.message.role == "assistant" and self.responses):
                out.extend(self._assistant_items(group.natives, part.message.content))
            else:
                out.append(self._with_text(self.items[group.natives[0]], part.message.content))
        return out

    def _with_text(self, item: Mapping[str, Any], text: str) -> dict[str, Any]:
        key = "output" if _is_output(item) else "content"
        default = "input_text" if self.responses else "text"
        return {**item, key: with_text(item.get(key), text, self.text_types, default)}

    def _assistant_items(self, natives: tuple[int, ...], text: str) -> list[Any]:
        """The group's items; the first assistant message holds `text`, later ones lose theirs."""
        out: list[Any] = []
        placed = False
        for n in natives:
            item = self.items[n]
            if item.get("role") != "assistant":
                out.append(item)
            elif not placed:
                content = with_text(item.get("content"), text, self.text_types, "output_text")
                out.append({**item, "content": content})
                placed = True
            elif isinstance(item.get("content"), list):
                rest = [b for b in item["content"] if not _is_text_block(b, self.text_types)]
                if rest:
                    out.append({**item, "content": rest})
        return out

    def render(self, messages: list[Message]) -> list[Any]:
        out: list[Any] = []
        for message in messages:
            out.extend(_responses(message) if self.responses else [_chat(message)])
        return out


def _legacy_ids(items: list[Any]) -> dict[int, str]:
    """Index -> call id for each legacy `function_call` and the `function` messages answering it."""
    ids: dict[int, str] = {}
    current = ""
    for index, item in enumerate(items):
        role = item.get("role")
        if role == "assistant" and isinstance(item.get("function_call"), Mapping):
            current = f"{_LEGACY}{index}"
            ids[index] = current
        elif role == "function" and current:
            ids[index] = current
        else:
            current = ""
    return ids


def _legacy_call(call: Mapping[str, Any], call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id, name=str(call.get("name", "")), arguments=arguments_of(call.get("arguments"))
    )


def _assistant_side(item: Mapping[str, Any]) -> bool:
    """A Responses item that belongs to the model's turn: its messages, calls, reasoning."""
    if item.get("role") == "assistant":
        return True
    return is_responses_item(item) and item.get("type") != "message" and not _is_output(item)


def _is_output(item: Mapping[str, Any]) -> bool:
    kind = item.get("type")
    return isinstance(kind, str) and kind.endswith("_output") and "call_id" in item


def _is_text_block(block: Any, text_types: frozenset[str]) -> bool:
    return isinstance(block, str) or (isinstance(block, dict) and block.get("type") in text_types)


def _chat_call(call: Mapping[str, Any]) -> ToolCall:
    function = call.get("function") or call.get("custom") or {}
    arguments = function.get("arguments", function.get("input"))
    return ToolCall(
        id=str(call.get("id") or ""),
        name=str(function.get("name", "")),
        arguments=arguments_of(arguments),
    )


def _chat(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    if message.role == "assistant" and message.tool_calls:
        calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": _dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
        return {"role": "assistant", "content": message.content or None, "tool_calls": calls}
    return {"role": message.role, "content": message.content}


def _responses(message: Message) -> list[dict[str, Any]]:
    if message.role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": message.content,
            }
        ]
    out: list[dict[str, Any]] = []
    if message.content or not message.tool_calls:
        out.append({"role": message.role, "content": message.content})
    out.extend(
        {
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": _dumps(call.arguments),
        }
        for call in message.tool_calls
    )
    return out


def _dumps(arguments: Mapping[str, Any]) -> str:
    return json.dumps(dict(arguments), separators=(",", ":"), ensure_ascii=False)
