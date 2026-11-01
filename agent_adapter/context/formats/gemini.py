"""Gemini `contents` (Gemini, Google ADK): dicts or `google.genai.types.Content` objects."""

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
)
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall

NAME = "gemini"
_CALL = ("functionCall", "function_call")
_RESPONSE = ("functionResponse", "function_response")
_SYNTHETIC = "gemini-"  # prefix of call ids made up for calls Gemini sent without one


def is_content(item: Any) -> bool:
    """True for a Gemini content: a dict or object with `parts`."""
    if isinstance(item, Mapping):
        return "parts" in item
    return hasattr(item, "parts") and hasattr(item, "model_dump")


class GeminiConverter(Converter):
    """Function responses become tool messages, matched to calls by id, else by name."""

    name = NAME

    def __init__(self, items: Any) -> None:
        super().__init__(items)
        self.dicts: list[dict[str, Any]] = []
        self.content_type: Any = None  # the Content class, when the input holds objects
        for index, item in enumerate(self.items):
            if not is_content(item):
                raise not_a_message(NAME, index, item)
            if isinstance(item, Mapping):
                self.dicts.append(dict(item))
            else:
                self.content_type = self.content_type or type(item)
                self.dicts.append(item.model_dump(exclude_none=True))

    def groups(self) -> list[Group]:
        groups: list[Group] = []
        pending: list[ToolCall] = []  # calls of the last model turn not yet answered
        for index, content in enumerate(self.dicts):
            parts = list(content.get("parts") or [])
            role = content.get("role")
            if role == "model":
                message = _model(parts, index)
                pending = list(message.tool_calls)
                groups.append(Group((index,), (message,), (tuple(range(len(parts))),)))
                continue
            messages: list[Message] = []
            layout: list[tuple[int, ...]] = []
            rest: list[int] = []
            for position, part in enumerate(parts):
                response = _field(part, _RESPONSE)
                if response is None:
                    rest.append(position)
                    continue
                messages.append(_response(response, pending, f"{_SYNTHETIC}{index}-{position}"))
                layout.append((position,))
            if rest or not messages:
                text = _text([parts[p] for p in rest])
                messages.append(
                    Message(role="system" if role == "system" else "user", content=text)
                )
                layout.append(tuple(rest))
            groups.append(Group((index,), tuple(messages), tuple(layout)))
        return groups

    def rebuild(self, group: Group, present: list[Present]) -> list[Any]:
        content = self.dicts[group.natives[0]]
        parts = list(content.get("parts") or [])
        rebuilt: list[Any] = []
        for part in present:
            mine = [parts[p] for p in group.layout[part.part]]
            if part.unchanged:
                rebuilt.extend(mine)
            elif part.message.role == "tool":
                rebuilt.append(_with_response(mine[0], part.message.content))
            else:
                rebuilt.extend(_with_text(mine, part.message.content))
        if not rebuilt:
            return []
        return [self._native({**content, "parts": rebuilt})]

    def render(self, messages: list[Message]) -> list[Any]:
        out: list[Any] = []
        for message in messages:
            role = "model" if message.role == "assistant" else "user"
            parts: list[dict[str, Any]] = [{"text": message.content}] if message.content else []
            parts.extend(
                {"functionCall": {"name": call.name, "args": dict(call.arguments)}}
                for call in message.tool_calls
            )
            out.append(self._native({"role": role, "parts": parts}))
        return out

    def _native(self, content: dict[str, Any]) -> Any:
        if self.content_type is None:
            return content
        return self.content_type.model_validate(content)


def _field(part: Any, names: tuple[str, str]) -> Any:
    if not isinstance(part, Mapping):
        return None
    for name in names:
        if part.get(name) is not None:
            return part[name]
    return None


def _is_text(part: Any) -> bool:
    return (
        isinstance(part, Mapping) and isinstance(part.get("text"), str) and not part.get("thought")
    )


def _text(parts: list[Any]) -> str:
    return "\n".join(part["text"] for part in parts if _is_text(part) and part["text"])


def _model(parts: list[Any], index: int) -> Message:
    calls = []
    for position, part in enumerate(parts):
        call = _field(part, _CALL)
        if isinstance(call, Mapping):
            calls.append(
                ToolCall(
                    id=str(call.get("id") or f"{_SYNTHETIC}{index}-{position}"),
                    name=str(call.get("name", "")),
                    arguments=arguments_of(call.get("args")),
                )
            )
    return Message(role="assistant", content=_text(parts), tool_calls=tuple(calls))


def _response(response: Any, pending: list[ToolCall], fallback: str) -> Message:
    """A tool message for a function response; the call id comes from the response or the call."""
    response = response if isinstance(response, Mapping) else {}
    name = str(response.get("name", ""))
    call_id = str(response.get("id") or "")
    match = next((c for c in pending if (c.id == call_id if call_id else c.name == name)), None)
    if match is not None:
        pending.remove(match)
        call_id = match.id
    body = response.get("response")
    content = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, default=str)
    return Message(role="tool", tool_call_id=call_id or fallback, name=name, content=content)


def _with_text(parts: list[Any], text: str) -> list[Any]:
    """The first plain text part holds `text`; other plain text parts go."""
    out: list[Any] = []
    placed = False
    for part in parts:
        if not _is_text(part):
            out.append(part)
        elif not placed:
            out.append({**part, "text": text})
            placed = True
    if not placed:
        out.insert(0, {"text": text})
    return out


def _with_response(part: Mapping[str, Any], text: str) -> dict[str, Any]:
    """The function response part with a shortened `response`, kept a JSON object."""
    key = next(name for name in _RESPONSE if part.get(name) is not None)
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    if not isinstance(body, dict):
        body = {"result": text}
    return {**part, key: {**part[key], "response": body}}
