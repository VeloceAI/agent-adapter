"""What a recorded run's events hold as input and output, when content was captured."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from agent_adapter.hooks.builtin._content import MASKS

_CLIPPED = re.compile(r"\.\.\. \[\+\d+ chars\]\Z")  # the recorder's mark on a cut string
_MORE_ITEMS = re.compile(r"\A\.\.\. \+\d+ items\Z")
_TOO_DEEP = "[too deep]"

Events = Iterable[Mapping[str, Any]]


def clipped(value: Any) -> bool:
    """True when the recorder cut `value` (a long string, a long list or mapping, deep nesting)."""
    if isinstance(value, str):
        return value == _TOO_DEEP or _CLIPPED.search(value) is not None
    if isinstance(value, Mapping):
        return "..." in value or any(clipped(item) for item in value.values())
    if isinstance(value, list):
        if value and isinstance(value[-1], str) and _MORE_ITEMS.match(value[-1]):
            return True
        return any(clipped(item) for item in value)
    return False


def message_text(message: Any) -> str | None:
    """The text of one message: `content` text, or the text parts (OpenTelemetry GenAI)."""
    if isinstance(message, str):
        return message
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts = message.get("parts", content)
    if not isinstance(parts, list):
        return None
    texts = [
        text
        for part in parts
        if isinstance(part, Mapping) and part.get("type", "text") == "text"
        for text in (part.get("content", part.get("text")),)
        if isinstance(text, str)
    ]
    return "\n".join(texts) if texts else None


def model_calls(events: Events) -> list[dict[str, Any]]:
    """The run's own (top-level) model calls in order: stage, input and output messages."""
    calls: dict[str, dict[str, Any]] = {}
    for event in events:
        kind = event.get("type")
        if kind not in ("llm.started", "llm.finished"):
            continue
        data = _data(event)
        call = calls.setdefault(
            str(event.get("call_id", "")),
            {"stage": data.get("stage"), "parent": str(event.get("parent_id") or "")},
        )
        call["nested"] = call.get("nested") or data.get("nested") is True
        for key in ("input_messages", "output_messages"):
            if key in data:
                call[key] = data[key]
    return [c for c in calls.values() if not c["nested"] and c["parent"] not in calls.keys() - {""}]


def first_user_text(events: Events) -> str | None:
    """The current question: the last user message of the run's first top-level model call."""
    for call in model_calls(events):
        messages = call.get("input_messages")
        if messages is None:
            continue
        if isinstance(messages, str):
            return messages or None
        texts = [
            message_text(m)
            for m in (messages if isinstance(messages, list) else ())
            if isinstance(m, Mapping) and m.get("role") == "user"
        ]
        found = [text for text in texts if text]
        return found[-1] if found else None
    return None


def last_answer_text(events: Events) -> str | None:
    """The last assistant text of the top-level calls at the first call's stage."""
    calls = model_calls(events)
    stage = next((c["stage"] for c in calls if "input_messages" in c), None)
    answer = None
    for call in calls:
        messages = call.get("output_messages")
        if messages is None or call["stage"] != stage:
            continue
        if isinstance(messages, str):
            answer = messages or answer
            continue
        for message in messages if isinstance(messages, list) else ():
            role = message.get("role", "assistant") if isinstance(message, Mapping) else ""
            text = message_text(message) if role == "assistant" else None
            if text:
                answer = text
    return answer


def masked(value: Any) -> bool:
    """True when the recorder masked part of `value` (a secret-named key, secret or PII text)."""
    if isinstance(value, str):
        return value in MASKS or any(mark in value for mark in MASKS if mark.startswith("["))
    if isinstance(value, Mapping):
        return any(masked(item) for item in value.values())
    if isinstance(value, list):
        return any(masked(item) for item in value)
    return False


def flow_io(events: Events) -> tuple[Any, Any] | None:
    """(input, output) of the run's first recorded flow; output is None until it finished done."""
    name, given = None, None
    for event in events:
        data = _data(event)
        if name is None and event.get("type") == "flow.started" and "input" in data:
            name, given = data.get("flow"), data["input"]
        elif name is not None and event.get("type") == "flow.finished" and data.get("flow") == name:
            return given, data.get("output")
    return None if name is None else (given, None)


def first_tool_call(events: Events) -> tuple[str, str, Any] | None:
    """(call id, tool, arguments) of the first tool call whose arguments were recorded."""
    for event in events:
        data = _data(event)
        if event.get("type") in ("tool.started", "tool.finished") and "arguments" in data:
            return str(event.get("call_id", "")), str(data.get("tool", "")), data["arguments"]
    return None


def tool_result(events: Events, call_id: str) -> tuple[bool, Any]:
    """(found, result) of the call's recorded `tool.finished` result."""
    for event in events:
        data = _data(event)
        if (
            event.get("type") == "tool.finished"
            and str(event.get("call_id", "")) == call_id
            and "result" in data
        ):
            return True, data["result"]
    return False, None


def names(events: Events, kind: str, key: str) -> list[str]:
    """The distinct `data[key]` values of events of type `kind`, in order."""
    found: dict[str, None] = {}
    for event in events:
        value = _data(event).get(key)
        if event.get("type") == kind and isinstance(value, str) and value:
            found.setdefault(value, None)
    return list(found)


def _data(event: Mapping[str, Any]) -> Mapping[str, Any]:
    data = event.get("data")
    return data if isinstance(data, Mapping) else {}
