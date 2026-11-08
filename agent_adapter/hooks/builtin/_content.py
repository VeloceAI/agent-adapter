"""Captured content made safe: JSON, bounded, secret-named keys and secret or PII text masked."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from agent_adapter.core.messages import Message
from agent_adapter.core.types import to_jsonable
from agent_adapter.hooks.builtin.guards.pii import PII_KINDS, find_pii, redact_pii
from agent_adapter.hooks.builtin.logging import REDACTED, redact_secrets

if TYPE_CHECKING:
    from agent_adapter.core.tool import ToolSpec

MAX_TEXT = 2_000  # characters kept of one string, marked like the run recorder marks a cut
MAX_ITEMS = 200  # items kept of one list or mapping
MAX_DEPTH = 12
MAX_WITHHELD = 10_000  # confirm-tool calls remembered, so their tool messages stay blank
WITHHELD = "[withheld: confirm tool]"
MASKS = frozenset({REDACTED, *(f"[{kind.upper()}]" for kind in PII_KINDS)})

_withheld: OrderedDict[tuple[str, str], None] = OrderedDict()
_lock = threading.Lock()


def safe_value(value: Any) -> Any:
    """`value` as bounded JSON; secret-named entries, and secrets or PII inside text, masked."""
    return _scan(redact_secrets(to_jsonable(value)), 0)


def _scan(value: Any, depth: int) -> Any:
    if depth > MAX_DEPTH:
        return "[too deep]"
    if isinstance(value, str):
        text = value if len(value) <= MAX_TEXT else value[:MAX_TEXT]
        found = find_pii(text)
        text = redact_pii(text, found) if found else text
        return text if len(value) <= MAX_TEXT else f"{text}... [+{len(value) - MAX_TEXT} chars]"
    if isinstance(value, Mapping):
        items = list(value.items())
        out = {str(k): _scan(v, depth + 1) for k, v in items[:MAX_ITEMS]}
        if len(items) > MAX_ITEMS:
            out["..."] = f"+{len(items) - MAX_ITEMS} keys"
        return out
    if isinstance(value, list):
        kept = [_scan(item, depth + 1) for item in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            kept.append(f"... +{len(value) - MAX_ITEMS} items")
        return kept
    return value


def shows_content(spec: ToolSpec | None) -> bool:
    """False for a `confirm` tool (any rule) or an unknown one: its content is withheld."""
    return spec is not None and spec.confirm is False


def tool_arguments(spec: ToolSpec | None, arguments: Any) -> dict[str, Any]:
    """`{"arguments": ...}` for a content sink; nothing for a `confirm` (or unknown) tool."""
    return {"arguments": safe_value(arguments)} if shows_content(spec) else {}


def withhold(run_id: str, call_id: str) -> None:
    """Remember a confirm-tool call, so its tool message is never recorded as content."""
    with _lock:
        _withheld[(run_id, call_id)] = None
        _withheld.move_to_end((run_id, call_id))
        while len(_withheld) > MAX_WITHHELD:
            _withheld.popitem(last=False)


def withheld(run_id: str, call_id: str) -> bool:
    """True for a remembered confirm-tool call."""
    with _lock:
        return (run_id, call_id) in _withheld


def message_dicts(messages: Iterable[Message], run_id: str = "") -> list[Any]:
    """Messages as role and text; tool calls by id and name only; confirm-tool replies blank."""
    found: list[dict[str, Any]] = []
    for message in messages:
        text = message.content
        if message.role == "tool" and withheld(run_id, message.tool_call_id):
            text = WITHHELD
        item: dict[str, Any] = {"role": message.role, "content": text}
        if message.tool_calls:
            item["tool_calls"] = [{"id": c.id, "name": c.name} for c in message.tool_calls]
        if message.name:
            item["name"] = message.name
        found.append(item)
    return list(safe_value(found))


__all__ = [
    "MASKS",
    "WITHHELD",
    "message_dicts",
    "safe_value",
    "shows_content",
    "tool_arguments",
    "withheld",
    "withhold",
]
