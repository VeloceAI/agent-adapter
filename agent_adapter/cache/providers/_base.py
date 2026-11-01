"""Helpers shared by the prompt-cache plug-ins: capability lookup and wire-block matching."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agent_adapter.cache.keys import digest
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.ports import CapabilityLookup
from agent_adapter.core.types import ModelCaps
from agent_adapter.cost.prices import Prices

# (message, wire entry, content block) -> does this block end that message?
BlockMatch = Callable[[Message, dict[str, Any], dict[str, Any]], bool]
# (message, wire entry) -> may this entry hold that message?
EntryMatch = Callable[[Message, dict[str, Any]], bool]
# wire entry -> its text-block type, or the key that holds its content
EntryText = Callable[[dict[str, Any]], str]


def caps_for(lookup: CapabilityLookup | None, model: str) -> ModelCaps:
    """Capabilities of `model`; no lookup reads the packaged price table (unknown model: none)."""
    return Prices.default().caps(model) if lookup is None else lookup(model)


def system_text(request: LLMRequest) -> str:
    """The request's system messages, joined by blank lines."""
    return "\n\n".join(m.content for m in request.messages if m.role == "system" and m.content)


def default_cache_key(request: LLMRequest) -> str:
    """Digest of the system text; "" when the request has none."""
    text = system_text(request)
    return digest(text) if text else ""


def describe(value: Any) -> Any:
    """JSON-safe description of a constructor option (callables by qualified name)."""
    if callable(value):
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"callable:{getattr(value, '__module__', '')}.{name}"
    return value


def content_field(entry: dict[str, Any]) -> str:
    """The key holding an entry's content blocks (the default for every wire format)."""
    return "content"


def content_blocks(
    entry: dict[str, Any], text_type: str, field: str = "content"
) -> list[dict[str, Any]]:
    """The entry's `field` as a list of blocks (a string becomes one text block)."""
    content = entry.get(field)
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def locate(
    messages: Sequence[Message],
    entries: Sequence[Any],
    *,
    entry_ok: EntryMatch,
    block_ok: BlockMatch,
    text_type: EntryText,
    field_of: EntryText = content_field,
) -> list[tuple[Message, int, int]]:
    """In-order (message, entry, block) wire positions, matched by content after the last hit."""
    # Ports may merge neighbouring messages into one entry, hence block-level positions.
    found: list[tuple[Message, int, int]] = []
    position = (0, 0)
    for message in messages:
        hit = _find(message, entries, position, entry_ok, block_ok, text_type, field_of)
        if hit is not None:
            found.append((message, *hit))
            position = (hit[0], hit[1] + 1)
    return found


def _find(
    message: Message,
    entries: Sequence[Any],
    position: tuple[int, int],
    entry_ok: EntryMatch,
    block_ok: BlockMatch,
    text_type: EntryText,
    field_of: EntryText,
) -> tuple[int, int] | None:
    entry_pos, block_pos = position
    for index in range(entry_pos, len(entries)):
        entry = entries[index]
        if not isinstance(entry, dict) or not entry_ok(message, entry):
            continue
        blocks = content_blocks(entry, text_type(entry), field_of(entry))
        start = block_pos if index == entry_pos else 0
        for block in range(start, len(blocks)):
            if block_ok(message, entry, blocks[block]):
                return index, block
    return None


def mark_block(
    entries: list[Any],
    index: int,
    block: int,
    marker: tuple[str, Any],
    text_type: str,
    field: str = "content",
) -> None:
    """Set `marker` (key, value) on one block, copying the entry and its blocks."""
    entry = dict(entries[index])
    blocks = [dict(b) for b in content_blocks(entry, text_type, field)]
    key, value = marker
    blocks[block][key] = value
    entry[field] = blocks
    entries[index] = entry


def text_equals(message: Message, block: dict[str, Any], types: frozenset[str]) -> bool:
    """True when `block` is a text block holding exactly the message content."""
    return (
        bool(message.content)
        and block.get("type") in types
        and block.get("text") == message.content
    )
