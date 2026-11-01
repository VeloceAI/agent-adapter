"""Anthropic prompt caching: `cache_control` on the system prompt, tools and breakpoints."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Literal

from agent_adapter.cache.providers._base import (
    caps_for,
    content_blocks,
    locate,
    mark_block,
    text_equals,
)
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.ports import CapabilityLookup

logger = logging.getLogger(__name__)

_Target = Literal["system", "tools", "breakpoints"]
_MAX_BREAKPOINTS = 4  # Anthropic rejects requests with more cache_control blocks
_TEXT = frozenset({"text"})
_JOINERS = ("", "\n", "\n\n")  # how ports may join several text blocks into one reply
# A marker position: ("system" | "tools" | "messages", entry index, block index).
_Spot = tuple[str, int, int]


class AnthropicPromptCache:
    """Adds `cache_control` to the last system block, last tool and breakpoint messages (max 4)."""

    name = "anthropic_prompt_cache"
    providers = frozenset({"anthropic"})

    def __init__(
        self,
        *,
        ttl: Literal["5m", "1h"] = "5m",
        on: Iterable[_Target] = ("system", "tools"),
        caps: CapabilityLookup | None = None,
    ) -> None:
        if ttl not in ("5m", "1h"):
            raise ConfigError(f"unknown cache_control ttl {ttl!r}")
        targets = tuple(dict.fromkeys(on))
        unknown = set(targets) - {"system", "tools", "breakpoints"}
        if unknown:
            raise ConfigError(f"unknown cache_control targets {sorted(unknown)}")
        self.ttl = ttl
        self.on: tuple[_Target, ...] = targets
        self.caps = caps

    def key_fields(self) -> dict[str, Any]:
        """Constructor options."""
        return {"ttl": self.ttl, "on": list(self.on)}

    async def apply(
        self, request: LLMRequest, kwargs: dict[str, Any], *, model: str, client: Any
    ) -> dict[str, Any]:
        """Kwargs with `cache_control` markers, at most four in total."""
        if not caps_for(self.caps, model).cache_control:
            logger.debug("%s: %s does not take cache_control; unchanged", self.name, model)
            return kwargs
        out = dict(kwargs)
        lists = {
            "tools": _as_list(out.get("tools"), None),
            "system": _as_list(out.get("system"), "text"),
            "messages": _as_list(out.get("messages"), None),
        }
        existing = _existing_markers(lists)
        if any(marker.get("ttl", "5m") != self.ttl for marker in existing):
            # Longer TTLs must precede shorter ones; mixing with host markers could break that.
            logger.debug("%s: host markers use another ttl; unchanged", self.name)
            return kwargs
        spots = self._spots(request, lists)
        free = _MAX_BREAKPOINTS - len(existing)
        if len(spots) > free:
            logger.debug("%s: %d marker(s) dropped (limit 4)", self.name, len(spots) - max(free, 0))
            spots = spots[len(spots) - free :] if free > 0 else []
        marker = {"type": "ephemeral", "ttl": self.ttl}
        for where, index, block in spots:
            items = lists[where]
            if where == "messages":
                mark_block(items, index, block, ("cache_control", dict(marker)), "text")
            else:
                items[index] = {**items[index], "cache_control": dict(marker)}
        for where in {where for where, _, _ in spots}:
            out[where] = lists[where]
        return out

    def _spots(self, request: LLMRequest, lists: dict[str, list[Any]]) -> list[_Spot]:
        # In prefix order: tools, then system, then messages.
        spots: list[_Spot] = []
        for where in ("tools", "system"):
            items = lists[where]
            if where in self.on and items and _unmarked(items[-1]):
                spots.append((where, len(items) - 1, 0))
        if "breakpoints" in self.on:
            wanted = [m for m in request.messages if m.role != "system"]
            for message, index, block in locate(
                wanted,
                lists["messages"],
                entry_ok=_entry_ok,
                block_ok=_block_ok,
                text_type=lambda entry: "text",
            ):
                blocks = content_blocks(lists["messages"][index], "text")
                if message.cache_breakpoint and _unmarked(blocks[block]):
                    spots.append(("messages", index, block))
        return spots


def _as_list(value: Any, text_type: str | None) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str) and value and text_type is not None:
        return [{"type": text_type, "text": value}]
    return []


def _unmarked(item: Any) -> bool:
    return isinstance(item, dict) and not item.get("cache_control")


def _existing_markers(lists: dict[str, list[Any]]) -> list[dict[str, Any]]:
    items = [*lists["tools"], *lists["system"]]
    for entry in lists["messages"]:
        if isinstance(entry, dict):
            items += content_blocks(entry, "text")
    markers = [item.get("cache_control") for item in items if isinstance(item, dict)]
    return [marker if isinstance(marker, dict) else {} for marker in markers if marker]


def _entry_ok(message: Message, entry: dict[str, Any]) -> bool:
    role = "user" if message.role == "tool" else message.role
    return entry.get("role") == role


def _block_ok(message: Message, entry: dict[str, Any], block: dict[str, Any]) -> bool:
    if message.role == "tool":
        return block.get("type") == "tool_result" and block.get("tool_use_id") == (
            message.tool_call_id
        )
    if message.tool_calls:  # the message ends with its last tool_use block
        return block.get("type") == "tool_use" and block.get("id") == message.tool_calls[-1].id
    if text_equals(message, block, _TEXT):
        return True
    return message.role == "assistant" and _ends_split_text(message, entry, block)


def _ends_split_text(message: Message, entry: dict[str, Any], block: dict[str, Any]) -> bool:
    """True when `block` is the last of several text blocks that together hold the reply."""
    # Replies with citations or thinking arrive as several text blocks; ports join them.
    texts = [b for b in content_blocks(entry, "text") if b.get("type") == "text"]
    if len(texts) < 2 or block is not texts[-1] or not message.content:
        return False
    parts = [str(b.get("text", "")) for b in texts]
    return message.content in {joiner.join(parts) for joiner in _JOINERS}
