"""OpenAI prompt caching: `prompt_cache_key` / retention, and gpt-5.6+ `prompt_cache_options`."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from agent_adapter.cache.providers._base import (
    EntryText,
    caps_for,
    content_blocks,
    content_field,
    default_cache_key,
    describe,
    locate,
    mark_block,
    text_equals,
)
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.ports import CapabilityLookup

logger = logging.getLogger(__name__)

_CacheKey = str | Callable[[LLMRequest], str] | None

_ROLES = {
    "system": {"system", "developer"},
    "user": {"user"},
    "assistant": {"assistant"},
    "tool": {"tool"},
}
_TEXT_TYPES = frozenset({"text", "input_text", "output_text"})
_BREAKPOINT = {"mode": "explicit"}  # ttl inherits prompt_cache_options.ttl
_MAX_BREAKPOINTS = 4  # OpenAI writes at most four per request


class _KeyMixin:
    key: _CacheKey

    def _cache_key(self, request: LLMRequest) -> str:
        if self.key is None:
            return default_cache_key(request)
        return self.key if isinstance(self.key, str) else self.key(request)


class OpenAIPromptCache(_KeyMixin):
    """Sets `prompt_cache_key` / `prompt_cache_retention` where caps allow; keeps kwargs values."""

    name = "openai_prompt_cache"
    providers = frozenset({"openai"})

    def __init__(
        self,
        *,
        key: _CacheKey = None,
        retention: Literal["in_memory", "24h"] | None = None,
        caps: CapabilityLookup | None = None,
    ) -> None:
        if retention not in (None, "in_memory", "24h"):
            raise ConfigError(f"unknown prompt_cache_retention {retention!r}")
        self.key = key
        self.retention = retention
        self.caps = caps

    def key_fields(self) -> dict[str, Any]:
        """Constructor options (not the computed key)."""
        return {"key": describe(self.key), "retention": self.retention}

    async def apply(
        self, request: LLMRequest, kwargs: dict[str, Any], *, model: str, client: Any
    ) -> dict[str, Any]:
        """Kwargs with the cache parameters the model's capabilities allow."""
        caps = caps_for(self.caps, model)
        if not caps.prompt_cache_key:
            logger.debug("%s: %s does not take prompt_cache_key; unchanged", self.name, model)
            return kwargs
        out = dict(kwargs)
        key = self._cache_key(request)
        if key:
            out.setdefault("prompt_cache_key", key)
        if self.retention == "24h" and not caps.prompt_cache_retention_24h:
            logger.debug("%s: %s has no 24h retention; retention not set", self.name, model)
        elif self.retention is not None:
            out.setdefault("prompt_cache_retention", self.retention)
        return out


class OpenAIPromptCacheV2(_KeyMixin):
    """Sets `prompt_cache_options`, `prompt_cache_key` and, in explicit mode, block breakpoints."""

    name = "openai_prompt_cache_v2"
    providers = frozenset({"openai"})

    def __init__(
        self,
        *,
        mode: Literal["implicit", "explicit"] = "implicit",
        ttl: Literal["30m", "12h"] = "30m",
        key: _CacheKey = None,
        caps: CapabilityLookup | None = None,
    ) -> None:
        if mode not in ("implicit", "explicit"):
            raise ConfigError(f"unknown prompt cache mode {mode!r}")
        if ttl not in ("30m", "12h"):
            raise ConfigError(f"unknown prompt cache ttl {ttl!r}")
        self.mode = mode
        self.ttl = ttl
        self.key = key
        self.caps = caps
        self._warned = False

    def key_fields(self) -> dict[str, Any]:
        """Constructor options (not the computed key)."""
        return {"mode": self.mode, "ttl": self.ttl, "key": describe(self.key)}

    async def apply(
        self, request: LLMRequest, kwargs: dict[str, Any], *, model: str, client: Any
    ) -> dict[str, Any]:
        """Kwargs with `prompt_cache_options` (and breakpoints in explicit mode)."""
        if not caps_for(self.caps, model).prompt_cache_options:
            logger.debug("%s: %s does not take prompt_cache_options; unchanged", self.name, model)
            return kwargs
        out = dict(kwargs)
        key = self._cache_key(request)
        if key:
            out.setdefault("prompt_cache_key", key)
        options: dict[str, str] = {"mode": self.mode, "ttl": self.ttl}
        placed = _mark_breakpoints(request, out) if self.mode == "explicit" else None
        if placed == 0 and "prompt_cache_options" not in out:
            # Explicit mode without a breakpoint turns caching off; fall back to implicit.
            options = {"ttl": self.ttl}
            if not self._warned:
                self._warned = True
                logger.warning(
                    "%s: explicit mode but no cache breakpoint could be placed; using implicit",
                    self.name,
                )
        out.setdefault("prompt_cache_options", options)
        return out


def _mark_breakpoints(request: LLMRequest, out: dict[str, Any]) -> int:
    """Mark each `cache_breakpoint` message's last text block; returns how many were placed."""
    chat = isinstance(out.get("messages"), list)
    field = "messages" if chat else "input"
    entries = out.get(field)
    if not isinstance(entries, list):
        logger.debug("no message list in the kwargs; breakpoints not placed")
        return 0
    entry_ok = _chat_entry_ok if chat else _responses_entry_ok
    text_type: EntryText = _chat_text_type if chat else _responses_text_type
    field_of: EntryText = content_field if chat else _responses_field
    wanted = [m for m in request.messages if m.role in _ROLES]
    placed = list(entries)
    spots: list[tuple[int, int]] = []
    for message, index, block in locate(
        wanted,
        placed,
        entry_ok=entry_ok,
        block_ok=_block_ok,
        text_type=text_type,
        field_of=field_of,
    ):
        if not message.cache_breakpoint:
            continue
        if text_type(placed[index]) == "output_text":
            logger.debug("assistant output blocks cannot carry a cache breakpoint; skipped")
            continue
        spots.append((index, block))
    missing = sum(m.cache_breakpoint for m in wanted) - len(spots)
    if missing:  # e.g. a system prompt sent as Responses `instructions`
        logger.debug("%d cache breakpoint(s) had no content block to carry them", missing)
    free = max(_MAX_BREAKPOINTS - _host_breakpoints(placed, text_type, field_of), 0)
    if len(spots) > free:  # the latest win: each covers the whole prefix before it
        logger.debug("%d breakpoint(s) dropped (limit 4)", len(spots) - free)
        spots = spots[len(spots) - free :]
    for index, block in spots:
        entry = placed[index]
        marker = ("prompt_cache_breakpoint", dict(_BREAKPOINT))
        mark_block(placed, index, block, marker, text_type(entry), field_of(entry))
    out[field] = placed
    return len(spots)


def _host_breakpoints(entries: list[Any], text_type: EntryText, field_of: EntryText) -> int:
    return sum(
        "prompt_cache_breakpoint" in block
        for entry in entries
        if isinstance(entry, dict)
        for block in content_blocks(entry, text_type(entry), field_of(entry))
    )


def _block_ok(message: Message, entry: dict[str, Any], block: dict[str, Any]) -> bool:
    return text_equals(message, block, _TEXT_TYPES)


def _chat_entry_ok(message: Message, entry: dict[str, Any]) -> bool:
    if entry.get("role") not in _ROLES[message.role]:
        return False
    return message.role != "tool" or entry.get("tool_call_id") == message.tool_call_id


def _chat_text_type(entry: dict[str, Any]) -> str:
    return "text"


def _responses_entry_ok(message: Message, entry: dict[str, Any]) -> bool:
    if message.role == "tool":
        return (
            entry.get("type") == "function_call_output"
            and entry.get("call_id") == message.tool_call_id
        )
    return entry.get("type", "message") == "message" and entry.get("role") in _ROLES[message.role]


def _responses_text_type(entry: dict[str, Any]) -> str:
    return "output_text" if entry.get("role") == "assistant" else "input_text"


def _responses_field(entry: dict[str, Any]) -> str:
    return "output" if entry.get("type") == "function_call_output" else "content"
