"""Content (arguments, results, messages, flow state) only for sinks with `wants_content`."""

# Content rides on the event outside its fields: `data`, `to_dict`, `repr` and `asdict` never
# show it, and wrappers (hubs, fan-outs, durable slices) pass it on unchanged.

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from agent_adapter.core.errors import log_failure
from agent_adapter.core.ports import EventSink
from agent_adapter.core.types import Event
from agent_adapter.tracing.events import emit, members

logger = logging.getLogger(__name__)

WANTS_CONTENT = "wants_content"
WANTS_CONFIRM_CONTENT = "wants_confirm_content"
Content = Callable[[], Mapping[str, Any]]
_CONTENT = "_aa_content"
_CONFIRM = "_aa_confirm"


class ContentEvent(Event):
    """An event carrying captured content beside (never inside) its data."""


def _flag(sink: object, name: str) -> bool:
    try:
        return getattr(sink, name, False) is True
    except Exception as exc:  # tracing never breaks the call, not even a sink's property
        log_failure(logger, logging.WARNING, exc, "sink %s: %s not read", type(sink).__name__, name)
        return False


def wants_content(sink: EventSink | None, *, confirm: bool = False) -> bool:
    """True when a sink under `sink` asks for content (with `confirm`: for confirm tools too)."""
    return any(
        _flag(m, WANTS_CONTENT) and (not confirm or _flag(m, WANTS_CONFIRM_CONTENT))
        for m in members(sink)
    )


def content_data(event: Event, sink: object) -> Mapping[str, Any]:
    """The event's data plus the content `sink` may see (none unless it asks)."""
    content = getattr(event, _CONTENT, None) if isinstance(event, ContentEvent) else None
    if not content or not _flag(sink, WANTS_CONTENT):
        return event.data
    if getattr(event, _CONFIRM, False) and not _flag(sink, WANTS_CONFIRM_CONTENT):
        return event.data
    return {**event.data, **content}


def emit_with_content(
    sink: EventSink | None, event: Event, content: Content, *, confirm: bool = False
) -> None:
    """Send `event`; `content()` runs only when a sink asks; `confirm` marks a confirm tool's."""
    if sink is None:
        return
    if not wants_content(sink, confirm=confirm):
        emit(sink, event)
        return
    try:
        extra = dict(content())
    except Exception as exc:
        log_failure(logger, logging.WARNING, exc, "content for %s not recorded", event.type)
        emit(sink, event)
        return
    carried = ContentEvent(
        event.type, event.run_id, event.call_id, event.parent_id, event.ts, event.data
    )
    object.__setattr__(carried, _CONTENT, extra)
    object.__setattr__(carried, _CONFIRM, confirm)
    emit(sink, carried)


__all__ = [
    "WANTS_CONFIRM_CONTENT",
    "WANTS_CONTENT",
    "Content",
    "ContentEvent",
    "content_data",
    "emit_with_content",
    "wants_content",
]
