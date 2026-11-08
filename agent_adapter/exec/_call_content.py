"""A tool call's captured content: arguments once the guards passed, the guarded result."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.ports import EventSink
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import Event
from agent_adapter.hooks.builtin._content import safe_value, shows_content, withhold
from agent_adapter.tracing._content import emit_with_content, wants_content

NOT_PASSED = object()  # the arguments never passed the before-tool guards


def emit_finished(
    sink: EventSink | None, event: Event, spec: ToolSpec | None, arguments: Any, value: Any
) -> None:
    """`tool.finished` with content for the sinks that ask; a confirm tool's only on opt-in."""
    if not shows_content(spec):
        if wants_content(sink):
            withhold(event.run_id, event.call_id)  # its tool message stays blank later
        emit_with_content(sink, event, lambda: {"result": safe_value(value)}, confirm=True)
        return

    def build() -> dict[str, Any]:
        found = {"result": safe_value(value)}
        if arguments is not NOT_PASSED:
            found["arguments"] = safe_value(arguments)
        return found

    emit_with_content(sink, event, build)
