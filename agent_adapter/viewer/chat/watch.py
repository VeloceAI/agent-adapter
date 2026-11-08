"""TurnWatch: a hook and event sink on the host's executor that feeds live turns by run id."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any, ClassVar

from agent_adapter.core.errors import log_failure
from agent_adapter.core.types import Event, ToolResult
from agent_adapter.hooks.base import Hook, ShortCircuit, ToolInvocation
from agent_adapter.tracing.events import EventType
from agent_adapter.viewer.chat.live import LiveTurn, preview

logger = logging.getLogger(__name__)

_TOOL_STATUS = {
    EventType.TOOL_STARTED: "running",
    EventType.TOOL_FINISHED: "ok",
    EventType.TOOL_FAILED: "failed",
    EventType.TOOL_PENDING: "pending",
}


class TurnWatch(Hook):
    """Routes tool and model activity of a chat turn's run to its `LiveTurn`.

    Events arrive on the loop thread; `emit` never raises (an EventSink rule).
    """

    name: ClassVar[str] = "chat_watch"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, LiveTurn] = {}

    def add(self, live: LiveTurn) -> None:
        with self._lock:
            self._runs[live.run_id] = live

    def remove(self, live: LiveTurn) -> None:
        with self._lock:
            if self._runs.get(live.run_id) is live:
                del self._runs[live.run_id]

    def find(self, run_id: str) -> LiveTurn | None:
        with self._lock:
            return self._runs.get(run_id)

    async def before_tool(self, inv: ToolInvocation) -> ShortCircuit | None:
        live = self.find(inv.ctx.run_id)
        if live is not None:
            call_id = inv.ctx.call_id or inv.call.id
            live.tool(call_id, name=inv.spec.name, arguments=preview(inv.call.arguments))
        return None

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> ToolResult | None:
        live = self.find(inv.ctx.run_id)
        if live is not None and result.ok:
            live.tool(inv.ctx.call_id or inv.call.id, output=preview(result.value))
        return None

    def emit(self, event: Event) -> None:
        try:
            live = self.find(event.run_id)
            if live is not None:
                _route(live, event)
        except Exception as exc:  # the chat view must never break the traced call
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "chat: could not show %s of run %s",
                event.type,
                event.run_id,
            )


def _route(live: LiveTurn, event: Event) -> None:
    data: Mapping[str, Any] = event.data
    status = _TOOL_STATUS.get(event.type)
    if status is not None:
        ms = data.get("latency_ms")
        live.tool(
            event.call_id,
            name=str(data.get("tool", "")),
            status=status,
            ms=round(float(ms), 1) if isinstance(ms, int | float) else None,
            error_kind=str(data["error_kind"]) if "error_kind" in data else None,
            cached=bool(data["cached"]) if "cached" in data else None,
        )
    elif event.type == EventType.LLM_STARTED:
        live.close_drafts()
    elif event.type == EventType.LLM_FINISHED:
        usage = data.get("usage") or {}
        cost = data.get("cost_usd")
        live.usage(
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            float(cost) if isinstance(cost, int | float) else 0.0,
        )
