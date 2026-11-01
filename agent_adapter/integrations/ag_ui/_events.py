"""Executor activity as AG-UI events: a hook sees arguments and results, a sink the rest.

Trace events carry no arguments or results (by design), so `TOOL_CALL_ARGS` and
`TOOL_CALL_RESULT` come from the hook. `tool.failed` and `tool.pending` come from the
sink, because no `after_tool` hook runs for them. Other trace events become `CUSTOM`.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_adapter.core.errors import log_failure
from agent_adapter.core.types import Event, ToolResult, to_jsonable
from agent_adapter.hooks.base import Hook, ToolInvocation
from agent_adapter.integrations._native_gate import json_text
from agent_adapter.integrations.ag_ui._channel import EventChannel
from agent_adapter.integrations.ag_ui._support import METADATA_KEY, TARGET, core
from agent_adapter.integrations.base import model_visible_error
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.events import EventType
from agent_adapter.tracing.sinks import CONTENT_KEYS

logger = logging.getLogger(__name__)

TRACE_EVENT = f"{METADATA_KEY}.trace"
_HANDLED_ELSEWHERE = frozenset(
    {EventType.TOOL_STARTED, EventType.TOOL_FINISHED, EventType.RUN_STARTED, EventType.RUN_FINISHED}
)


@dataclass
class _Call:
    tool: str
    arguments: Mapping[str, Any] | None = None
    started: bool = False
    finished: bool = False


@dataclass(frozen=True)
class Paused:
    """A call that ended pending: what an interrupt needs to resume it."""

    run_id: str
    call_id: str
    tool: str
    arguments: Mapping[str, Any] | None
    approval_id: str


class ToolEvents(Hook):
    """Both the run's hook (added last, so it sees final results) and its `EventSink`."""

    name = TARGET

    def __init__(self, channel: EventChannel, *, trace: bool = True) -> None:
        self._channel = channel
        self._trace = trace
        self._lock = threading.Lock()
        self._calls: dict[str, _Call] = {}
        self._announced: set[str] = set()
        self._paused: dict[str, Paused] = {}  # by approval id: an outer call replaces its nested

    async def before_tool(self, inv: ToolInvocation) -> None:
        self.started(inv.call.id, inv.spec.name, inv.call.arguments)
        return None

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> None:
        if result.ok:
            self._result(result.call_id, result.name, json_text(to_jsonable(result.value)))
        else:
            error = (
                model_visible_error(result.exception)
                if result.exception is not None
                else {"error": {"kind": result.error.kind if result.error else "", "message": ""}}
            )
            self._result(result.call_id, result.name, json.dumps(error))
        return None

    def emit(self, event: Event) -> None:
        """EventSink: never raises into the traced call."""
        try:
            self._on(event)
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "ag_ui: event %s not sent", event.type)

    def announced(self, call_id: str) -> None:
        """Mark a call shown in an earlier run: its START, ARGS and END are not sent again."""
        with self._lock:
            self._announced.add(call_id)

    def started(self, call_id: str, tool: str, arguments: Mapping[str, Any] | None) -> None:
        """Send START, ARGS and END for a call once; remember its arguments."""
        with self._lock:
            call = self._calls.setdefault(call_id, _Call(tool))
            if call.arguments is None and arguments is not None:
                call.arguments = dict(arguments)
            if call.started:
                return
            call.started = True
            shown = call_id in self._announced
        if shown:
            return
        c = core()
        self._channel.put(c.ToolCallStartEvent(tool_call_id=call_id, tool_call_name=tool))
        if arguments is not None:
            delta = json.dumps(to_jsonable(dict(arguments)), ensure_ascii=False)
            self._channel.put(c.ToolCallArgsEvent(tool_call_id=call_id, delta=delta))
        self._channel.put(c.ToolCallEndEvent(tool_call_id=call_id))

    def denied(self, call_id: str, tool: str, error: BaseException) -> None:
        """Send a model-visible error as the result of a call that will not run."""
        self._result(call_id, tool, json.dumps(model_visible_error(error)))

    def pause(self, paused: Paused) -> None:
        """Record a call that is waiting for approval again (a resume the port left open)."""
        with self._lock:
            self._paused[paused.approval_id or paused.call_id] = paused

    def paused(self) -> list[Paused]:
        """Calls this run left waiting for approval, in the order they paused."""
        with self._lock:
            return list(self._paused.values())

    def _result(self, call_id: str, tool: str, content: str) -> None:
        self.started(call_id, tool, None)
        with self._lock:
            call = self._calls[call_id]
            if call.finished:
                return
            call.finished = True
        self._channel.put(
            core().ToolCallResultEvent(
                message_id=new_id("msg_"), tool_call_id=call_id, content=content, role="tool"
            )
        )

    def _on(self, event: Event) -> None:
        if event.type in _HANDLED_ELSEWHERE:
            return
        tool = str(event.data.get("tool", ""))
        if event.type == EventType.TOOL_FAILED:
            kind = str(event.data.get("error_kind", ""))
            error = {"error": {"kind": kind, "message": f"tool {tool!r} failed ({kind})"}}
            self._result(event.call_id, tool, json.dumps(error))
        elif event.type == EventType.TOOL_PENDING:
            self.started(event.call_id, tool, None)
            with self._lock:
                arguments = self._calls[event.call_id].arguments
            approval_id = str(event.data.get("approval_id", ""))
            self.pause(Paused(event.run_id, event.call_id, tool, arguments, approval_id))
        elif self._trace:
            data = {k: to_jsonable(v) for k, v in event.data.items() if k not in CONTENT_KEYS}
            value = {"type": event.type, "callId": event.call_id, "data": data}
            self._channel.put(core().CustomEvent(name=TRACE_EVENT, value=value), droppable=True)
