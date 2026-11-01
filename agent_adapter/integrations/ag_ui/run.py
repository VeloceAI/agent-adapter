"""`AGUIRun`: what an agent function gets for one AG-UI run (tools, text, the conversation)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from agent_adapter.core.context import ToolContext
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall, ToolResult
from agent_adapter.exec.executor import Executor
from agent_adapter.integrations.ag_ui._channel import MAX_EVENTS, EventChannel
from agent_adapter.integrations.ag_ui._events import ToolEvents
from agent_adapter.integrations.ag_ui._input import to_messages
from agent_adapter.integrations.ag_ui._support import core
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.sinks import FanoutSink

if TYPE_CHECKING:
    from ag_ui.core import RunAgentInput

TextRole = Literal["assistant", "system", "developer", "user"]


class AGUIRun:
    """One AG-UI run: its input, an executor and context bound to its event stream.

    Run tools with `call` or pass `executor` and `ctx` to `run_tool_loop`: either way each
    call streams as `TOOL_CALL_START/ARGS/END` and `TOOL_CALL_RESULT`, and a call that
    waits for approval ends the run with an interrupt. Built by `AGUIBridge.stream`.
    """

    def __init__(
        self,
        run_input: RunAgentInput,
        *,
        executor: Executor,
        principal: str = "",
        max_events: int = MAX_EVENTS,
        trace: bool = True,
    ) -> None:
        self.input = run_input
        self.principal = principal
        self.channel = EventChannel(max_events=max_events)
        self.events = ToolEvents(self.channel, trace=trace)
        self.executor = executor.replace(hooks=(*executor.hooks, self.events))
        base = executor.sink
        self.ctx: ToolContext = executor.context(
            thread_id=run_input.thread_id, extras={"principal": principal} if principal else {}
        )
        self.ctx.sink = self.events if base is None else FanoutSink(base, self.events)
        self.resumed: list[ToolResult] = []  # calls a resume entry decided, in entry order
        self.run_ids: set[str] = {self.ctx.run_id}  # executor runs to end with this run

    @property
    def thread_id(self) -> str:
        """The AG-UI thread (conversation) id."""
        return str(self.input.thread_id)

    @property
    def run_id(self) -> str:
        """The AG-UI run id the client sent (the executor's run id is `ctx.run_id`)."""
        return str(self.input.run_id)

    def messages(self) -> list[Message]:
        """The input's conversation as adapter messages (see `to_messages`)."""
        return to_messages(self.input)

    async def call(
        self, name: str, arguments: Mapping[str, Any] | None = None, *, call_id: str = ""
    ) -> ToolResult:
        """Run one registry tool through the executor; every failure is a `ToolResult`."""
        call = ToolCall(name=name, arguments=dict(arguments or {}), id=call_id or new_id("call_"))
        return await self.executor.run_call(call, self.ctx)

    def text(self, content: str, *, role: TextRole = "assistant") -> str:
        """Stream one complete text message; returns its message id."""
        c = core()
        message_id = new_id("msg_")
        self.channel.put(c.TextMessageStartEvent(message_id=message_id, role=role))
        if content:
            self.channel.put(c.TextMessageContentEvent(message_id=message_id, delta=content))
        self.channel.put(c.TextMessageEndEvent(message_id=message_id))
        return message_id

    def custom(self, name: str, value: Any) -> None:
        """Send an application `CUSTOM` event (`value` must be JSON)."""
        self.channel.put(core().CustomEvent(name=name, value=value))
