"""Speech-to-speech models (OpenAI Realtime, Gemini Live) with the executor's safety.

The model hears and speaks on its own socket; `RealtimeAgent` runs the tools it calls.
Speech is cheap, effects are not: when the provider cancels calls because the caller cut
in, read-only tools stop at once, but a tool that changes things finishes and the model is
told it is done. Repeats within the window return the first result. A `confirm` tool asks
aloud and only the caller's own transcribed words can approve it.

A codec turns one provider's JSON messages into the neutral events below and back.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_adapter.core.errors import log_failure
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolCall
from agent_adapter.exec._commit import changes_things
from agent_adapter.exec._loop_steps import _tool_content
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.progress import listen_progress
from agent_adapter.runtime.ids import new_id
from agent_adapter.voice.approval import VoiceApproval
from agent_adapter.voice.meter import record_units
from agent_adapter.voice.socket import TextSocket

logger = logging.getLogger(__name__)
DEDUPE_WINDOW_S = 120.0


@dataclass(frozen=True)
class ToolCalled:
    call_id: str
    name: str
    arguments: Mapping[str, Any] | None  # None: the model sent arguments that are not JSON


@dataclass(frozen=True)
class SpeechStarted:
    """The caller started talking over the model."""


@dataclass(frozen=True)
class CallerSaid:
    text: str  # a final transcript of the caller's own audio


@dataclass(frozen=True)
class CallsCancelled:
    ids: tuple[str, ...] = field(default_factory=tuple)


RealtimeEvent = ToolCalled | SpeechStarted | CallerSaid | CallsCancelled


class RealtimeCodec(Protocol):
    """One provider's wire format: JSON objects in and out."""

    def setup(self, instructions: str, tools: list[ToolSpec]) -> list[dict[str, Any]]: ...

    def decode(self, message: Mapping[str, Any]) -> list[RealtimeEvent]: ...

    def tool_result(
        self, call: ToolCalled, output: str, *, respond: bool
    ) -> list[dict[str, Any]]: ...

    def say(self, text: str) -> list[dict[str, Any]]: ...

    def note(self, text: str) -> list[dict[str, Any]]: ...


class RealtimeAgent:
    """Runs a realtime model's tool calls through `executor` over a socket you opened.

    `socket` is any `TextSocket` (a `websockets` connection works as it is) already
    connected and authenticated to the provider. With `minute_item` ("provider:name"),
    `run` bills the call's minutes when it ends.
    """

    def __init__(
        self,
        codec: RealtimeCodec,
        socket: TextSocket,
        executor: Executor,
        *,
        instructions: str = "",
        registry: Registry | None = None,
        run_id: str | None = None,
        dedupe_window_s: float | None = DEDUPE_WINDOW_S,
        minute_item: str | None = None,
    ) -> None:
        if dedupe_window_s is not None:
            executor = executor.replace(dedupe_window_s=dedupe_window_s)
        self.executor = executor
        self.codec = codec
        self.socket = socket
        self.instructions = instructions
        self.registry = registry if registry is not None else executor.registry
        self.ctx = executor.context(run_id=run_id or new_id("call_"))
        self._minute_item = minute_item
        self._calls: dict[str, tuple[asyncio.Task[None], str]] = {}  # id -> (task, tool)
        self._cancelled: set[str] = set()
        self._caller_spoke_at = 0.0
        self.approval = executor.approval if isinstance(executor.approval, VoiceApproval) else None
        if self.approval is not None:
            self.approval.bind(self._say)

    async def run(self) -> None:
        """Send the setup, then handle the provider's messages until the socket closes."""
        started = time.monotonic()
        try:
            await self._send(self.codec.setup(self.instructions, self._tools()))
            async for raw in self.socket:
                for event in self.codec.decode(_parse(raw)):
                    self._handle(event)
            await self._settle()
        finally:
            await self._stop_calls()
            if self._minute_item is not None:
                minutes = (time.monotonic() - started) / 60
                record_units(self.executor.ledger, self.ctx, self._minute_item, minutes)
            self.executor.end_run(self.ctx.run_id)

    def _tools(self) -> list[ToolSpec]:
        return list(self.registry)

    def _handle(self, event: RealtimeEvent) -> None:
        if isinstance(event, ToolCalled):
            task = asyncio.ensure_future(self._call(event))
            self._calls[event.call_id] = (task, event.name)
            task.add_done_callback(lambda _: self._calls.pop(event.call_id, None))
        elif isinstance(event, SpeechStarted):
            self._caller_spoke_at = time.monotonic()
        elif isinstance(event, CallerSaid):
            if self.approval is not None:
                self.approval.answer(event.text)
        else:
            self._cancel(event.ids)

    def _cancel(self, ids: tuple[str, ...]) -> None:
        for call_id in ids:
            self._cancelled.add(call_id)
            running = self._calls.get(call_id)
            if running is not None and not self._changes_things(running[1]):
                running[0].cancel()  # read-only: stop now; nothing to report

    def _changes_things(self, name: str) -> bool:
        return name in self.registry and changes_things(self.registry.get(name))

    async def _call(self, event: ToolCalled) -> None:
        asked_at = time.monotonic()
        if event.arguments is None:
            output = json.dumps({"error": {"kind": "args-invalid", "message": "not JSON"}})
        else:
            call = ToolCall(name=event.name, arguments=dict(event.arguments), id=event.call_id)
            with listen_progress(lambda text: _spawn(self._say(text))):
                output = _tool_content(await self.executor.run_call(call, self.ctx))
        if event.call_id in self._cancelled:
            await self._send(self.codec.note(f"Already done: {event.name} finished: {output}"))
            return
        respond = self._caller_spoke_at < asked_at  # a caller who cut in gets the next reply
        await self._send(self.codec.tool_result(event, output, respond=respond))

    async def _say(self, text: str) -> None:
        await self._send(self.codec.say(text))

    async def _send(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            await self.socket.send(json.dumps(message))

    async def _settle(self) -> None:
        if self._calls:
            await asyncio.gather(
                *(task for task, _ in self._calls.values()), return_exceptions=True
            )

    async def _stop_calls(self) -> None:
        tasks = [task for task, _ in self._calls.values()]
        for task in tasks:
            task.cancel()  # a call that changes things still finishes (the executor shields it)
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


def _parse(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        message = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return message if isinstance(message, dict) else {}


_BACKGROUND: set[asyncio.Task[Any]] = set()


def _spawn(work: Any) -> None:
    task = asyncio.ensure_future(work)
    _BACKGROUND.add(task)
    task.add_done_callback(_done)


def _done(task: asyncio.Task[Any]) -> None:
    _BACKGROUND.discard(task)
    error = None if task.cancelled() else task.exception()
    if error is not None:
        log_failure(logger, logging.WARNING, error, "realtime: progress line not sent")


__all__ = [
    "CallerSaid",
    "CallsCancelled",
    "RealtimeAgent",
    "RealtimeCodec",
    "RealtimeEvent",
    "SpeechStarted",
    "ToolCalled",
]
