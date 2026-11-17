"""`VoiceAgent`: turns, interruptions and spoken approval around the one executor."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Sequence
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.messages import Message
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ToolResult
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.progress import listen_progress
from agent_adapter.exec.tool_loop import run_tool_loop
from agent_adapter.hooks.base import Hook, ToolInvocation
from agent_adapter.runtime.ids import new_id
from agent_adapter.voice._speech import SpeechQueue
from agent_adapter.voice.approval import VoiceApproval
from agent_adapter.voice.meter import record_units
from agent_adapter.voice.ports import Speaker

logger = logging.getLogger(__name__)

DEDUPE_WINDOW_S = 120.0


class _Done(Hook):
    """Remembers the calls that finished in the current turn, for the note after a cut."""

    name = "voice_done"

    def __init__(self) -> None:
        self.finished: list[ToolResult] = []

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> ToolResult | None:
        if result.ok:
            self.finished.append(result)
        return None


class _Busy(Hook):
    """Counts the tool calls running now, for `mute_during_tools`."""

    name = "voice_busy"

    def __init__(self) -> None:
        self.running = 0

    async def before_tool(self, inv: ToolInvocation) -> None:
        self.running += 1

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> None:
        self.running = max(0, self.running - 1)

    async def on_tool_error(self, inv: ToolInvocation, error: BaseException) -> None:
        self.running = max(0, self.running - 1)


class VoiceAgent:
    """A voice agent from a model, an executor and a speaker; works with any audio stack.

    Feed it the caller's events: `user_started_speaking()` when they start talking and
    `user_said(text)` with each final transcript. Speech is cheap, effects are not:
    talking over the agent stops its speech and its thinking at once, but a tool that
    changes things (`effects=`) and has started finishes once and is reported in the next
    turn. Repeats of a finished call within `dedupe_window_s` return the first result.
    With `minute_item` ("provider:name"), `close()` bills the call's minutes at its unit price.
    With a `VoiceApproval`, a yes counts only if `user_started_speaking()` came after the
    question finished playing. With `mute_during_tools`, the caller cannot cut in while a
    tool runs (a `VoiceSession` drops that speech).
    """

    def __init__(
        self,
        model: LLMPort,
        executor: Executor,
        speaker: Speaker,
        *,
        instructions: str = "",
        registry: Registry | None = None,
        max_steps: int = 8,
        run_id: str | None = None,
        dedupe_window_s: float | None = DEDUPE_WINDOW_S,
        minute_item: str | None = None,
        mute_during_tools: bool = False,
    ) -> None:
        self._done = _Done()
        self._busy = _Busy()
        self.mute_during_tools = mute_during_tools
        changes: dict[str, Any] = {"hooks": (*executor.hooks, self._done, self._busy)}
        if dedupe_window_s is not None:
            changes["dedupe_window_s"] = dedupe_window_s
        self.executor = executor.replace(**changes)
        self.model = model
        self.registry = registry if registry is not None else executor.registry
        self.max_steps = max_steps
        self.ctx: ToolContext = self.executor.context(run_id=run_id or new_id("call_"))
        self._speech = SpeechQueue(speaker)
        self._system = [Message(role="system", content=instructions)] if instructions else []
        self.history: list[Message] = []
        self._turn: asyncio.Task[None] | None = None
        self._minute_item = minute_item
        self._started = time.monotonic()
        self._closed = False
        self.approval = executor.approval if isinstance(executor.approval, VoiceApproval) else None
        self._first_start: list[int | None] = []  # the question open as the caller began
        if self.approval is not None:
            self.approval.bind(self._ask_aloud, run_id=self.ctx.run_id)

    async def user_started_speaking(self) -> None:
        """The caller talks over the agent: stop speech now; cut the turn unless it asks yes/no."""
        approval, run_id = self.approval, self.ctx.run_id
        if approval is not None and approval.waiting_for(run_id):
            question = approval.armed_question(run_id)
            if not self._first_start:  # the earliest start since the last words: fail closed
                self._first_start.append(question)
            if question is not None:  # the question has played: they may be answering it
                await self._speech.stop()
            return
        await self._cut()

    async def user_said(self, text: str) -> None:
        """A final transcript: an answer to a waiting question, or a new turn."""
        first, self._first_start = self._first_start, []
        question = first[0] if first else None
        approval = self.approval
        if approval is not None and approval.answer(
            text, run_id=self.ctx.run_id, question=question
        ):
            return
        await self._cut()
        self._turn = asyncio.create_task(self._run_turn(text))

    @property
    def busy(self) -> bool:
        """A turn is running: the model is thinking, a tool runs, or the reply plays."""
        return self._turn is not None and not self._turn.done()

    @property
    def interruptible(self) -> bool:
        """Whether the caller may cut in now."""
        if self._speech.holding:
            return False
        return not (self.mute_during_tools and self._busy.running)

    def say(self, text: str, *, interruptible: bool = True, record: bool = True) -> None:
        """Speak `text` now, outside a model turn: a greeting, a notice, a goodbye.

        With `record`, what the caller heard of it joins the history. With
        `interruptible=False`, the caller cannot cut it (a `VoiceSession` drops that speech).
        """
        heard = self._heard if record else None
        self._speech.put(text, record=False, hold=not interruptible, heard=heard)

    async def wait_spoken(self) -> None:
        """Wait until everything queued to speak has played."""
        await self._speech.drain()

    def _heard(self, text: str) -> None:
        self.history.append(Message(role="assistant", content=text))

    async def wait(self) -> None:
        """Wait until the current turn has finished speaking."""
        if self._turn is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn

    def meter(self, item: str, units: float) -> None:
        """Bill `units` of `item` ("provider:name", e.g. STT seconds) to this call's run."""
        record_units(self.executor.ledger, self.ctx, item, units)

    async def close(self) -> None:
        """End the call: cut any turn, stop speaking and bill the call minutes (once)."""
        await self._cut()
        if self._closed:
            return
        self._closed = True
        if self.approval is not None:
            self.approval.unbind(self.ctx.run_id, self._ask_aloud)
        if self._minute_item is not None:
            self.meter(self._minute_item, (time.monotonic() - self._started) / 60)
        self.executor.end_run(self.ctx.run_id)

    async def _ask_aloud(self, text: str) -> None:
        self._first_start = []  # speech that began before this question cannot answer it
        self._speech.put(text, record=False)
        await self._speech.drain()

    async def _cut(self) -> None:
        """Cancel the turn first (so it cannot save a reply that never played), then hush."""
        turn, self._turn = self._turn, None
        live = turn is not None and not turn.done()
        if live and turn is not None:
            turn.cancel()
        await self._speech.stop()
        if live and turn is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await turn  # a call that changes things finishes before this returns

    async def _run_turn(self, text: str) -> None:
        start = [*self.history, Message(role="user", content=text)]
        self._done.finished = []
        self._speech.take_played()
        reply: Sequence[Message] | None = None
        try:
            with listen_progress(lambda line: self._speech.put(line, record=False)):
                result = await run_tool_loop(
                    self.model,
                    self.registry,
                    [*self._system, *start],
                    max_steps=self.max_steps,
                    executor=self.executor,
                    ctx=self.ctx,
                    on_text=self._speech.feed,
                )
            reply = result.messages[len(self._system) :]
            self._speech.flush()
            await self._speech.drain()
            self.history = list(reply)
        except asyncio.CancelledError:
            self.history = self._after_cut(start, reply)
            raise
        finally:
            self._busy.running = 0  # a hook that short-circuits can skip after_tool

    def _after_cut(self, start: list[Message], reply: Sequence[Message] | None) -> list[Message]:
        """The history after a cut: only what was really said, plus what was really done."""
        said = self._speech.take_played()
        history = list(start if reply is None else reply[:-1])
        if said:
            history.append(Message(role="assistant", content=said))
        done = self._done.finished if reply is None else []
        if done:
            lines = "; ".join(f"{r.name} finished: {r.value!r}" for r in done)
            history.append(
                Message(role="user", content=f"(The caller interrupted. Already done: {lines}.)")
            )
        return history
