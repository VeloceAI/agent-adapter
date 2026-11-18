"""`Speed`: the speed tricks of a `VoiceAgent`, and the per-call fast path that runs them."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import ToolUsageError, log_failure
from agent_adapter.core.messages import Message
from agent_adapter.core.registry import Registry
from agent_adapter.voice.ack import Acknowledgements
from agent_adapter.voice.answers import Answer, AnswerCache
from agent_adapter.voice.speculate import (
    SPECULATIVE_STAGE,
    Speculation,
    SpeculationGate,
    SpeculativeTools,
    cancel_quietly,
    same_words,
)
from agent_adapter.voice.timing import TurnTiming
from agent_adapter.voice.warm import Warmup

if TYPE_CHECKING:
    from agent_adapter.voice._speech import SpeechQueue
    from agent_adapter.voice.agent import VoiceAgent

logger = logging.getLogger(__name__)


class Speed:
    """What a `VoiceAgent` does to answer sooner; each part is optional.

    - `speculate`: start the reply on the early transcript; kept only if the final one has
      the same words and nothing was said since, dropped if the caller goes on.
    - `tools`: the read-only tools that reply may run before the turn is final.
    - `acks`: short pre-rendered lines while a slow reply is on its way.
    - `answers` with `tenant`: replies to repeated questions, with their audio.
    - `warmup`: connections, lines and read-only prefetches at call start.
    - `first_words`: speak a reply's first clause once it has this many words, not its
      first whole sentence (None: sentence by sentence).
    """

    def __init__(
        self,
        *,
        speculate: bool = True,
        tools: SpeculativeTools | None = None,
        acks: Acknowledgements | None = None,
        answers: AnswerCache | None = None,
        tenant: str = "",
        warmup: Warmup | None = None,
        first_words: int | None = 4,
    ) -> None:
        if answers is not None and not tenant:
            raise ToolUsageError("Speed(answers=...) needs tenant=, so tenants never share")
        if tools is not None and not speculate:
            raise ToolUsageError("speculative tools need speculate=True")
        self.speculate = speculate
        self.tools = tools
        self.acks = acks
        self.answers = answers
        self.tenant = tenant
        self.warmup = warmup
        self.first_words = first_words

    def check(self, registry: Registry) -> None:
        if self.tools is not None:
            self.tools.check(registry)
        if self.warmup is not None:
            self.warmup.check(registry)


class _Tap:
    """Passes a turn's words to the speech queue and notes when its first line is queued."""

    def __init__(self, speech: SpeechQueue, on_line: Any) -> None:
        self._speech = speech
        self._on_line = on_line
        self.replied = False

    def feed(self, chunk: str) -> None:
        before = self._speech.queued
        self._speech.feed(chunk)
        self._check(before)

    def put(self, text: str, *, record: bool = True) -> None:
        before = self._speech.queued
        self._speech.put(text, record=record)
        self._check(before)

    def flush(self) -> None:
        before = self._speech.queued
        self._speech.flush()
        self._check(before)

    def _check(self, before: int) -> None:
        if not self.replied and self._speech.queued > before:
            self.replied = True
            self._on_line()


class FastPath:
    """One call's speed state: the running speculation, the gate and the turn plans."""

    def __init__(self, speed: Speed) -> None:
        self.speed = speed
        self.gate = SpeculationGate(speed.tools)
        self.speculation: Speculation | None = None
        self._agent: VoiceAgent | None = None
        self._early_ack: asyncio.Task[str | None] | None = None
        self._early_acked: float | None = None

    def bind(self, agent: VoiceAgent) -> None:
        self.speed.check(agent.registry)
        self._agent = agent

    @property
    def agent(self) -> VoiceAgent:
        assert self._agent is not None, "FastPath.bind was not called"
        return self._agent

    def speculate(self, text: str) -> None:
        agent, old = self.agent, self.speculation
        if not self.speed.speculate or not text.strip() or agent.busy or agent._waiting:
            return
        if old is not None and not old.dropped and old.key == same_words(text):
            return
        self.drop()
        spec = Speculation(text, list(agent.history))
        spec.task = asyncio.create_task(self._speculate(spec))
        self.speculation = spec
        agent._trace("speculation", outcome="started")

    def drop(self) -> None:
        early, self._early_ack, self._early_acked = self._early_ack, None, None
        if early is not None:
            early.cancel()
        spec, self.speculation = self.speculation, None
        if spec is not None and not spec.adopted:
            spec.drop()
            self.agent._trace("speculation", outcome="dropped", tool_calls=spec.tool_calls)

    @property
    def acked_early(self) -> bool:
        """An acknowledgement played at the end of the turn, before its transcript."""
        return self._early_acked is not None

    def turn_ended(self) -> None:
        """The caller finished a real turn: acknowledge now if the reply will be slow."""
        acks, spec, agent = self.speed.acks, self.speculation, self.agent
        if acks is None or agent.busy or agent._waiting or self._early_ack is not None:
            return
        if spec is not None and (spec.ready or self._known(spec.text)):
            return
        now = time.monotonic()
        thought_ms = (now - spec.started) * 1000 if spec is not None else 0.0

        def ready() -> bool:
            return spec is not None and spec.ready

        def say(line: str) -> None:
            self._early_acked = time.monotonic()
            agent._speech.put(line, record=False)

        self._early_ack = asyncio.create_task(acks.run(thought_ms, ready, say))

    async def close(self) -> None:
        spec, self.speculation = self.speculation, None
        if spec is not None:
            await cancel_quietly(spec.task)
        await cancel_quietly(self._early_ack)

    def turn(self, text: str, timing: TurnTiming | None) -> asyncio.Task[None]:
        """The task that answers `text`: a cached answer, the kept speculation, or a new turn."""
        agent, now = self.agent, time.monotonic()
        timing = timing if timing is not None else TurnTiming(started=now)
        early, self._early_ack = self._early_ack, None
        if early is not None:
            early.cancel()  # not played yet: the turn plans its own
        timing.acked, self._early_acked = self._early_acked, None
        spec, self.speculation = self.speculation, None
        answer = self._cached(text)
        if spec is not None and (answer is not None or not spec.matches(text, agent.history)):
            self.speculation = spec
            self.drop()
            spec = None
        start = spec.started if spec is not None else now
        tap = _Tap(agent._speech, lambda: self._replied(timing, start))
        work: Awaitable[None]
        if answer is not None:
            timing.answer_cached = True
            work = agent._play_answer(text, answer.text, tap)
            return asyncio.create_task(self._guard(work, tap, timing, start, None))
        if spec is not None and spec.task is not None:
            timing.adopted, timing.speculated = True, spec.started
            agent._new_turn()
            spec.adopt(tap)
            agent._trace("speculation", outcome="adopted", tool_calls=spec.tool_calls)
            return asyncio.create_task(self._guard(spec.task, tap, timing, start, text))
        work = agent._run_turn(text, tap)
        return asyncio.create_task(self._guard(work, tap, timing, start, text))

    async def _speculate(self, spec: Speculation) -> None:
        spec.bind()
        try:
            await self.agent._run_turn(spec.text, spec, stage=SPECULATIVE_STAGE)
        except Exception as exc:
            if spec.adopted:
                raise
            spec.failed = True  # the final transcript gets a fresh turn instead
            log_failure(logger, logging.INFO, exc, "speculative reply failed")

    async def _guard(
        self,
        work: Awaitable[None],
        tap: _Tap,
        timing: TurnTiming,
        start: float,
        text: str | None,
    ) -> None:
        agent, acks = self.agent, self.speed.acks
        before = len(agent.history)
        ack = None
        if acks is not None and not timing.answer_cached and timing.acked is None:
            thought_ms = (time.monotonic() - start) * 1000
            ack = asyncio.create_task(
                acks.run(thought_ms, lambda: tap.replied, self._acked(timing))
            )
        try:
            await work
        finally:
            await cancel_quietly(ack)
        if text is not None:
            self._store(text, agent.history[before:])

    def _acked(self, timing: TurnTiming) -> Any:
        def say(line: str) -> None:
            timing.acked = time.monotonic()
            self.agent._speech.put(line, record=False)

        return say

    def _replied(self, timing: TurnTiming, start: float) -> None:
        timing.replied = now = time.monotonic()
        if self.speed.acks is not None and not timing.answer_cached:
            self.speed.acks.learn((now - start) * 1000)

    def _cached(self, text: str) -> Answer | None:
        answers = self.speed.answers
        return answers.get(self.speed.tenant, text) if answers is not None else None

    def _known(self, text: str) -> bool:
        answers = self.speed.answers
        return answers is not None and answers.has(self.speed.tenant, text)

    def _store(self, text: str, added: list[Message]) -> None:
        answers, history = self.speed.answers, self.agent.history
        if answers is None or len(added) != 2:
            return
        opened = not any(message.role == "user" for message in history[: -len(added)])
        reply = added[1]
        if opened and reply.role == "assistant" and not reply.tool_calls and reply.content.strip():
            answers.put(self.speed.tenant, text, reply.content)


__all__ = ["FastPath", "Speed"]
