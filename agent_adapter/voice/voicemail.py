"""`VoicemailDetector`: tell an answering machine from a person on an outbound call, offline."""

from __future__ import annotations

import asyncio
import inspect
import itertools
import math
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal

from agent_adapter.voice.observers import CallObserver
from agent_adapter.voice.pcm import samples_of
from agent_adapter.voice.playout import fire

if TYPE_CHECKING:
    from agent_adapter.voice.session import VoiceSession

RATE = 16_000
Reason = Literal["greeting", "beep"]
PHRASES = (
    "leave a message",
    "leave your message",
    "after the tone",
    "after the beep",
    "not available",
    "unavailable",
    "can't take your call",
    "cannot take your call",
    "can't come to the phone",
    "voicemail",
    "voice mail",
    "mailbox",
    "record your message",
)


class VoicemailDetector(CallObserver):
    """Listens to the first `window_s` of a call for a machine.

    Two offline signals: a greeting that says what machines say (`PHRASES`, or your own
    `classify(text)`), and the beep (a steady tone of 400 to 2,500 Hz for `beep_ms`).
    Once the greeting is heard, the machine's words are kept from the agent. With
    `message`, the agent waits for the beep (or `beep_wait_s` of quiet after the
    greeting), speaks it and hangs up; with `on_detect(session, reason)`, you decide.
    """

    def __init__(
        self,
        *,
        message: str | None = None,
        on_detect: Callable[[VoiceSession, Reason], Awaitable[None] | None] | None = None,
        classify: Callable[[str], bool | Awaitable[bool]] | None = None,
        phrases: tuple[str, ...] = PHRASES,
        window_s: float = 20.0,
        beep_ms: float = 160.0,
        beep_wait_s: float = 4.0,
        min_level: float = 0.02,
    ) -> None:
        self.message = message
        self.on_detect = on_detect
        self.classify = classify
        self._phrases = re.compile("|".join(re.escape(p) for p in phrases), re.IGNORECASE)
        self.window_s = window_s
        self.beep_ms = beep_ms
        self.beep_wait_s = beep_wait_s
        self.min_level = min_level
        self.reasons: list[Reason] = []
        self._session: VoiceSession | None = None
        self._t0 = 0.0
        self._tone_ms = 0.0
        self._tone_hz = 0.0
        self._waiting: asyncio.TimerHandle | None = None

    @property
    def machine(self) -> bool:
        return bool(self.reasons)

    def started(self, session: VoiceSession) -> None:
        self._session, self._t0 = session, time.monotonic()

    def caller_audio(self, pcm: bytes) -> None:
        if "beep" in self.reasons or time.monotonic() - self._t0 > self.window_s + 30:
            return
        hz = tone_of(pcm, self.min_level)
        steady = bool(hz and self._tone_hz) and abs(hz - self._tone_hz) <= 0.06 * self._tone_hz
        self._tone_ms = self._tone_ms + len(pcm) * 500 / RATE if steady else 0.0
        self._tone_hz = hz
        if self._tone_ms >= self.beep_ms and (self.machine or self._in_window()):
            self._detect("beep")

    def caller_said(self, text: str) -> bool:
        if self.machine:
            return True  # the machine's own words: never a turn for the agent
        if not self._in_window():
            return False
        if self.classify is None:
            if self._phrases.search(text):
                self._detect("greeting")
                return True
            return False
        fire(self._classify(text))
        return False

    async def ended(self) -> None:
        if self._waiting is not None:
            self._waiting.cancel()

    def _in_window(self) -> bool:
        return time.monotonic() - self._t0 <= self.window_s

    async def _classify(self, text: str) -> None:
        assert self.classify is not None
        verdict = self.classify(text)
        if inspect.isawaitable(verdict):
            verdict = await verdict
        if verdict and not self.machine:
            self._detect("greeting")

    def _detect(self, reason: Reason) -> None:
        if reason in self.reasons or self._session is None:
            return
        self.reasons.append(reason)
        session = self._session
        if self.on_detect is not None:
            result = self.on_detect(session, reason)
            if inspect.isawaitable(result):
                fire(_awaited(result))
        if self.message is None:
            return
        if reason == "beep":
            if self._waiting is not None:
                self._waiting.cancel()
            fire(self._leave_message(session))
        else:
            loop = asyncio.get_running_loop()
            self._waiting = loop.call_later(self.beep_wait_s, self._detect, "beep")

    async def _leave_message(self, session: VoiceSession) -> None:
        agent = session.agent
        if agent is None or self.message is None:
            return
        await agent.wait()
        agent.say(self.message, interruptible=False)
        await session.hang_up(after_speech=True)


async def _awaited(value: Awaitable[None]) -> None:
    await value


def tone_of(pcm: bytes, min_level: float = 0.02) -> float:
    """The frequency of a steady tone in `pcm` (16 kHz), or 0.0 if it is not one.

    A pure tone crosses zero at even intervals; speech and noise do not.
    """
    samples = samples_of(pcm)
    if len(samples) < 64:
        return 0.0
    if math.sqrt(sum(s * s for s in samples) / len(samples)) / 32_768 < min_level:
        return 0.0
    crossings = [
        i - samples[i] / (samples[i] - samples[i - 1])
        for i in range(1, len(samples))
        if (samples[i - 1] < 0) != (samples[i] < 0) and samples[i] != samples[i - 1]
    ]
    if len(crossings) < 6:
        return 0.0
    gaps = [b - a for a, b in itertools.pairwise(crossings)]
    mean = sum(gaps) / len(gaps)
    spread = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
    hz = RATE / (2 * mean)
    if spread > 0.12 * mean or not 400 <= hz <= 2500:
        return 0.0
    return hz


__all__ = ["PHRASES", "VoicemailDetector", "tone_of"]
