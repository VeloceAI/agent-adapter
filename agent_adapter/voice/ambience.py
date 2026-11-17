"""`ThinkingSound`: a quiet sound while the agent thinks or a tool runs."""

from __future__ import annotations

import asyncio
import contextlib
import math
import struct
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_adapter.voice.observers import CallObserver
from agent_adapter.voice.pcm import Resampler

if TYPE_CHECKING:
    from agent_adapter.voice.session import VoiceSession


class ThinkingSound(CallObserver):
    """Loops `pcm` (16-bit mono at `sample_rate`) at `gain` once a turn has gone `delay_s`
    with nothing to say, and stops as soon as the agent speaks or the caller talks.

    It is queued `chunk_ms` at a time, so speech waits at most that long behind it. It
    never counts as the agent speaking: no cut, no echo guard, no voice-to-voice time.
    """

    def __init__(
        self,
        pcm: bytes,
        sample_rate: int,
        *,
        gain: float = 0.3,
        delay_s: float = 0.6,
        chunk_ms: int = 60,
    ) -> None:
        self.sample_rate = sample_rate
        self.delay_s = delay_s
        self.chunk_ms = chunk_ms
        count = len(pcm) // 2
        samples = struct.unpack(f"<{count}h", pcm[: count * 2])
        self._pcm = struct.pack(f"<{count}h", *(int(s * gain) for s in samples))
        self._task: asyncio.Task[None] | None = None

    @classmethod
    def from_wav(cls, path: str | Path, **options: Any) -> ThinkingSound:
        """Load a 16-bit mono WAV file."""
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                raise ValueError("ThinkingSound needs a 16-bit mono WAV")
            return cls(wav.readframes(wav.getnframes()), wav.getframerate(), **options)

    @classmethod
    def ticks(cls, *, every_s: float = 0.5, **options: Any) -> ThinkingSound:
        """A soft tick, like a clock, built in so no file is needed."""
        rate = 16_000
        tick = [
            int(9000 * math.sin(2 * math.pi * 1800 * i / rate) * math.exp(-i / 60))
            for i in range(400)
        ]
        loop = tick + [0] * (int(rate * every_s) - len(tick))
        return cls(struct.pack(f"<{len(loop)}h", *loop), rate, **options)

    def started(self, session: VoiceSession) -> None:
        self._task = asyncio.ensure_future(self._play(session))

    async def ended(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _play(self, session: VoiceSession) -> None:
        buffer = session.buffer
        sound = Resampler(self.sample_rate, buffer.sample_rate).feed(self._pcm)
        chunk = buffer.sample_rate * 2 * self.chunk_ms // 1000
        if not sound or chunk <= 0:
            return
        offset, thinking_since = 0, None
        while True:
            await asyncio.sleep(self.chunk_ms / 2000)
            agent, now = session.agent, time.monotonic()
            if agent is None or not agent.busy or session.speaker.speaking:
                thinking_since = None
                continue
            if session.caller_talking or buffer.paused:
                continue
            thinking_since = thinking_since or now
            if now - thinking_since < self.delay_s or not buffer.idle:
                continue
            piece = (sound[offset:] + sound)[:chunk]
            offset = (offset + chunk) % len(sound)
            segment = buffer.open(ambient=True)
            buffer.write(segment, piece)
            buffer.finish(segment)


__all__ = ["ThinkingSound"]
