"""Offline fakes for voice agents: test interruptions and approvals with no audio and no keys."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

from agent_adapter.voice.playout import PlayoutBuffer

__all__ = [
    "MemoryTransport",
    "RecordingSpeaker",
    "ScriptedSTT",
    "WordTTS",
    "settle",
    "silence",
    "tone",
]


class RecordingSpeaker:
    """A `Speaker` that records what it said; with `hold=True` each line waits for `release`."""

    def __init__(self, *, hold: bool = False) -> None:
        self.started: list[str] = []  # lines it began to say
        self.said: list[str] = []  # lines that played to the end
        self.stops = 0
        self._hold = hold
        self._allowance = asyncio.Semaphore(0)

    def release(self, lines: int = 1) -> None:
        """Let `lines` more held lines play to the end."""
        for _ in range(lines):
            self._allowance.release()

    async def say(self, text: str) -> None:
        self.started.append(text)
        if self._hold:
            await self._allowance.acquire()
        self.said.append(text)

    async def stop(self) -> None:
        self.stops += 1


async def settle(rounds: int = 50) -> None:
    """Let every ready task run (a few event-loop turns)."""
    for _ in range(rounds):
        await asyncio.sleep(0)


def tone(ms: float, *, level: int = 8_000, rate: int = 16_000) -> bytes:
    """`ms` of a loud square wave as PCM16: a stand-in for speech."""
    samples = int(rate * ms / 1000)
    return b"".join(
        (level if i % 40 < 20 else -level).to_bytes(2, "little", signed=True)
        for i in range(samples)
    )


def silence(ms: float, *, rate: int = 16_000) -> bytes:
    return bytes(int(rate * ms / 1000) * 2)


class MemoryTransport:
    """An `AudioTransport` a test drives: `speak(pcm)` sends caller audio, `play(ms)` plays
    `ms` of the agent's audio into `heard`, and `hang_up()` ends the call."""

    def __init__(self, *, in_rate: int = 16_000, out_rate: int = 16_000) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.heard = bytearray()
        self.buffer: PlayoutBuffer | None = None
        self.closed = False
        self._frames: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def frames(self) -> AsyncIterator[bytes]:
        while (frame := await self._frames.get()) is not None:
            yield frame

    async def start(self, buffer: PlayoutBuffer) -> None:
        self.buffer = buffer

    async def close(self) -> None:
        self.closed = True

    def speak(self, pcm: bytes, *, frame_ms: int = 20) -> None:
        step = self.in_rate * 2 * frame_ms // 1000
        for i in range(0, len(pcm), step):
            self._frames.put_nowait(pcm[i : i + step])

    def hang_up(self) -> None:
        self._frames.put_nowait(None)

    def play(self, ms: float) -> bytes:
        assert self.buffer is not None, "the session has not started"
        out = self.buffer.pull(int(self.out_rate * ms / 1000) * 2)
        self.heard += out
        return out


class WordTTS:
    """A `TTS` with `ms_per_word` of tone per word, in one chunk per word."""

    def __init__(self, *, sample_rate: int = 16_000, ms_per_word: float = 100) -> None:
        self.sample_rate = sample_rate
        self.ms_per_word = ms_per_word
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        for _ in text.split():
            yield tone(self.ms_per_word, rate=self.sample_rate)


class ScriptedSTT:
    """An `STT` that returns the given transcripts in order, then empty strings."""

    def __init__(self, texts: Iterable[str]) -> None:
        self._texts = list(texts)
        self.heard: list[int] = []  # bytes of audio in each turn

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.heard.append(len(pcm))
        return self._texts.pop(0) if self._texts else ""
