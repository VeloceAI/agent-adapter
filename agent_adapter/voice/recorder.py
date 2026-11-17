"""`CallRecorder`: a stereo WAV of the call, the caller on the left and the agent on the right."""

from __future__ import annotations

import asyncio
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from agent_adapter.voice.observers import CallObserver
from agent_adapter.voice.pcm import Resampler

if TYPE_CHECKING:
    from agent_adapter.voice.session import VoiceSession

RATE = 16_000


class _Track:
    """Audio placed at the time it was heard, with silence in the gaps."""

    def __init__(self, limit: int) -> None:
        self.data = bytearray()
        self.limit = limit

    def add(self, pcm: bytes, at_byte: int) -> None:
        if at_byte > len(self.data):
            self.data += bytes(at_byte - len(self.data))
        room = self.limit - len(self.data)
        if room > 0:
            self.data += pcm[: room - room % 2]


class CallRecorder(CallObserver):
    """Writes `path` when the call ends: 16 kHz, 16-bit, two channels.

    Each side is placed at the time it was heard, so the two line up. Recording stops
    after `max_minutes`. Tell callers they are recorded where the law asks you to.
    """

    def __init__(self, path: str | Path, *, max_minutes: float = 60.0) -> None:
        self.path = Path(path)
        limit = int(RATE * 60 * max_minutes) * 2
        self._caller, self._agent = _Track(limit), _Track(limit)
        self._lock = threading.Lock()
        self._t0 = 0.0
        self._resampler: Resampler | None = None

    def started(self, session: VoiceSession) -> None:
        self._t0 = time.monotonic()
        self._resampler = Resampler(session.transport.out_rate, RATE)

    def caller_audio(self, pcm: bytes) -> None:
        with self._lock:
            self._caller.add(pcm, max(self._now() - len(pcm), len(self._caller.data)))

    def agent_audio(self, pcm: bytes) -> None:
        with self._lock:
            if self._resampler is None:
                return
            self._agent.add(self._resampler.feed(pcm), self._now())

    async def ended(self) -> None:
        with self._lock:
            caller, agent = bytes(self._caller.data), bytes(self._agent.data)
        await asyncio.to_thread(_write_stereo, self.path, caller, agent)

    def _now(self) -> int:
        """Bytes of 16 kHz audio since the call started."""
        return int((time.monotonic() - self._t0) * RATE) * 2


def _write_stereo(path: Path, left: bytes, right: bytes) -> None:
    size = max(len(left), len(right))
    frames = array("h", bytes(size * 2))
    frames[0::2] = array("h", left.ljust(size, b"\0"))
    frames[1::2] = array("h", right.ljust(size, b"\0"))  # moves 2-byte samples, never reads them
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(frames.tobytes())


__all__ = ["CallRecorder"]
