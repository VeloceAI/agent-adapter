"""Turn detection from raw audio: when the caller starts talking and when they have finished.

`TurnDetector` is the port; Silero VAD, Smart Turn or your STT's own end-of-utterance
signal plug in behind it. `SilenceTurns` is a dependency-free default for 16-bit mono PCM:
loud for `start_ms` starts a turn, quiet for `end_ms` ends it.
"""

from __future__ import annotations

import math
import sys
from array import array
from dataclasses import dataclass
from typing import Literal, Protocol

TurnEvent = Literal["started", "ended"]


class TurnDetector(Protocol):
    """Takes audio frames as they arrive; returns the turn events each frame caused."""

    def feed(self, pcm: bytes) -> list[TurnEvent]: ...


class TurnTimer:
    """Turns from per-frame speech decisions: `start_ms` of speech starts one, `end_ms` of
    quiet ends it. Any VAD (an energy level, Silero, your STT's flags) can drive it."""

    def __init__(self, *, start_ms: float = 60, end_ms: float = 600) -> None:
        if start_ms < 0 or end_ms <= 0:
            raise ValueError("start_ms >= 0 and end_ms > 0")
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.speaking = False
        self.quiet_ms = 0.0
        self._loud_ms = 0.0

    def step(self, speech: bool, ms: float) -> list[TurnEvent]:
        """Advance by one frame of `ms` milliseconds."""
        if speech:
            self.quiet_ms = 0.0
            self._loud_ms += ms
            if not self.speaking and self._loud_ms >= self.start_ms:
                self.speaking = True
                return ["started"]
            return []
        self._loud_ms = 0.0
        self.quiet_ms += ms
        if self.speaking and self.quiet_ms >= self.end_ms:
            self.speaking = False
            return ["ended"]
        return []


@dataclass(frozen=True)
class EchoGuard:
    """When speech counts as a barge-in while the agent talks on speakers with no echo
    cancelling: `loud_ms` of audio at `level` RMS or above (a fraction of full scale; the
    agent's voice through the mic is quieter than a caller at it), or `speech_ms` of speech.

    Tune `level` on your hardware: above the agent's echo, below your voice.
    """

    level: float = 0.1
    loud_ms: float = 200
    speech_ms: float = 1200

    def __post_init__(self) -> None:
        if not 0 < self.level < 1 or self.loud_ms <= 0 or self.speech_ms <= 0:
            raise ValueError("0 < level < 1, loud_ms > 0 and speech_ms > 0")


def pcm16(data: bytes) -> array[int]:
    """16-bit little-endian PCM as native integers."""
    samples = array("h", data)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


class SilenceTurns:
    """Energy-based turns over 16-bit little-endian mono PCM.

    `threshold` is the RMS level, as a fraction of full scale, above which a frame is
    speech. Short noises under `start_ms` never start a turn, and pauses under `end_ms`
    never end one.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        threshold: float = 0.02,
        start_ms: float = 60,
        end_ms: float = 600,
    ) -> None:
        if sample_rate <= 0 or not 0 < threshold < 1:
            raise ValueError("sample_rate > 0 and 0 < threshold < 1")
        self.sample_rate = sample_rate
        self.level = threshold * 32_768
        self.timer = TurnTimer(start_ms=start_ms, end_ms=end_ms)
        self._odd = b""  # a byte left over from a frame of odd length

    @property
    def speaking(self) -> bool:
        return self.timer.speaking

    def feed(self, pcm: bytes) -> list[TurnEvent]:
        data, self._odd = self._odd + pcm, b""
        if len(data) % 2:
            data, self._odd = data[:-1], data[-1:]
        if not data:
            return []
        samples = pcm16(data)
        ms = len(samples) * 1000 / self.sample_rate
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        return self.timer.step(rms >= self.level, ms)


__all__ = ["EchoGuard", "SilenceTurns", "TurnDetector", "TurnEvent", "TurnTimer", "pcm16"]
