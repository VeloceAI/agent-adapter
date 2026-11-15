"""Turn detection from raw audio: when the caller starts talking and when they have finished.

`TurnDetector` is the port; Silero VAD, Smart Turn or your STT's own end-of-utterance
signal plug in behind it. `SilenceTurns` is a dependency-free default for 16-bit mono PCM:
loud for `start_ms` starts a turn, quiet for `end_ms` ends it.
"""

from __future__ import annotations

import math
import sys
from array import array
from typing import Literal, Protocol

TurnEvent = Literal["started", "ended"]


class TurnDetector(Protocol):
    """Takes audio frames as they arrive; returns the turn events each frame caused."""

    def feed(self, pcm: bytes) -> list[TurnEvent]: ...


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
        if sample_rate <= 0 or not 0 < threshold < 1 or start_ms < 0 or end_ms <= 0:
            raise ValueError("sample_rate > 0, 0 < threshold < 1, start_ms >= 0, end_ms > 0")
        self.sample_rate = sample_rate
        self.level = threshold * 32_768
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.speaking = False
        self._loud_ms = 0.0
        self._quiet_ms = 0.0
        self._odd = b""  # a byte left over from a frame of odd length

    def feed(self, pcm: bytes) -> list[TurnEvent]:
        data, self._odd = self._odd + pcm, b""
        if len(data) % 2:
            data, self._odd = data[:-1], data[-1:]
        if not data:
            return []
        samples = array("h", data)
        if sys.byteorder == "big":
            samples.byteswap()
        ms = len(samples) * 1000 / self.sample_rate
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        return self._step(rms >= self.level, ms)

    def _step(self, loud: bool, ms: float) -> list[TurnEvent]:
        if loud:
            self._quiet_ms = 0.0
            self._loud_ms += ms
            if not self.speaking and self._loud_ms >= self.start_ms:
                self.speaking = True
                return ["started"]
            return []
        self._loud_ms = 0.0
        self._quiet_ms += ms
        if self.speaking and self._quiet_ms >= self.end_ms:
            self.speaking = False
            return ["ended"]
        return []


__all__ = ["SilenceTurns", "TurnDetector", "TurnEvent"]
