"""`TurnTiming`: when each stage of one caller turn happened, for traces and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STAGES = ("early", "ended", "transcribed", "speculated", "replied", "acked", "first_audio")


@dataclass
class TurnTiming:
    """When each step of one caller turn happened, in seconds on the monotonic clock."""

    started: float  # the detector heard speech start (playback paused here)
    confirmed: float | None = None  # the cut went to the agent
    early: float | None = None  # quiet for `early_ms`: the final transcription started
    ended: float | None = None  # the detector called the turn finished
    speech_end: float | None = None  # the last speech (ended minus the detector's quiet time)
    transcribed: float | None = None
    speculated: float | None = None  # a speculative reply started on the early transcript
    replied: float | None = None  # the reply's first line was queued to speak
    acked: float | None = None  # an acknowledgement was queued
    first_audio: float | None = None  # the agent's next audio started playing
    text: str = ""
    resumed: bool = False  # a false start: playback went on where it paused
    adopted: bool = False  # the speculative reply was kept
    answer_cached: bool = False  # the reply came from the answer cache

    @property
    def voice_to_voice_ms(self) -> float | None:
        if self.first_audio is None or self.speech_end is None:
            return None
        return (self.first_audio - self.speech_end) * 1000

    @property
    def cut_ms(self) -> float | None:
        return None if self.confirmed is None else (self.confirmed - self.started) * 1000

    def stages(self) -> dict[str, float]:
        """Each stage that happened, in ms after the caller's last speech (negative if before)."""
        if self.speech_end is None:
            return {}
        out: dict[str, float] = {}
        for name in STAGES:
            at = getattr(self, name)
            if at is not None:
                out[name] = round((at - self.speech_end) * 1000, 1)
        return out


def event_data(timing: TurnTiming) -> dict[str, Any]:
    """Timings only: the transcript is the caller's words and stays out of traces."""
    return {
        "voice_to_voice_ms": timing.voice_to_voice_ms,
        "cut_ms": timing.cut_ms,
        "resumed": timing.resumed,
        "stages_ms": timing.stages(),
        "adopted": timing.adopted,
        "answer_cached": timing.answer_cached,
    }


__all__ = ["STAGES", "TurnTiming", "event_data"]
