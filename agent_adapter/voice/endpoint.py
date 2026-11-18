"""`EagerTurns`: end a turn sooner when the caller's words look finished, later when not.

No model and no provider feature: it reads the early transcript `VoiceSession` already
makes (any STT) and moves the end-of-turn wait of the detector it wraps.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any

from agent_adapter.voice.turns import TurnDetector, TurnEvent, TurnTimer

_TRAILING_WORDS = (
    "and but so or because um uh er erm hmm the a an to of with for in on at my your our "
    "is are was were if then like that which who when where from about than as by into "
    "plus minus also just"
)
_SHORT_WORDS = "yes yeah yep no nope sure okay ok correct right exactly thanks thank bye "
_TRAILING = frozenset(_TRAILING_WORDS.split())
_SHORT_ANSWERS = frozenset((_SHORT_WORDS + "done stop cancel please").split())
_WORD = re.compile(r"[a-z0-9']+")


def looks_done(text: str) -> bool | None:
    """True if `text` reads as a finished turn, False if it trails off, None if unsure."""
    stripped = text.strip()
    if not stripped:
        return None
    words = _WORD.findall(stripped.lower())
    if not words:
        return None
    if stripped.endswith((",", "-", "...", "…")) or words[-1] in _TRAILING:
        return False
    if len(words) >= 2 and words[-1].isdigit() and words[-2].isdigit():
        return False  # digits read out in groups: more may follow
    if stripped.endswith((".", "?", "!")):
        return True
    return True if len(words) <= 3 and words[0] in _SHORT_ANSWERS else None


class EagerTurns:
    """Wraps a detector with a `TurnTimer` (`SilenceTurns`, `SileroTurns`, or one `SmartTurn`
    wraps). After `hint(text)` the turn ends `done_ms` after speech when the words look
    finished, or waits `hold_ms` when they trail off ("and", "the", a comma).

    It learns each caller: mid-turn pauses and turns it ended too soon (the caller went on
    within `regret_ms`) raise the wait for finished-looking words, up to the detector's own.
    """

    def __init__(
        self,
        detector: TurnDetector,
        *,
        done_ms: float = 250,
        hold_ms: float = 1500,
        regret_ms: float = 1000,
    ) -> None:
        timer = _timer_in(detector)
        if timer is None:
            raise ValueError("EagerTurns needs a detector with a TurnTimer (Silence, Silero)")
        self.detector = detector
        self.timer = timer
        self.base_ms = timer.end_ms
        self.done_ms = min(done_ms, self.base_ms)
        self.hold_ms = max(hold_ms, self.base_ms)
        self.regret_ms = regret_ms
        self.verdict: bool | None = None  # what the last hint said about this turn
        self._pauses: deque[float] = deque(maxlen=20)
        self._ended_early = False
        self._since_end: float | None = None
        self._quiet = 0.0

    def feed(self, pcm: bytes) -> list[TurnEvent]:
        events = self.detector.feed(pcm)
        timer, ms = self.timer, len(pcm) * 500 / 16_000
        if timer.quiet_ms == 0 and self._quiet > 0 and timer.speaking:
            if self._quiet >= 100:
                self._pauses.append(self._quiet)  # the caller paused, then went on
            self._reset()
        self._quiet = timer.quiet_ms
        if self._since_end is not None:
            self._since_end += ms
        for event in events:
            if event == "started":
                if self._ended_early and (self._since_end or 0) < self.regret_ms:
                    self.done_ms = min(self.base_ms, self.done_ms + 100)  # cut them off
                self._ended_early, self._since_end = False, None
                self._reset()
            else:
                self._ended_early = timer.end_ms < self.base_ms
                self._since_end, self._quiet = 0.0, 0.0
                self._reset()
        return events

    def hint(self, text: str) -> bool | None:
        """The transcript so far; moves this turn's end. Returns `looks_done(text)`."""
        verdict = self.verdict = looks_done(text)
        if verdict is True:
            pause = sorted(self._pauses)[int(len(self._pauses) * 0.9)] if self._pauses else 0
            self.timer.end_ms = min(self.base_ms, max(self.done_ms, pause + 50))
        elif verdict is False:
            self.timer.end_ms = self.hold_ms
        else:
            self.timer.end_ms = self.base_ms
        return verdict

    def _reset(self) -> None:
        self.timer.end_ms, self.verdict = self.base_ms, None


def _timer_in(detector: Any) -> TurnTimer | None:
    for owner in (detector, getattr(detector, "detector", None)):
        timer = getattr(owner, "timer", None)
        if isinstance(timer, TurnTimer):
            return timer
    return None


__all__ = ["EagerTurns", "looks_done"]
