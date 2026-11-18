"""`InputCheck`: the caller's audio, frame by frame; gate hiss, ask again after a bad turn."""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_adapter.tracing.events import emit, make_event
from agent_adapter.voice.observers import CallObserver
from agent_adapter.voice.quality._levels import SILENT_DB, FrameStats, measure

if TYPE_CHECKING:
    from agent_adapter.voice.session import VoiceSession

logger = logging.getLogger(__name__)
RATE = 16_000  # what a VoiceSession's filters and observers get


@dataclass
class TurnAudio:
    """The caller's audio since the last transcript, and what was wrong with it."""

    speech_ms: float = 0.0
    clipped_ms: float = 0.0
    dropout_ms: float = 0.0
    level_db: float = SILENT_DB  # loudest speech frame
    floor_db: float = SILENT_DB  # the noise floor when the turn ended
    problems: list[str] = field(default_factory=list)


class InputCheck(CallObserver):
    """Watches the caller's audio for clipping, dropouts, a high noise floor or a voice too
    quiet to hear. Pass it as both a filter and an observer:
    `VoiceSession(filters=[check], observers=[check])`.

    As a filter, frames below `gate_db` become silence (hiss stops waking the turn detector),
    holding open `hold_ms` after speech so word endings stay. The noise floor is the
    quietest of the last `floor_frames` frames. A turn has a problem when `clip_share` of
    its sound clipped, its dropouts (exact silence for under `max_gap_ms` with speech on both
    sides) add up to `dropout_ms`, its loudest speech stays under `quiet_db`, or the floor
    sits above `noisy_db`. After such a turn it keeps the transcript from the agent and says
    `repeat_prompt` instead, at most `max_repeats` times in a row; `ask_again=False` only
    measures. Each frame costs well under a millisecond, in pure Python.
    """

    def __init__(
        self,
        *,
        gate_db: float | None = None,
        hold_ms: float = 250,
        clip_share: float = 0.25,
        dropout_ms: float = 120,
        max_gap_ms: float = 200,
        quiet_db: float = -45.0,
        noisy_db: float = -35.0,
        speech_margin_db: float = 10.0,
        floor_frames: int = 75,
        ask_again: bool = True,
        repeat_prompt: str = "Sorry, I didn't catch that clearly. Could you say it again?",
        max_repeats: int = 1,
    ) -> None:
        self.gate_db = gate_db
        self.hold_ms = hold_ms
        self.clip_share = clip_share
        self.dropout_ms = dropout_ms
        self.max_gap_ms = max_gap_ms
        self.quiet_db = quiet_db
        self.noisy_db = noisy_db
        self.speech_margin_db = speech_margin_db
        self.ask_again = ask_again
        self.repeat_prompt = repeat_prompt
        self.max_repeats = max_repeats
        self.floor_db = SILENT_DB
        self.turn = TurnAudio()
        self.last_turn: TurnAudio | None = None
        self.counts: Counter[str] = Counter()
        self.repeats = 0
        self._filtering = False
        self._open_ms = 0.0
        self._levels: deque[float] = deque(maxlen=floor_frames)
        self._after_speech = False
        self._gap_ms = 0.0
        self._session: VoiceSession | None = None

    def __call__(self, pcm: bytes) -> bytes:
        """The filter: measure the frame, then gate it."""
        self._filtering = True
        stats = self.feed(pcm)
        if self.gate_db is None:
            return pcm
        ms = stats.samples * 1000 / RATE
        if stats.level_db >= self.gate_db:
            self._open_ms = self.hold_ms
            return pcm
        if self._open_ms > 0:
            self._open_ms -= ms
            return pcm
        return bytes(len(pcm))

    def feed(self, pcm: bytes) -> FrameStats:
        """Measure one 16 kHz frame and add it to the current turn."""
        stats = measure(pcm)
        if not stats.samples:
            return stats
        ms = stats.samples * 1000 / RATE
        self._levels.append(stats.level_db)
        self.floor_db = min(self._levels)
        speech = stats.level_db >= self.floor_db + self.speech_margin_db
        turn = self.turn
        if stats.clipped:
            turn.clipped_ms += ms
        if stats.zero_run == stats.samples:
            # digital silence is how many phone lines pause; a dropout is a short one mid-speech
            self._gap_ms += ms if self._after_speech else 0.0
            return stats
        if speech:
            turn.speech_ms += ms
            turn.level_db = max(turn.level_db, stats.level_db)
            if self._gap_ms <= self.max_gap_ms:
                turn.dropout_ms += self._gap_ms
            hole = stats.zero_run * 1000 / RATE
            turn.dropout_ms += hole if hole >= 10 else 0.0
        self._after_speech, self._gap_ms = speech, 0.0
        return stats

    def judge(self) -> TurnAudio:
        """Close the current turn: name its problems and start a new one."""
        turn, self.turn = self.turn, TurnAudio()
        turn.floor_db = self.floor_db
        if turn.clipped_ms and turn.clipped_ms >= (turn.speech_ms or 1) * self.clip_share:
            turn.problems.append("clipping")
        if turn.speech_ms and turn.level_db < self.quiet_db:
            turn.problems.append("too quiet")
        if turn.floor_db > self.noisy_db:
            turn.problems.append("noisy")
        if turn.dropout_ms >= self.dropout_ms:
            turn.problems.append("dropouts")
        self.counts.update(turn.problems)
        self.last_turn = turn
        return turn

    def started(self, session: VoiceSession) -> None:
        self._session = session

    def caller_audio(self, pcm: bytes) -> None:
        if not self._filtering:
            self.feed(pcm)

    def caller_said(self, text: str) -> bool | None:
        turn = self.judge()
        if not turn.problems or not self.ask_again or self.repeats >= self.max_repeats:
            self.repeats = 0
            return None
        session = self._session
        agent = session.agent if session is not None else None
        if agent is None:
            return None
        self.repeats += 1
        logger.info("caller audio: %s; asked to repeat", ", ".join(turn.problems))
        agent.say(self.repeat_prompt, record=False)
        return True

    async def ended(self) -> None:
        agent = self._session.agent if self._session is not None else None
        if agent is not None and self.counts:
            data: dict[str, Any] = {"side": "caller", **self.counts}
            emit(agent.executor.sink, make_event("custom.voice_quality", agent.ctx, **data))
