"""`VoiceSession`: our own voice pipeline, from raw audio to a `VoiceAgent` and back.

transport -> turn detector -> speech-to-text -> VoiceAgent -> text-to-speech -> transport

When the caller starts talking, playback pauses within one audio frame. The turn is cut
only once the speech is real: `confirm_ms` of speech, or words in the transcript. A cough,
or an "uh-huh" while the agent speaks, resumes playback where it paused.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import log_failure
from agent_adapter.tracing.events import emit, make_event
from agent_adapter.voice.media import STT, TTS, AudioTransport, StreamingSTT, STTStream
from agent_adapter.voice.pcm import Resampler
from agent_adapter.voice.player import Player
from agent_adapter.voice.playout import PlayoutBuffer, Segment, fire
from agent_adapter.voice.pump import EventPump
from agent_adapter.voice.turns import TurnDetector, TurnTimer

if TYPE_CHECKING:
    from agent_adapter.voice.agent import VoiceAgent

logger = logging.getLogger(__name__)

RATE = 16_000  # what turn detectors and speech-to-text get
BACKCHANNELS = frozenset(
    {"uh", "huh", "uh-huh", "mhm", "mm", "hmm", "mm-hmm", "mmhmm", "hm", "yeah", "yep", "yes"}
    | {"ok", "okay", "right", "sure", "got", "it", "i", "see"}
)


@dataclass
class TurnTiming:
    """When each step of one caller turn happened, in seconds on the monotonic clock."""

    started: float  # the detector heard speech start (playback paused here)
    confirmed: float | None = None  # the cut went to the agent
    ended: float | None = None  # the detector called the turn finished
    speech_end: float | None = None  # the last speech (ended minus the detector's quiet time)
    transcribed: float | None = None
    first_audio: float | None = None  # the agent's next audio started playing
    text: str = ""
    resumed: bool = False  # a false start: playback went on where it paused

    @property
    def voice_to_voice_ms(self) -> float | None:
        if self.first_audio is None or self.speech_end is None:
            return None
        return (self.first_audio - self.speech_end) * 1000

    @property
    def cut_ms(self) -> float | None:
        return None if self.confirmed is None else (self.confirmed - self.started) * 1000


@dataclass
class _Turn:
    timing: TurnTiming
    hushed: bool  # the agent was speaking when the caller started
    text: asyncio.Task[str]  # started when the caller went quiet, or when the turn ended


class VoiceSession:
    """Runs a `VoiceAgent` on an `AudioTransport`; give the agent `session.speaker`.

    `turns` gets 16 kHz audio. `confirm_ms` of speech confirms a cut before the transcript
    is ready; `backchannels` are words that never cut the agent while it speaks.

    Transcription starts early: a `StreamingSTT` hears the turn as it is spoken, and the
    final pass starts after `early_ms` of quiet, while the detector is still deciding
    whether the turn is over. If the caller goes on, that pass is dropped.
    """

    def __init__(
        self,
        transport: AudioTransport,
        *,
        stt: STT,
        tts: TTS,
        turns: TurnDetector,
        confirm_ms: float = 300,
        pre_roll_ms: float = 300,
        early_ms: float = 200,
        backchannels: frozenset[str] = BACKCHANNELS,
    ) -> None:
        self.transport = transport
        self.stt = stt
        self.turns = turns
        self.confirm_ms = confirm_ms
        self.early_ms = early_ms
        self.backchannels = backchannels
        self.buffer = PlayoutBuffer(transport.out_rate)
        self.speaker = Player(tts, self.buffer)
        self.timings: list[TurnTiming] = []
        self._timer = _timer_of(turns)
        self._pre_roll = int(RATE * pre_roll_ms / 1000) * 2
        self._recent = bytearray()
        self._turn: bytearray | None = None
        self._timing: TurnTiming | None = None
        self._speech_ms = 0.0
        self._hushed = False
        self._waiting_audio: TurnTiming | None = None
        self._stream: STTStream | None = None
        self._early: asyncio.Task[str] | None = None
        self._early_stream: STTStream | None = None
        self._texts: asyncio.Queue[_Turn] = asyncio.Queue()
        self._agent: VoiceAgent | None = None
        self._pump: EventPump | None = None
        self.buffer.on_start.append(self._audio_started)

    async def run(self, agent: VoiceAgent) -> None:
        """Run the call until the transport ends; then close the agent and the transport."""
        self._agent, self._pump = agent, EventPump(agent)
        resampler = Resampler(self.transport.in_rate, RATE)
        worker = asyncio.create_task(self._transcribe_turns())
        try:
            await self.transport.start(self.buffer)
            async for frame in self.transport.frames():
                self._frame(resampler.feed(frame))
            await self._texts.join()
            await self._pump.idle()
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            self._drop_early()
            stream, self._stream = self._stream, None
            if stream is not None:
                await stream.cancel()
            await self._pump.aclose()
            await agent.close()
            await self.transport.close()

    def _frame(self, pcm: bytes) -> None:
        if not pcm:
            return
        now = time.monotonic()
        self._recent += pcm
        del self._recent[: max(0, len(self._recent) - self._pre_roll)]
        if self._turn is not None:
            self._turn += pcm
            if self._stream is not None:
                self._stream.feed(pcm)
        for event in self.turns.feed(pcm):
            if event == "started":
                self._started(now)
            else:
                self._ended(now)
        timing = self._timing
        if timing is None or self._turn is None:
            return
        speech = self._timer is None or self._timer.quiet_ms == 0
        if timing.confirmed is None:
            self._speech_ms += len(pcm) * 500 / RATE if speech else 0
            if self._speech_ms >= self.confirm_ms:
                self._confirm(timing, now)
        if self._timer is None:
            return
        if speech and self._early is not None:
            self._drop_early()  # the caller went on: hear the whole turn again
            self._open_stream()
        elif not speech and self._early is None and self._timer.quiet_ms >= self.early_ms:
            self._early_stream = self._stream
            self._early = self._start_text()

    def _started(self, now: float) -> None:
        self._timing = TurnTiming(started=now)
        self.timings.append(self._timing)
        self._turn = bytearray(self._recent)
        self._speech_ms = 0.0
        self._hushed = self.speaker.speaking
        self.speaker.pause()
        self._drop_early()
        self._open_stream()

    def _ended(self, now: float) -> None:
        timing = self._timing
        if timing is None or self._turn is None:
            return
        text, self._early, self._early_stream = self._early or self._start_text(), None, None
        self._turn, self._timing = None, None
        timing.ended = now
        quiet = self._timer.quiet_ms if self._timer is not None else 0.0
        timing.speech_end = now - quiet / 1000
        self._texts.put_nowait(_Turn(timing, self._hushed, text))

    def _open_stream(self) -> None:
        if self._stream is not None:
            fire(self._stream.cancel())
            self._stream = None
        if isinstance(self.stt, StreamingSTT) and self._turn is not None:
            self._stream = self.stt.open(RATE)
            self._stream.feed(bytes(self._turn))

    def _start_text(self) -> asyncio.Task[str]:
        """Transcribe the turn so far: finish its stream, or send the audio at once."""
        audio, stream = bytes(self._turn or b""), self._stream
        self._stream = None
        return asyncio.ensure_future(self._text_of(audio, stream))

    def _drop_early(self) -> None:
        early, stream = self._early, self._early_stream
        self._early = self._early_stream = None
        if early is not None:
            early.cancel()
        if stream is not None:
            fire(stream.cancel())

    def _confirm(self, timing: TurnTiming, now: float) -> None:
        timing.confirmed = now
        if self._pump is not None:
            self._pump.started_speaking()

    async def _transcribe_turns(self) -> None:
        while True:
            turn = await self._texts.get()
            timing = turn.timing
            try:
                text = await turn.text
                timing.transcribed, timing.text = time.monotonic(), text
                self._deliver(timing, text, turn.hushed)
            except Exception as exc:
                log_failure(logger, logging.WARNING, exc, "speech-to-text failed")
                if timing.confirmed is None:
                    self._resume(timing)
            finally:
                self._texts.task_done()

    async def _text_of(self, audio: bytes, stream: STTStream | None) -> str:
        if stream is not None:
            try:
                return await stream.finish()
            except Exception as exc:  # fall back to the whole turn at once
                log_failure(logger, logging.WARNING, exc, "streaming speech-to-text failed")
        return await self.stt.transcribe(audio, RATE)

    def _deliver(self, timing: TurnTiming, text: str, hushed: bool) -> None:
        words = _words(text)
        if timing.confirmed is None:
            if not words or (hushed and self._backchannel(words)):
                self._resume(timing)
                return
            self._confirm(timing, time.monotonic())
        if words and self._pump is not None:
            self._waiting_audio = timing
            self._pump.said(text)

    def _backchannel(self, words: list[str]) -> bool:
        agent = self._agent
        if agent is not None and agent.approval is not None and agent.approval.waiting:
            return False  # "yeah" may be the answer to the question
        return " ".join(words) in self.backchannels or set(words) <= self.backchannels

    def _resume(self, timing: TurnTiming) -> None:
        timing.resumed = True
        if self._timing is None:  # no newer turn holds the pause
            self.speaker.resume()

    def _audio_started(self, segment: Segment) -> None:
        timing, self._waiting_audio = self._waiting_audio, None
        if timing is None or segment.started_at is None:
            return
        timing.first_audio = segment.started_at
        agent = self._agent
        if agent is not None:
            data = _event_data(timing)
            emit(agent.executor.sink, make_event("custom.voice_turn", agent.ctx, **data))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+(?:-[a-z']+)*", text.lower())


def _timer_of(turns: Any) -> TurnTimer | None:
    """The per-frame speech timer inside a detector (Silence, Silero, or one Smart Turn wraps)."""
    for owner in (turns, getattr(turns, "detector", None)):
        timer = getattr(owner, "timer", None)
        if isinstance(timer, TurnTimer):
            return timer
    return None


def _event_data(timing: TurnTiming) -> dict[str, Any]:
    """Timings only: the transcript is the caller's words and stays out of traces."""
    return {
        "voice_to_voice_ms": timing.voice_to_voice_ms,
        "cut_ms": timing.cut_ms,
        "resumed": timing.resumed,
    }


__all__ = ["BACKCHANNELS", "RATE", "TurnTiming", "VoiceSession"]
