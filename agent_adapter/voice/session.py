"""`VoiceSession`: our own voice pipeline, from raw audio to a `VoiceAgent` and back.

transport -> turn detector -> speech-to-text -> VoiceAgent -> text-to-speech -> transport

When the caller starts talking, playback pauses within one audio frame. The turn is cut
only once the speech is real: `confirm_ms` of speech, or words in the transcript. A cough,
or an "uh-huh" while the agent speaks, resumes playback where it paused.

With an `EchoGuard` (speakers with no echo cancelling), speech that starts while the agent
talks pauses nothing until it is loud enough or long enough; one that never is was the
agent's own voice in the mic and is dropped.

While the agent asks a spoken yes/no question, only a turn that began after the question
finished playing can answer it; earlier speech, an echo of the prompt included, is dropped.
Speech that starts while the agent cannot be cut (`VoiceAgent.interruptible`) is dropped too.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import log_failure
from agent_adapter.tracing.events import emit, make_event
from agent_adapter.voice.media import STT, TTS, AudioTransport, StreamingSTT, STTStream
from agent_adapter.voice.observers import CallObserver
from agent_adapter.voice.pcm import Resampler, rms
from agent_adapter.voice.player import Player
from agent_adapter.voice.playout import PlayoutBuffer, Segment, fire
from agent_adapter.voice.pump import EventPump
from agent_adapter.voice.timing import TurnTiming, event_data
from agent_adapter.voice.turns import EchoGuard, TurnDetector, TurnTimer

if TYPE_CHECKING:
    from agent_adapter.voice.agent import VoiceAgent

logger = logging.getLogger(__name__)

RATE = 16_000  # what turn detectors and speech-to-text get
BACKCHANNELS = frozenset(
    {"uh", "huh", "uh-huh", "mhm", "mm", "hmm", "mm-hmm", "mmhmm", "hm", "yeah", "yep", "yes"}
    | {"ok", "okay", "right", "sure", "got", "it", "i", "see"}
)


@dataclass
class _Turn:
    timing: TurnTiming
    hushed: bool  # the agent was speaking when the caller started
    text: asyncio.Task[str]  # started when the caller went quiet, or when the turn ended
    question: int | None = None  # the approval question already asked when the turn began


class VoiceSession:
    """Runs a `VoiceAgent` on an `AudioTransport`; give the agent `session.speaker`.

    `turns` gets 16 kHz audio. `confirm_ms` of speech confirms a cut before the transcript
    is ready; `backchannels` are words that never cut the agent while it speaks.

    Transcription starts early: a `StreamingSTT` hears the turn as it is spoken, and the
    final pass starts after `early_ms` of quiet, while the detector is still deciding
    whether the turn is over. If the caller goes on, that pass is dropped. With an agent
    `Speed`, the early transcript also starts the reply (`VoiceAgent.speculate`), first on
    the stream's interim words (a `partial` attribute) if it has them, and the agent's
    warm-up runs as the call starts. With `EagerTurns`, those words also move the turn's end.

    Keypad digits from a transport with `on_digit` (`TwilioStream`) reach the agent as
    `keypad_text`, once `keypad_wait_ms` passes with no new digit, at once on `#`, or
    after `max_digits`. A turn that never pauses ends after `max_turn_s`.

    `filters` transform the caller's 16 kHz audio first (a noise filter); `observers`
    (`CallObserver`) add call features such as `CallRecorder`, `IdlePolicy` and
    `VoicemailDetector`.
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
        echo_guard: EchoGuard | None = None,
        keypad_wait_ms: float = 1500,
        keypad_text: str = "(keypad) {digits}",
        max_turn_s: float = 30.0,
        max_digits: int = 32,
        filters: Sequence[Callable[[bytes], bytes]] = (),
        observers: Sequence[CallObserver] = (),
    ) -> None:
        self.transport = transport
        self.stt = stt
        self.turns = turns
        self.confirm_ms = confirm_ms
        self.early_ms = early_ms
        self.backchannels = backchannels
        self.echo_guard = echo_guard
        self.keypad_wait_ms = keypad_wait_ms
        self.keypad_text = keypad_text
        self.max_digits = max_digits
        self.filters = tuple(filters)
        self.observers = tuple(observers)
        self._max_turn = int(RATE * max_turn_s) * 2
        self.buffer = PlayoutBuffer(transport.out_rate)
        self.speaker = Player(tts, self.buffer)
        self.timings: list[TurnTiming] = []
        self._timer = _timer_of(turns)
        self._pre_roll = int(RATE * pre_roll_ms / 1000) * 2
        self._recent = bytearray()
        self._turn: bytearray | None = None
        self._timing: TurnTiming | None = None
        self._turn_question: int | None = None
        self._speech_ms = 0.0
        self._hushed = False
        self._guarded = False  # began over the agent and has not yet passed the echo guard
        self._loud_ms = 0.0
        self._digits = ""
        self._digit_timer: asyncio.TimerHandle | None = None
        self._waiting_audio: TurnTiming | None = None
        self._stream: STTStream | None = None
        self._early: asyncio.Task[str] | None = None
        self._early_stream: STTStream | None = None
        self._texts: asyncio.Queue[_Turn] = asyncio.Queue()
        self._agent: VoiceAgent | None = None
        self._pump: EventPump | None = None
        self._muted = False  # began while the agent could not be cut: dropped
        self._reader: asyncio.Task[None] | None = None
        self._ending: asyncio.Task[None] | None = None
        self.buffer.on_start.append(self._audio_started)
        self.buffer.on_pull.extend(observer.agent_audio for observer in self.observers)

    @property
    def agent(self) -> VoiceAgent | None:
        """The agent this session runs, once `run` started."""
        return self._agent

    @property
    def caller_talking(self) -> bool:
        """A caller turn is open."""
        return self._turn is not None

    def send_text(self, text: str) -> None:
        """A typed message from the caller, handled like a spoken turn."""
        if self._pump is not None:
            self._pump.said(text)

    async def hang_up(self, *, after_speech: bool = False) -> None:
        """End the call; with `after_speech`, once the current reply has played."""
        agent = self._agent
        if after_speech and agent is not None:
            await agent.wait()
            await agent.wait_spoken()
            while self.buffer.busy:
                await asyncio.sleep(0.02)
        end = getattr(self.transport, "hang_up", None)
        try:
            if callable(end) and inspect.isawaitable(result := end()):
                await result
        finally:
            reader = self._reader
            if reader is not None:
                reader.cancel()

    def end_after_reply(self) -> None:
        """Hang up once the agent's current reply has played (an `end_call` tool)."""
        if self._ending is None or self._ending.done():
            self._ending = asyncio.ensure_future(self.hang_up(after_speech=True))

    async def run(self, agent: VoiceAgent) -> None:
        """Run the call until the transport ends; then close the agent and the transport."""
        self._agent, self._pump = agent, EventPump(agent)
        resampler = Resampler(self.transport.in_rate, RATE)
        worker = asyncio.create_task(self._transcribe_turns())
        on_digit = getattr(self.transport, "on_digit", None)
        if isinstance(on_digit, list):
            on_digit.append(self._digit)
        for observer in self.observers:
            observer.started(self)
        warming = asyncio.ensure_future(agent.warm_up((self.stt, self.speaker.tts)))
        try:
            await self.transport.start(self.buffer)
            self._reader = asyncio.create_task(self._read(resampler))
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            if not self._reader.cancelled():  # after a hang-up, the last words are moot
                await self._texts.join()
                await self._pump.idle()
        finally:
            worker.cancel()
            warming.cancel()
            ending = self._ending
            if ending is not None and ending is not asyncio.current_task():
                ending.cancel()
            if self._digit_timer is not None:
                self._digit_timer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            self._drop_early()
            stream, self._stream = self._stream, None
            if stream is not None:
                await stream.cancel()
            await self._pump.aclose()
            await agent.close()
            await self.transport.close()
            for observer in self.observers:
                try:
                    await observer.ended()
                except Exception as exc:
                    log_failure(logger, logging.WARNING, exc, "call observer failed")

    async def _read(self, resampler: Resampler) -> None:
        async for frame in self.transport.frames():
            self._frame(resampler.feed(frame))

    def _frame(self, pcm: bytes) -> None:
        if not pcm:
            return
        for step in self.filters:
            pcm = step(pcm)
        for observer in self.observers:
            observer.caller_audio(pcm)
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
        if self._turn is not None and len(self._turn) >= self._max_turn:
            self._ended(now)  # bounds memory and the speech-to-text request
            return
        timing = self._timing
        if timing is None or self._turn is None or self._muted:
            return
        speech = self._timer is None or self._timer.quiet_ms == 0
        ms = len(pcm) * 500 / RATE
        if timing.confirmed is None:
            self._speech_ms += ms if speech else 0
        if self._guarded and not self._passes_guard(pcm, ms):
            return
        if timing.confirmed is None and self._speech_ms >= self.confirm_ms:
            self._confirm(timing, now)
        if self._timer is None:
            return
        if speech and self._early is not None:
            self._drop_early()  # the caller went on: hear the whole turn again
            self._open_stream()
        elif not speech and self._early is None and self._timer.quiet_ms >= self.early_ms:
            partial = getattr(self._stream, "partial", "")
            self._early_stream = self._stream
            self._early = self._start_text()
            timing.early = now
            self._early.add_done_callback(self._early_text)
            if isinstance(partial, str) and partial.strip():
                self._think(partial)  # the STT's interim words: no wait for the final ones

    def _started(self, now: float) -> None:
        self._timing = TurnTiming(started=now)
        self._turn = bytearray(self._recent)
        self._turn_question = self._question()[1]
        self._speech_ms = self._loud_ms = 0.0
        self._drop_early()
        agent = self._agent
        self._muted = agent is not None and not agent.interruptible
        if self._muted:
            return
        self._guarded = self.echo_guard is not None and self.speaker.speaking
        if not self._guarded:
            self._barge_in(self._timing)

    def _barge_in(self, timing: TurnTiming) -> None:
        self.timings.append(timing)
        self._hushed = self.speaker.speaking
        self.speaker.pause()
        self._open_stream()

    def _passes_guard(self, pcm: bytes, ms: float) -> bool:
        guard, timing = self.echo_guard, self._timing
        if guard is None or timing is None:
            return True
        if rms(pcm) >= guard.level:
            self._loud_ms += ms
        if self._loud_ms < guard.loud_ms and self._speech_ms < guard.speech_ms:
            return False
        self._guarded = False
        self._barge_in(timing)
        return True

    def _ended(self, now: float) -> None:
        timing = self._timing
        if timing is None or self._turn is None:
            return
        if self._guarded or self._muted:  # its own voice, or speech while it could not be cut
            self._guarded = self._muted = False
            self._turn, self._timing = None, None
            return
        text, self._early, self._early_stream = self._early or self._start_text(), None, None
        self._turn, self._timing = None, None
        timing.ended = now
        quiet = self._timer.quiet_ms if self._timer is not None else 0.0
        timing.speech_end = now - quiet / 1000
        self._texts.put_nowait(_Turn(timing, self._hushed, text, self._turn_question))
        real = timing.confirmed is not None and not self._hushed and not self._question()[0]
        if real and self._pump is not None:
            self._pump.turn_ended()

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

    def _early_text(self, task: asyncio.Task[str]) -> None:
        live = task is self._early and not task.cancelled() and task.exception() is None
        if live:
            self._think(task.result())

    def _think(self, text: str) -> None:
        """Early words: start the reply on them, and let `EagerTurns` judge the turn's end."""
        hint = getattr(self.turns, "hint", None)
        if callable(hint):
            hint(text)
        if self._pump is not None:
            self._pump.speculate(text)

    def _drop_early(self) -> None:
        early, stream = self._early, self._early_stream
        self._early = self._early_stream = None
        if early is not None:
            early.cancel()
            if self._pump is not None:
                self._pump.drop_speculation()
        if stream is not None:
            fire(stream.cancel())

    def _confirm(self, timing: TurnTiming, now: float) -> None:
        timing.confirmed = now
        if self._pump is not None and not self._question()[0]:
            self._pump.started_speaking()  # during a question it goes with the words instead

    async def _transcribe_turns(self) -> None:
        while True:
            turn = await self._texts.get()
            timing = turn.timing
            try:
                text = await turn.text
                timing.transcribed, timing.text = time.monotonic(), text
                if self._observed(text):
                    if timing.confirmed is None:
                        self._resume(timing)
                    continue
                self._deliver(turn, text)
            except Exception as exc:
                log_failure(logger, logging.WARNING, exc, "speech-to-text failed")
                if timing.confirmed is None:
                    self._resume(timing)
            finally:
                self._texts.task_done()

    def _observed(self, text: str) -> bool:
        """Observers see each transcript; True if one keeps it from the agent."""
        if not text.strip():
            return False
        kept = False
        for observer in self.observers:
            try:
                kept = bool(observer.caller_said(text)) or kept
            except Exception as exc:
                log_failure(logger, logging.WARNING, exc, "call observer failed")
        return kept

    async def _text_of(self, audio: bytes, stream: STTStream | None) -> str:
        if stream is not None:
            try:
                return await stream.finish()
            except Exception as exc:  # fall back to the whole turn at once
                log_failure(logger, logging.WARNING, exc, "streaming speech-to-text failed")
        return await self.stt.transcribe(audio, RATE)

    def _deliver(self, turn: _Turn, text: str) -> None:
        timing, words = turn.timing, _words(text)
        if timing.confirmed is None:
            if not words or (turn.hushed and self._backchannel(words)):
                self._resume(timing)
                return
            self._confirm(timing, time.monotonic())
        if not words or self._pump is None:
            return
        asking, question = self._question()
        if asking:
            if turn.question is None or turn.question != question:
                self._resume(timing)  # began before the question had played: not an answer
                return
            self._pump.started_speaking()
        self._waiting_audio = timing
        self._pump.said(text, timing)

    def _question(self) -> tuple[bool, int | None]:
        """Whether the agent's question is open, and its id once its prompt has played."""
        agent = self._agent
        if agent is None or agent.approval is None:
            return False, None
        run_id = agent.ctx.run_id
        if not agent.approval.waiting_for(run_id):
            return False, None
        return True, agent.approval.armed_question(run_id)

    def _backchannel(self, words: list[str]) -> bool:
        if self._question()[0]:
            return False  # "yeah" may be the answer to the question
        return " ".join(words) in self.backchannels or set(words) <= self.backchannels

    def _resume(self, timing: TurnTiming) -> None:
        timing.resumed = True
        if self._timing is None or self._guarded:  # no newer turn holds the pause
            self.speaker.resume()
        if self._timing is None and self._pump is not None:
            self._pump.drop_speculation()  # a false start: its reply is moot

    def _digit(self, digit: str) -> None:
        if self._pump is None:
            return
        if not self._digits:
            self._pump.started_speaking()
        self._digits += digit
        if self._digit_timer is not None:
            self._digit_timer.cancel()
            self._digit_timer = None
        if digit == "#" or len(self._digits) >= self.max_digits:
            self._send_digits()
        else:
            loop = asyncio.get_running_loop()
            self._digit_timer = loop.call_later(self.keypad_wait_ms / 1000, self._send_digits)

    def _send_digits(self) -> None:
        digits, self._digits, self._digit_timer = self._digits, "", None
        if digits and self._pump is not None:
            self._pump.said(self.keypad_text.format(digits=digits))

    def _audio_started(self, segment: Segment) -> None:
        timing, self._waiting_audio = self._waiting_audio, None
        if timing is None or segment.started_at is None:
            return
        timing.first_audio = segment.started_at
        agent = self._agent
        if agent is not None:
            data = event_data(timing)
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


__all__ = ["BACKCHANNELS", "RATE", "TurnTiming", "VoiceSession"]
