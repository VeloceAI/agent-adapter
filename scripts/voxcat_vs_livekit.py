"""LiveKit's side of voxcat_vs.py: the real `AgentSession` pipeline on the shared fake parts."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import voxcat_vs_common as vc
from livekit import rtc
from livekit.agents import Agent, AgentSession, llm, stt, tts, vad
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.voice import io
from voxcat_vs_common import FRAME_BYTES, RATE, Delays, Scenario, Setup, Trial

# AgentSession's default VAD is the bundled `inference.VAD(model="silero")`
# (livekit/agents/inference/vad.py): min_silence_duration=0.25, min_speech_duration=0.05.
# livekit-plugins-silero is not installed here, and the session does not use it.
DEFAULT_SILENCE_S = 0.25
MIN_SPEECH_S = 0.05
BARGE_WORDS = ["Hold", "on."]  # what the caller says when cutting in past the script
EPS = 1e-6
Event = stt.SpeechEventType


class CallerInput(io.AudioInput):
    def __init__(self) -> None:
        super().__init__(label="caller")
        self.queue: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue()

    def push(self, pcm: bytes) -> None:
        self.queue.put_nowait(rtc.AudioFrame(pcm, RATE, 1, len(pcm) // 2))

    async def __anext__(self) -> rtc.AudioFrame:
        return await self.queue.get()


class Speaker(io.AudioOutput):
    """A leaf sink like RoomIO's: buffers whatever LiveKit pushes, plays it in real time."""

    def __init__(self, playout: vc.Playout) -> None:
        caps = io.AudioOutputCapabilities(pause=True)
        super().__init__(label="speaker", capabilities=caps, sample_rate=RATE)
        self.playout = playout
        self._chunks: deque[bytes] = deque()
        self._wake, self._idle, self._enabled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self._idle.set()
        self._enabled.set()
        self._interrupted = asyncio.Event()
        self._pushed = self._played = 0.0
        self._segment = 0  # a chunk still playing after a cut must not count toward the next
        self._started = False
        self._flush_task: asyncio.Task[None] | None = None
        self._player = asyncio.create_task(self._play())

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if self._flush_task is not None and not self._flush_task.done():
            await self._flush_task
        data = bytes(frame.data)
        self._chunks.extend(data[i : i + FRAME_BYTES] for i in range(0, len(data), FRAME_BYTES))
        self._pushed += frame.duration
        self._idle.clear()
        self._wake.set()

    def flush(self) -> None:
        super().flush()
        if not self._pushed:
            return
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = asyncio.create_task(self._wait_for_playout())

    def clear_buffer(self) -> None:
        self._chunks.clear()
        if self._pushed:
            self._interrupted.set()

    def pause(self) -> None:
        super().pause()
        self._enabled.clear()

    def resume(self) -> None:
        super().resume()
        self._enabled.set()

    async def _play(self) -> None:
        while True:
            await self._wake.wait()
            await self._enabled.wait()
            if not self._chunks:
                self._wake.clear()
                self._idle.set()
                continue
            chunk, segment = self._chunks.popleft(), self._segment
            if not self._started:
                self._started = True
                self.on_playback_started(created_at=time.time())
            await self.playout.play(chunk)
            if segment == self._segment:
                self._played += len(chunk) / 2 / RATE

    async def _wait_for_playout(self) -> None:
        async def drained() -> None:
            while self._chunks or not self._idle.is_set():
                await self._idle.wait()
                await asyncio.sleep(0)

        done = asyncio.ensure_future(drained())
        cut = asyncio.ensure_future(self._interrupted.wait())
        try:
            await asyncio.wait([done, cut], return_when=asyncio.FIRST_COMPLETED)
        finally:
            done.cancel()
            cut.cancel()
        interrupted = self._interrupted.is_set()
        played = min(self._played, self._pushed) if interrupted else self._pushed
        self._pushed = self._played = 0.0
        self._segment += 1
        self._started = False
        self._interrupted.clear()
        self.on_playback_finished(playback_position=played, interrupted=interrupted)

    async def aclose(self) -> None:
        for task in (self._flush_task, self._player):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


class EnergyVAD(vad.VAD):
    def __init__(self, min_silence_s: float) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=vc.FRAME_MS / 1000))
        self.min_silence_s = min_silence_s

    def stream(self) -> vad.VADStream:
        return EnergyVADStream(self)


class EnergyVADStream(vad.VADStream):
    """inference.VAD's event bookkeeping, with `is_speech` energy in place of silero."""

    def _reset(self) -> None:
        self.speaking, self.speech, self.silence = False, 0.0, 0.0
        self.raw_speech = self.raw_silence = 0.0

    def _send(self, kind: vad.VADEventType, frame: rtc.AudioFrame, p: float = 0.0) -> None:
        ev = vad.VADEvent(kind, self.samples, self.t, self.speech, self.silence, [frame], p)
        ev.speaking, ev.raw_accumulated_silence = self.speaking, self.raw_silence
        ev.raw_accumulated_speech = self.raw_speech
        self._event_ch.send_nowait(ev)

    async def _main_task(self) -> None:
        min_silence = self._vad.min_silence_s  # type: ignore[attr-defined]
        self._reset()
        self.t, self.samples = 0.0, 0
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                self._reset()
                continue
            step, loud = item.duration, vc.is_speech(bytes(item.data))
            self.t, self.samples = self.t + step, self.samples + item.samples_per_channel
            if self.speaking:
                self.speech += step
            else:
                self.silence += step
            self._send(vad.VADEventType.INFERENCE_DONE, item, float(loud))  # raw: before update
            if loud:
                self.raw_speech, self.raw_silence = self.raw_speech + step, 0.0
                if not self.speaking and self.raw_speech >= MIN_SPEECH_S - EPS:
                    self.speaking, self.silence, self.speech = True, 0.0, self.raw_speech
                    self._send(vad.VADEventType.START_OF_SPEECH, item)
            else:
                self.raw_silence, self.raw_speech = self.raw_silence + step, 0.0
                if self.speaking and self.raw_silence >= min_silence - EPS:
                    self.speaking, self.silence = False, self.raw_silence
                    self.speech = max(0.0, self.speech - self.raw_silence)
                    self._send(vad.VADEventType.END_OF_SPEECH, item)
                    self.speech = 0.0


def said(kind: Event, text: str = "", **fields: object) -> stt.SpeechEvent:
    words = [stt.SpeechData(language="en", text=text, confidence=1.0)] if text else []  # type: ignore[arg-type]
    return stt.SpeechEvent(type=kind, alternatives=words, **fields)  # type: ignore[arg-type]


class FakeSTT(stt.STT):
    # No manual_flush, like LiveKit's own inference.STT: LiveKit then never flushes the stream
    # and the final words come from the STT's own endpoint.
    def __init__(self, scenario: Scenario, delays: Delays, flux: bool) -> None:
        caps = stt.STTCapabilities(streaming=True, interim_results=True, offline_recognize=False)
        super().__init__(capabilities=caps)
        self.scenario, self.delays, self.flux = scenario, delays, flux

    async def _recognize_impl(self, buffer: object, **_: object) -> stt.SpeechEvent:
        return said(Event.FINAL_TRANSCRIPT, self.scenario.transcript)

    def stream(self, **kw: Any) -> FakeSTTStream:
        return FakeSTTStream(self, kw.get("conn_options", DEFAULT_API_CONNECT_OPTIONS))


class FakeSTTStream(stt.RecognizeStream):
    def __init__(self, owner: FakeSTT, conn_options: APIConnectOptions) -> None:
        self._owner = owner
        self._timers: list[asyncio.Task[None]] = []
        self._speech_ms, self._final_words = 0.0, 0
        self._last_speech = self._last_speech_wall = 0.0
        super().__init__(stt=owner, conn_options=conn_options)

    def _words(self) -> list[str]:
        scripted = vc.speech_ms(self._owner.scenario)
        words = vc.heard(self._owner.scenario, min(self._speech_ms, scripted)).split()
        return words + (BARGE_WORDS if self._speech_ms > scripted + EPS else [])

    def _pending(self) -> str:
        return " ".join(self._words()[self._final_words :])

    def _send(self, kind: Event, text: str = "", **fields: object) -> None:
        self._event_ch.send_nowait(said(kind, text, **fields))

    def _later(self, ms: float, fn: Callable[[], None]) -> None:
        async def fire() -> None:
            await asyncio.sleep(max(0.0, self._last_speech + ms / 1000 - time.perf_counter()))
            fn()

        self._timers.append(asyncio.create_task(fire()))

    def _cancel_timers(self) -> None:
        for task in self._timers:
            task.cancel()
        self._timers.clear()

    def _preflight(self) -> None:
        if text := self._pending():
            self._send(Event.PREFLIGHT_TRANSCRIPT, text)

    def _finalize(self, ended_at: float | None = None) -> None:
        if text := self._pending():
            self._final_words = len(self._words())
            self._send(Event.FINAL_TRANSCRIPT, text)
        if ended_at is not None:  # Flux-like: the STT itself says the turn ended
            self._send(Event.END_OF_SPEECH, speech_end_time=ended_at)

    async def _run(self) -> None:
        delays, flux, talking, interim = self._owner.delays, self._owner.flux, False, ""
        try:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    self._last_speech = time.perf_counter()
                    self._later(delays.stt_ms, self._finalize)
                elif vc.is_speech(bytes(item.data)):
                    if not talking:
                        talking = True
                        self._cancel_timers()
                        if flux:
                            self._send(Event.START_OF_SPEECH)
                    self._speech_ms += item.duration * 1000
                    self._last_speech, self._last_speech_wall = time.perf_counter(), time.time()
                    if (text := self._pending()) != interim:
                        interim = text
                        self._send(Event.INTERIM_TRANSCRIPT, text)
                elif talking:
                    talking, interim = False, ""
                    if flux:
                        end = functools.partial(self._finalize, self._last_speech_wall)
                        self._later(delays.flux_eager_ms, self._preflight)
                        self._later(delays.flux_end_ms, end)
                    else:
                        self._later(delays.stt_endpoint_ms, self._finalize)
        finally:
            self._cancel_timers()


class FakeLLM(llm.LLM):
    def __init__(self, delays: Delays, counters: vc.Counters) -> None:
        super().__init__()
        self.delays, self.counters = delays, counters

    def chat(self, *, chat_ctx: llm.ChatContext, **kw: Any) -> llm.LLMStream:
        options = kw.get("conn_options", DEFAULT_API_CONNECT_OPTIONS)
        return FakeLLMStream(self, chat_ctx=chat_ctx, tools=[], conn_options=options)


class FakeLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        owner: FakeLLM = self._llm  # type: ignore[assignment]
        async for token in vc.reply_tokens(owner.delays, owner.counters):
            delta = llm.ChoiceDelta(role="assistant", content=token)
            self._event_ch.send_nowait(llm.ChatChunk(id="fake", delta=delta))


class FakeTTS(tts.TTS):
    """Not streaming: AgentSession wraps it in tts.StreamAdapter (blingfire sentences)."""

    def __init__(self, delays: Delays) -> None:
        caps = tts.TTSCapabilities(streaming=False)
        super().__init__(capabilities=caps, sample_rate=RATE, num_channels=1)
        self.delays = delays

    def synthesize(self, text: str, **kw: Any) -> FakeChunks:
        options = kw.get("conn_options", DEFAULT_API_CONNECT_OPTIONS)
        return FakeChunks(tts=self, input_text=text, conn_options=options)


class FakeChunks(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        delays: Delays = self._tts.delays  # type: ignore[attr-defined]
        pcm = {"sample_rate": RATE, "num_channels": 1, "mime_type": "audio/pcm"}
        output_emitter.initialize(request_id="fake", frame_size_ms=vc.FRAME_MS, **pcm)  # type: ignore[arg-type]
        await asyncio.sleep(delays.tts_first_ms / 1000)
        output_emitter.push(vc.tts_audio(self.input_text, delays))


def build_session(
    setup: Setup, scenario: Scenario, delays: Delays, counters: vc.Counters, notes: list[str]
) -> AgentSession:
    # mode "vad" is what LiveKit picks anyway outside dev mode and LiveKit Cloud
    turns: dict[str, object] = {"interruption": {"mode": "vad"}}
    if setup.provider_tricks:
        turns |= {"turn_detection": "stt", "preemptive_generation": {"enabled": True}}
        notes.append("turn_detection=stt: STT START/PREFLIGHT/FINAL/END like Flux; min_delay 0.5")
    elif setup.end_ms is not None:
        turns |= {"turn_detection": "vad", "endpointing": {"min_delay": 0.0}}
        notes.append(f"turn_detection=vad, VAD silence {setup.end_ms / 1000:g} s, min_delay 0")
    else:
        # LiveKit's real default is its EOU turn-detector model; it is not run here
        turns["turn_detection"] = "vad"
        notes.append("turn_detection=vad, VAD silence 0.25 s, min_delay 0.5 s (no EOU model)")
    notes.append("aec_warmup off (its 3 s default would ignore the barge-in); preemptive LLM on")
    silence = setup.end_ms / 1000 if setup.end_ms is not None else DEFAULT_SILENCE_S
    return AgentSession(
        stt=FakeSTT(scenario, delays, flux=setup.provider_tricks),
        vad=EnergyVAD(silence),
        llm=FakeLLM(delays, counters),
        tts=FakeTTS(delays),
        turn_handling=turns,  # type: ignore[arg-type]
        aec_warmup_duration=None,
    )


async def run_trial(setup: Setup, scenario: Scenario, delays: Delays) -> Trial:
    trial = Trial("livekit", setup.name, scenario.name)
    counters, playout, marks = vc.Counters(), vc.Playout(), {}
    barge = asyncio.Event() if scenario.barge_after_ms is not None else None

    async def cut_in() -> None:
        await playout.on_first.wait()
        await asyncio.sleep((scenario.barge_after_ms or 0) / 1000)
        assert barge is not None
        barge.set()

    session: AgentSession | None = None
    speaker: Speaker | None = None
    watcher = asyncio.create_task(cut_in()) if barge is not None else None
    cpu = time.process_time()
    try:
        caller, speaker = CallerInput(), Speaker(playout)
        session = build_session(setup, scenario, delays, counters, trial.notes)
        session.input.audio, session.output.audio = caller, speaker
        await session.start(Agent(instructions="Take restaurant bookings."))
        cpu = time.process_time()  # like voxcat_vs_voxcat: CPU while the caller talks
        await vc.send_caller(scenario, caller.push, marks, barge)
    except Exception as exc:
        trial.error = repr(exc)
    frames = sum(1 for _ in vc.caller_frames(scenario))
    trial.cpu_us_per_frame = round((time.process_time() - cpu) * 1e6 / frames, 1)
    if watcher is not None:
        watcher.cancel()
    try:
        if session is not None:
            await session.aclose()
        if speaker is not None:
            await speaker.aclose()
    except Exception as exc:
        trial.error = trial.error or f"close: {exc!r}"
    trial.llm_calls, trial.tokens = counters.llm_calls, counters.tokens
    return vc.score(trial, marks, playout)


if __name__ == "__main__":
    from voxcat_vs_common import SCENARIOS, SETUPS

    logging.getLogger("livekit").setLevel(logging.ERROR)

    async def main() -> None:
        for setup in SETUPS:
            for scenario in SCENARIOS:
                print(await run_trial(setup, scenario, Delays()))
        left = [t.get_name() for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        print(f"tasks left after all trials: {len(left)}", left[:5])

    asyncio.run(main())
