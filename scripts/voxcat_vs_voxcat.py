"""VoxCat's side of voxcat_vs.py: `VoiceSession` + `VoiceAgent` on the shared fake parts.

No provider tricks ever: plain streaming STT (interim words, then final ones), plain TTS.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator

from voxcat_vs_common import (
    FRAME_BYTES,
    REPLY,
    Counters,
    Delays,
    Playout,
    Scenario,
    Setup,
    Trial,
    caller_frames,
    heard,
    is_speech,
    reply_tokens,
    score,
    send_caller,
    tts_audio,
)

from agent_adapter import CallableModel, Executor, LLMRequest, LLMResponse, Message, Registry
from agent_adapter.models import StreamDone, TextDelta
from agent_adapter.voice import VoiceAgent, VoiceSession
from agent_adapter.voice.endpoint import EagerTurns
from agent_adapter.voice.playout import PlayoutBuffer
from agent_adapter.voice.speed import Speed
from agent_adapter.voice.turns import SilenceTurns, TurnDetector

EARLY_MS = 100  # quiet before the early pass: interim words start the reply


class LiveTransport:
    """Caller frames from a queue; the agent's audio drained into `Playout` in real time."""

    in_rate = out_rate = 16_000

    def __init__(self, playout: Playout) -> None:
        self.playout = playout
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._drain: asyncio.Task[None] | None = None

    async def frames(self) -> AsyncIterator[bytes]:
        while (frame := await self.queue.get()) is not None:
            yield frame

    async def start(self, buffer: PlayoutBuffer) -> None:
        self._drain = asyncio.create_task(self._play(buffer))

    async def _play(self, buffer: PlayoutBuffer) -> None:
        while True:
            await self.playout.play(buffer.pull(FRAME_BYTES))

    async def close(self) -> None:
        if self._drain is not None:
            self._drain.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain


def speech_in(pcm: bytes) -> float:
    return sum(20.0 for i in range(0, len(pcm), FRAME_BYTES) if is_speech(pcm[i : i + FRAME_BYTES]))


class FakeStream:
    def __init__(self, stt: FakeSTT) -> None:
        self.stt = stt
        self.ms = 0.0

    def feed(self, pcm: bytes) -> None:
        self.ms += speech_in(pcm)

    @property
    def partial(self) -> str:
        return heard(self.stt.scenario, self.ms) if self.ms else ""

    async def finish(self) -> str:
        await asyncio.sleep(self.stt.delays.stt_ms / 1000)
        return heard(self.stt.scenario, self.ms)

    async def cancel(self) -> None:
        return None


class FakeSTT:
    def __init__(self, scenario: Scenario, delays: Delays) -> None:
        self.scenario, self.delays = scenario, delays

    def open(self, sample_rate: int) -> FakeStream:
        return FakeStream(self)

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        await asyncio.sleep(self.delays.stt_ms / 1000)
        return heard(self.scenario, speech_in(pcm))


class FakeTTS:
    sample_rate = 16_000

    def __init__(self, delays: Delays) -> None:
        self.delays = delays

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        await asyncio.sleep(self.delays.tts_first_ms / 1000)
        audio = tts_audio(text, self.delays)
        for i in range(0, len(audio), FRAME_BYTES * 10):
            yield audio[i : i + FRAME_BYTES * 10]


def fake_model(delays: Delays, counters: Counters) -> CallableModel:
    async def stream(request: LLMRequest) -> AsyncIterator[TextDelta | StreamDone]:
        async for token in reply_tokens(delays, counters):
            yield TextDelta(text=token)
        message = Message(role="assistant", content=REPLY)
        done = LLMResponse(provider="fake", model="fake", text=REPLY, message=message)
        yield StreamDone(response=done)

    return CallableModel(lambda request: REPLY, provider="fake", model="fake", stream_fn=stream)


def turns_for(setup: Setup) -> TurnDetector:
    if setup.end_ms is not None:
        return SilenceTurns(start_ms=60, end_ms=setup.end_ms)
    return EagerTurns(SilenceTurns(start_ms=60, end_ms=600), done_ms=250)


async def run_trial(
    setup: Setup, scenario: Scenario, delays: Delays, first_words: int | None = 4
) -> Trial:
    trial = Trial("voxcat", setup.name, scenario.name)
    counters, playout, marks = Counters(), Playout(), {}
    transport = LiveTransport(playout)
    session = VoiceSession(
        transport,
        stt=FakeSTT(scenario, delays),
        tts=FakeTTS(delays),
        turns=turns_for(setup),
        early_ms=EARLY_MS,
    )
    agent = VoiceAgent(
        fake_model(delays, counters),
        Executor(Registry()),
        session.speaker,
        speed=Speed(first_words=first_words),
    )
    run = asyncio.create_task(session.run(agent))
    barge = asyncio.Event() if scenario.barge_after_ms is not None else None

    async def cut_in() -> None:
        await playout.on_first.wait()
        await asyncio.sleep((scenario.barge_after_ms or 0) / 1000)
        assert barge is not None
        barge.set()

    watcher = asyncio.create_task(cut_in()) if barge is not None else None
    cpu = time.process_time()
    try:
        await send_caller(scenario, transport.queue.put_nowait, marks, barge)
    except Exception as exc:  # a trial reports, never stops the run
        trial.error = repr(exc)
    frames = sum(1 for _ in caller_frames(scenario))
    trial.cpu_us_per_frame = round((time.process_time() - cpu) * 1e6 / frames, 1)
    if watcher is not None:
        watcher.cancel()
    await session.hang_up()
    with contextlib.suppress(asyncio.CancelledError, TimeoutError):
        await asyncio.wait_for(run, 5)
    trial.llm_calls, trial.tokens = counters.llm_calls, counters.tokens
    return score(trial, marks, playout)


if __name__ == "__main__":
    from voxcat_vs_common import SCENARIOS, SETUPS

    async def main() -> None:
        for setup in SETUPS:
            for scenario in SCENARIOS:
                print(await run_trial(setup, scenario, Delays()))

    asyncio.run(main())
