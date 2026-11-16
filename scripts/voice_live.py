"""voice-live: our own voice pipeline end to end, no Pipecat, no LiveKit, a cut mid-booking.

    python scripts/voice_live.py                  # scripted caller, agent on the speakers
    python scripts/voice_live.py --mic 40         # you are the caller (use headphones)
    python scripts/voice_live.py --webrtc         # call it from a browser on this machine

Pipeline: audio -> SmartTurn(SileroTurns) -> MoonshineSTT -> VoiceAgent -> MoonshineTTS ->
playout buffer -> the transport. Needs agent-adapter[voice-local,vad] (and [webrtc,server]
for --webrtc); give --silero and --smart-turn, or have pipecat-ai installed for its copies.

Only the model is scripted. The caller asks for a booking, then talks over the agent while
the booking is half-way and asks "did that go through?"; the script exits non-zero unless
the booking was charged exactly once.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from voice_live_common import (
    LINES,
    Bookings,
    Measure,
    caller_lines,
    caller_script,
    log,
    model_paths,
    registry,
    report,
    scripted_model,
)

from agent_adapter.exec import Executor
from agent_adapter.voice import STT, CallableSTT, LocalAudio, VoiceAgent, VoiceSession
from agent_adapter.voice.moonshine import MoonshineSTT, MoonshineTTS
from agent_adapter.voice.playout import PlayoutBuffer
from agent_adapter.voice.silero import SileroTurns
from agent_adapter.voice.smart_turn import RATE, SmartTurn

OUT_RATE = 24_000


class ScriptedLine:
    """An `AudioTransport`: the scripted caller in, the agent out to the speakers (or a
    silent clock that pulls 20 ms every 20 ms, as a sound card would)."""

    in_rate = RATE
    out_rate = OUT_RATE

    def __init__(self, measure: Measure, *, silent: bool) -> None:
        self.measure = measure
        self.silent = silent
        self.frames_in: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._jobs: list[Any] = []

    async def send(self, pcm: bytes) -> None:
        self.frames_in.put_nowait(pcm)

    async def frames(self) -> AsyncIterator[bytes]:
        while (frame := await self.frames_in.get()) is not None:
            yield frame

    async def start(self, buffer: PlayoutBuffer) -> None:
        block = OUT_RATE * 2 // 50
        if not self.silent:
            import sounddevice

            def play(out: Any, frames: int, when: Any, status: Any) -> None:
                pcm = buffer.pull(len(out))
                self.measure.hear(pcm)
                out[:] = pcm

            stream = sounddevice.RawOutputStream(
                samplerate=OUT_RATE, blocksize=block // 2, channels=1, dtype="int16", callback=play
            )
            stream.start()
            self._jobs.append(stream)
            return

        async def clock() -> None:
            due = time.monotonic()
            while True:
                self.measure.hear(buffer.pull(block))
                due += 0.02
                await asyncio.sleep(max(0.0, due - time.monotonic()))

        self._jobs.append(asyncio.create_task(clock()))

    async def close(self) -> None:
        for job in self._jobs:
            if isinstance(job, asyncio.Task):
                job.cancel()
            else:
                job.stop()
                job.close()


def session_for(transport: Any, args: argparse.Namespace) -> VoiceSession:
    silero, smart = model_paths(args.silero, args.smart_turn)
    stt: STT = MoonshineSTT(args.stt_model)
    if args.batch_stt:  # transcribe each turn once it ends (still started early)
        stt = CallableSTT(stt.transcribe)
    return VoiceSession(
        transport,
        stt=stt,
        tts=MoonshineTTS(),
        turns=SmartTurn(smart, SileroTurns(silero)),
    )


def agent_for(session: VoiceSession, bookings: Bookings) -> VoiceAgent:
    executor = Executor(registry(bookings))
    return VoiceAgent(scripted_model(), executor, session.speaker, instructions="You book shows.")


async def scripted(args: argparse.Namespace) -> int:
    measure, bookings = Measure(), Bookings()
    line = ScriptedLine(measure, silent=args.silent)
    session = session_for(line, args)
    agent = agent_for(session, bookings)
    measure.watch(agent)
    lines = caller_lines()
    call = asyncio.create_task(session.run(agent))
    await caller_script(line.send, lines, bookings, agent, args.cut, measure)
    line.frames_in.put_nowait(None)
    await call
    for timing in session.timings:
        base = timing.speech_end or timing.started
        steps = {k: getattr(timing, k) for k in ("ended", "transcribed", "first_audio")}
        after = ", ".join(f"{k} +{(v - base) * 1000:.0f} ms" for k, v in steps.items() if v)
        log(f"[turn] {timing.text!r}: cut {timing.cut_ms:.0f} ms; after speech end: {after}")
    report(measure.summary("agent-adapter", args.cut, bookings), args.json)
    return 0 if bookings.reserved == bookings.charged == 1 else 1


async def mic(args: argparse.Namespace) -> int:
    bookings = Bookings()
    audio = LocalAudio(out_rate=OUT_RATE)
    session = session_for(audio, args)
    agent = agent_for(session, bookings)
    print(f"Say: {LINES[0]!r}; while it books, say: {LINES[1]!r}")
    asyncio.get_running_loop().call_later(args.mic, audio.hang_up)
    await session.run(agent)
    print(f"reserved {bookings.reserved}, charged {bookings.charged}")
    return 0 if bookings.reserved == bookings.charged == 1 else 1


def webrtc(args: argparse.Namespace) -> int:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route

    from agent_adapter.voice.webrtc import WebRTCAudio
    from agent_adapter.voice.webrtc_page import webrtc_page

    calls: set[asyncio.Task[None]] = set()

    async def page(request: Request) -> HTMLResponse:
        return HTMLResponse(webrtc_page("/offer"))

    async def offer(request: Request) -> JSONResponse:
        rtc = WebRTCAudio()
        answer = await rtc.accept(await request.json())
        bookings = Bookings()
        session = session_for(rtc, args)
        task = asyncio.create_task(session.run(agent_for(session, bookings)))
        calls.add(task)
        task.add_done_callback(calls.discard)
        return JSONResponse(answer)

    app = Starlette(routes=[Route("/", page), Route("/offer", offer, methods=["POST"])])
    print(f"Open http://127.0.0.1:{args.port}/ and press Call. Say: {LINES[0]!r}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mic", type=float, help="be the caller for N seconds instead")
    parser.add_argument("--webrtc", action="store_true", help="serve a browser call page")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--cut", type=float, default=0.8, help="cut in N s into the booking")
    parser.add_argument("--silent", action="store_true", help="no speaker output")
    parser.add_argument("--stt-model", default="medium-streaming")
    parser.add_argument("--batch-stt", action="store_true", help="no streaming speech-to-text")
    parser.add_argument("--silero", type=Path)
    parser.add_argument("--smart-turn", type=Path)
    parser.add_argument("--json", type=Path, help="append the run's numbers to this file")
    args = parser.parse_args()
    if args.webrtc:
        sys.exit(webrtc(args))
    sys.exit(asyncio.run(mic(args) if args.mic else scripted(args)))


if __name__ == "__main__":
    main()
