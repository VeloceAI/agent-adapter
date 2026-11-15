"""voice-live: a VoiceAgent in a real Pipecat pipeline, real audio, a cut mid-booking.

    python scripts/voice_live_pipecat.py               # scripted caller, agent on the speakers
    python scripts/voice_live_pipecat.py --mic 40      # you are the caller (use headphones)

Pipeline: caller audio -> our Silero + Smart Turn detectors -> Moonshine speech-to-text ->
`voice_processor()` (the VoiceAgent) -> Moonshine Kokoro text-to-speech -> Pipecat's local
audio output. Everything runs on this machine with no keys: pipecat-ai[local] (PyAudio, MIT),
moonshine-voice (MIT; English models MIT, Kokoro voice Apache-2.0), agent-adapter[vad].
Moonshine downloads its models once (about 360 MB); set MOONSHINE_VOICE_CACHE to choose where.

Only the model is scripted. The scripted caller asks for a booking, then talks over the agent
while the booking is half-way (reserved, not yet charged) and asks "did that go through?". A
model that never saw the result would book again; the script exits non-zero unless the
booking was charged exactly once.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from voice_live_common import (
    FRAME,
    LINES,
    Bookings,
    answered,
    caller_lines,
    log,
    model_paths,
    registry,
    scripted_model,
    to_pcm16,
)

from agent_adapter.exec import Executor
from agent_adapter.integrations.pipecat import voice_processor
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.silero import SileroTurns
from agent_adapter.voice.smart_turn import RATE, SmartTurn


def _pipecat_parts(silero: Path, smart: Path) -> tuple[Any, Any]:
    """Our turn detectors as a Pipecat processor, and Moonshine TTS as a Pipecat service."""
    from moonshine_voice import TextToSpeech
    from pipecat.frames.frames import (
        InputAudioRawFrame,
        TTSAudioRawFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.settings import TTSSettings
    from pipecat.services.tts_service import TTSService

    class Turns(FrameProcessor):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.detector = SmartTurn(smart, SileroTurns(silero))

        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            if not isinstance(frame, InputAudioRawFrame):
                return
            for event in await asyncio.to_thread(self.detector.feed, frame.audio):
                if event == "started":
                    log("[turns] caller started speaking")
                    pair = (VADUserStartedSpeakingFrame(), UserStartedSpeakingFrame())
                else:
                    log("[turns] caller's turn ended")
                    pair = (VADUserStoppedSpeakingFrame(), UserStoppedSpeakingFrame())
                for out in pair:
                    await self.push_frame(out, FrameDirection.DOWNSTREAM)

    class MoonshineTTS(TTSService):  # type: ignore[misc]
        def __init__(self, voice: str = "kokoro_af_heart") -> None:
            settings = TTSSettings(model=None, voice=voice, language=None)
            super().__init__(push_start_frame=True, push_stop_frames=True, settings=settings)
            self.engine = TextToSpeech().language("en_us").voice(voice).load()

        def can_generate_metrics(self) -> bool:
            return False

        async def run_tts(self, text: str, context_id: str) -> Any:
            log(f"[agent] says: {text!r}")
            samples, rate = await asyncio.to_thread(self.engine.synthesize, text)
            pcm = to_pcm16(samples, rate)
            yield TTSAudioRawFrame(
                audio=pcm, sample_rate=rate, num_channels=1, context_id=context_id
            )

    return Turns, MoonshineTTS


async def scripted_caller(task: Any, lines: list[bytes], bookings: Bookings, agent: VoiceAgent,
                          cut_s: float) -> None:  # fmt: skip
    """Streams caller audio in real time, 20 ms a frame, quiet between lines."""
    from pipecat.frames.frames import EndFrame, InputAudioRawFrame

    quiet, clock = b"\x00" * FRAME * 2, time.monotonic()

    async def play(pcm: bytes) -> None:
        nonlocal clock
        for i in range(0, len(pcm), FRAME * 2):
            chunk = pcm[i : i + FRAME * 2].ljust(FRAME * 2, b"\x00")
            await task.queue_frame(
                InputAudioRawFrame(audio=chunk, sample_rate=RATE, num_channels=1)
            )
            clock += FRAME / RATE
            await asyncio.sleep(max(0.0, clock - time.monotonic()))

    async def quiet_until(done: Any, limit: float) -> None:
        end = time.monotonic() + limit
        while not done() and time.monotonic() < end:
            await play(quiet)

    await play(quiet * 25)
    log(f"[caller] says: {LINES[0]!r}")
    await play(lines[0])
    await quiet_until(bookings.started.is_set, 20)
    await quiet_until(lambda: False, cut_s)
    log(f"[caller] cuts in: {LINES[1]!r}")
    await play(lines[1])
    await quiet_until(lambda: bool(bookings.charged) and answered(agent), 25)
    await quiet_until(lambda: False, 5)  # let the answer play
    await task.queue_frame(EndFrame())


async def run(args: argparse.Namespace, silero: Path, smart: Path) -> int:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.services.moonshine.stt import MoonshineSTTService
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    from pipecat.workers.runner import WorkerRunner

    Turns, MoonshineTTS = _pipecat_parts(silero, smart)
    lines = [] if args.mic else caller_lines()
    bookings = Bookings()
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=bool(args.mic), audio_out_enabled=not args.silent,
        audio_in_sample_rate=RATE))  # fmt: skip
    processor = voice_processor()
    stt = MoonshineSTTService(settings=MoonshineSTTService.Settings(model=args.stt_model))
    steps = [Turns(), stt, processor, MoonshineTTS()]
    if args.mic:
        steps.insert(0, transport.input())
    if not args.silent:
        steps.append(transport.output())
    task = PipelineWorker(Pipeline(steps), params=PipelineParams(audio_in_sample_rate=RATE),
                          idle_timeout_secs=None)  # fmt: skip
    tools = registry(bookings)
    agent = VoiceAgent(scripted_model(), Executor(tools), processor.speaker,
                       instructions="You book show tickets.")  # fmt: skip
    processor.use(agent)
    heard = _Heard(processor)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    jobs = [asyncio.create_task(runner.run())]
    if args.mic:
        print(f"Say: {LINES[0]!r}; while it books, say: {LINES[1]!r}")
        jobs.append(asyncio.create_task(_end_after(task, args.mic)))
    else:
        jobs.append(asyncio.create_task(scripted_caller(task, lines, bookings, agent, args.cut)))
    await asyncio.gather(*jobs)
    await agent.close()
    heard.restore()
    print(f"\nreserved {bookings.reserved}, charged {bookings.charged}; history:")
    for message in agent.history:
        print(f"  {message.role}: {message.content or [c.name for c in message.tool_calls]}")
    return 0 if bookings.reserved == bookings.charged == 1 else 1


class _Heard:
    """Logs the transcripts the voice processor receives."""

    def __init__(self, processor: Any) -> None:
        from pipecat.frames.frames import TranscriptionFrame

        self.processor, self.original = processor, processor.process_frame

        async def process_frame(frame: Any, direction: Any) -> None:
            if isinstance(frame, TranscriptionFrame):
                log(f"[stt] heard: {frame.text!r}")
            await self.original(frame, direction)

        processor.process_frame = process_frame

    def restore(self) -> None:
        self.processor.process_frame = self.original


async def _end_after(task: Any, seconds: float) -> None:
    from pipecat.frames.frames import EndFrame

    await asyncio.sleep(seconds)
    await task.queue_frame(EndFrame())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mic", type=float, help="be the caller for N seconds instead")
    parser.add_argument("--cut", type=float, default=0.8, help="cut in N s into the booking")
    parser.add_argument("--silent", action="store_true", help="no speaker output")
    parser.add_argument("--stt-model", default="medium-streaming")
    parser.add_argument("--silero", type=Path)
    parser.add_argument("--smart-turn", type=Path)
    args = parser.parse_args()
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("PIPECAT_LOG", "WARNING"))
    silero, smart = model_paths(args.silero, args.smart_turn)
    sys.exit(asyncio.run(run(args, silero, smart)))


if __name__ == "__main__":
    main()
