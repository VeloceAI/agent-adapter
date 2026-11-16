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
                          cut_s: float, measure: Measure) -> None:  # fmt: skip
    """The shared scripted caller, as Pipecat input audio frames."""
    from pipecat.frames.frames import EndFrame, InputAudioRawFrame

    async def send(chunk: bytes) -> None:
        await task.queue_frame(InputAudioRawFrame(audio=chunk, sample_rate=RATE, num_channels=1))

    await caller_script(send, lines, bookings, agent, cut_s, measure)
    await task.queue_frame(EndFrame())


def _speaker(transport: Any, silent: bool, measure: Measure) -> Any:
    """Pipecat's audio output, timed; when silent, one that plays at a sound card's pace."""
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import TransportParams

    if not silent:
        out = transport.output()
        write = out.write_audio_frame

        async def timed(frame: Any) -> bool:
            measure.hear(frame.audio)
            return bool(await write(frame))

        out.write_audio_frame = timed
        return out

    class NullSpeaker(BaseOutputTransport):  # type: ignore[misc]
        async def start(self, frame: Any) -> None:
            await super().start(frame)
            await self.set_transport_ready(frame)

        async def write_audio_frame(self, frame: Any) -> bool:
            measure.hear(frame.audio)
            await asyncio.sleep(len(frame.audio) / (2 * frame.sample_rate))
            return True

    return NullSpeaker(TransportParams(audio_out_enabled=True))


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
    measure = Measure()
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=bool(args.mic), audio_out_enabled=not args.silent,
        audio_in_sample_rate=RATE))  # fmt: skip
    processor = voice_processor()
    stt = MoonshineSTTService(settings=MoonshineSTTService.Settings(model=args.stt_model))
    steps = [Turns(), stt, processor, MoonshineTTS()]
    if args.mic:
        steps.insert(0, transport.input())
    steps.append(_speaker(transport, args.silent, measure))
    task = PipelineWorker(Pipeline(steps), params=PipelineParams(audio_in_sample_rate=RATE),
                          idle_timeout_secs=None)  # fmt: skip
    tools = registry(bookings)
    agent = VoiceAgent(scripted_model(), Executor(tools), processor.speaker,
                       instructions="You book show tickets.")  # fmt: skip
    processor.use(agent)
    measure.watch(agent)
    heard = _Heard(processor)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    jobs = [asyncio.create_task(runner.run())]
    if args.mic:
        print(f"Say: {LINES[0]!r}; while it books, say: {LINES[1]!r}")
        jobs.append(asyncio.create_task(_end_after(task, args.mic)))
    else:
        caller = scripted_caller(task, lines, bookings, agent, args.cut, measure)
        jobs.append(asyncio.create_task(caller))
    await asyncio.gather(*jobs)
    await agent.close()
    heard.restore()
    print(f"\nreserved {bookings.reserved}, charged {bookings.charged}; history:")
    for message in agent.history:
        print(f"  {message.role}: {message.content or [c.name for c in message.tool_calls]}")
    if not args.mic:
        report(measure.summary("pipecat", args.cut, bookings), args.json)
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
    parser.add_argument("--json", type=Path, help="append the run's numbers to this file")
    args = parser.parse_args()
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("PIPECAT_LOG", "WARNING"))
    silero, smart = model_paths(args.silero, args.smart_turn)
    sys.exit(asyncio.run(run(args, silero, smart)))


if __name__ == "__main__":
    main()
