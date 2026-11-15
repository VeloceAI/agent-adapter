"""voice-live-livekit: a VoiceAgent on a LiveKit AgentSession in a real room, a cut mid-booking.

    livekit-server --dev --bind 127.0.0.1          # Apache-2.0, in another terminal
    python scripts/voice_live_livekit.py            # agent heard on the speakers

Two participants join a room on the local server over WebRTC. The scripted caller publishes
a microphone track (a second Kokoro voice, in real time). The agent side is a LiveKit
`AgentSession` with no LLM, driven by `attach_voice_agent`: our Silero + Smart Turn detectors
and Moonshine speech-to-text run as a LiveKit streaming STT, Moonshine's Kokoro voice as its
TTS, and `LiveKitSpeaker` speaks for the `VoiceAgent`. The caller plays the agent's track on
the speakers. Needs agent-adapter[livekit,vad], moonshine-voice and sounddevice; the dev
server's default key and secret are used. Exits non-zero unless the booking was charged once.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
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
from agent_adapter.integrations.livekit import LiveKitSpeaker, attach_voice_agent
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.silero import SileroTurns
from agent_adapter.voice.smart_turn import RATE, SmartTurn

ROOM = "booking"
PRE_ROLL = RATE // 2 * 2  # bytes in 0.5 s of 16-bit audio before "started"


def token(key: str, secret: str, identity: str) -> str:
    from livekit import api

    grants = api.VideoGrants(room_join=True, room=ROOM)
    return api.AccessToken(key, secret).with_identity(identity).with_grants(grants).to_jwt()


def livekit_speech(silero: Path, smart: Path, stt_model: str) -> tuple[Any, Any]:
    """Our turn detectors plus Moonshine as a LiveKit streaming STT; Moonshine as its TTS."""
    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, stt, tts
    from moonshine_voice import (
        TextToSpeech,
        Transcriber,
        get_model_for_language,
        string_to_model_arch,
    )

    path, arch = get_model_for_language("en", string_to_model_arch(stt_model))
    transcriber = Transcriber(model_path=path, model_arch=arch)

    def transcribe(pcm: bytes) -> str:
        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32_768
        lines = transcriber.transcribe_without_streaming(audio.tolist(), RATE).lines
        return " ".join(line.text for line in lines).strip()

    class TurnStream(stt.RecognizeStream):  # type: ignore[misc]
        async def _run(self) -> None:
            detector = SmartTurn(smart, SileroTurns(silero))
            turn, recent = bytearray(), bytearray()
            async for frame in self._input_ch:
                if isinstance(frame, self._FlushSentinel):
                    continue
                pcm = bytes(frame.data)
                recent = (recent + pcm)[-PRE_ROLL:]
                if detector.speaking:
                    turn += pcm
                for event in detector.feed(pcm):  # a thread hop per 10 ms frame fell behind
                    if event == "started":
                        log("[turns] caller started speaking")
                        turn = bytearray(recent)
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(stt.SpeechEventType.START_OF_SPEECH)
                        )
                        continue
                    log("[turns] caller's turn ended")
                    text = await asyncio.to_thread(transcribe, bytes(turn))
                    log(f"[stt] heard: {text!r}")
                    data = stt.SpeechData(language="en", text=text)
                    final = stt.SpeechEvent(
                        stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[data]
                    )
                    self._event_ch.send_nowait(final)
                    self._event_ch.send_nowait(stt.SpeechEvent(stt.SpeechEventType.END_OF_SPEECH))

    class TurnsSTT(stt.STT):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(
                capabilities=stt.STTCapabilities(streaming=True, interim_results=False)
            )

        async def _recognize_impl(self, buffer: Any, **_: Any) -> Any:
            raise NotImplementedError("streaming only")

        def stream(self, *, conn_options: Any = DEFAULT_API_CONNECT_OPTIONS, **_: Any) -> Any:
            return TurnStream(stt=self, conn_options=conn_options, sample_rate=RATE)

    engine = TextToSpeech().language("en_us").voice("kokoro_af_heart").load()

    class Spoken(tts.ChunkedStream):  # type: ignore[misc]
        async def _run(self, output_emitter: Any) -> None:
            log(f"[agent] says: {self._input_text!r}")
            samples, rate = await asyncio.to_thread(engine.synthesize, self._input_text)
            output_emitter.initialize(
                request_id=str(time.monotonic()), sample_rate=rate, num_channels=1,
                mime_type="audio/pcm",
            )  # fmt: skip
            output_emitter.push(to_pcm16(samples, rate))
            output_emitter.flush()

    class MoonshineTTS(tts.TTS):  # type: ignore[misc]
        def __init__(self) -> None:
            caps = tts.TTSCapabilities(streaming=False)
            super().__init__(capabilities=caps, sample_rate=24_000, num_channels=1)

        def synthesize(self, text: str, *, conn_options: Any = DEFAULT_API_CONNECT_OPTIONS) -> Any:
            return Spoken(tts=self, input_text=text, conn_options=conn_options)

    return TurnsSTT(), MoonshineTTS()


def _speaker(frames: queue.Queue[bytes | None]) -> None:
    import sounddevice as sd

    with sd.RawOutputStream(samplerate=24_000, channels=1, dtype="int16") as out:
        while (chunk := frames.get()) is not None:
            out.write(chunk)


def play_agent(room: Any, silent: bool) -> Callable[[], None]:
    """Play the agent's audio tracks on the default output device; returns `stop`."""
    from livekit import rtc

    frames: queue.Queue[bytes | None] = queue.Queue()
    speaker = threading.Thread(target=_speaker, args=(frames,), daemon=True)
    players: set[asyncio.Task[None]] = set()

    async def play(track: Any) -> None:
        async for event in rtc.AudioStream(track, sample_rate=24_000, num_channels=1):
            if not silent:
                frames.put(bytes(event.frame.data))

    @room.on("track_subscribed")  # type: ignore[misc]
    def on_track(track: Any, *_: Any) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            task = asyncio.ensure_future(play(track))
            players.add(task)
            task.add_done_callback(players.discard)

    def stop() -> None:
        for task in players:
            task.cancel()
        frames.put(None)
        if speaker.is_alive():
            speaker.join(timeout=5)

    if not silent:
        speaker.start()
    return stop


async def scripted_caller(room: Any, bookings: Bookings, agent: VoiceAgent, cut_s: float) -> None:
    """Publishes the caller's microphone track and speaks in real time, quiet between lines."""
    from livekit import rtc

    lines = caller_lines()
    source = rtc.AudioSource(RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("caller-mic", source)
    # DTX sends nothing in quiet, and the first syllable after quiet ("Wait") came out clipped
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE, dtx=False)
    await room.local_participant.publish_track(track, options)
    quiet, clock = b"\x00" * FRAME * 2, time.monotonic()

    async def play(pcm: bytes) -> None:
        nonlocal clock
        for i in range(0, len(pcm), FRAME * 2):
            chunk = pcm[i : i + FRAME * 2].ljust(FRAME * 2, b"\x00")
            frame = rtc.AudioFrame(chunk, RATE, 1, FRAME)
            await source.capture_frame(frame)
            clock += FRAME / RATE
            await asyncio.sleep(max(0.0, clock - time.monotonic()))

    async def quiet_until(done: Any, limit: float) -> None:
        end = time.monotonic() + limit
        while not done() and time.monotonic() < end:
            await play(quiet)

    await quiet_until(lambda: False, 1.0)
    log(f"[caller] says: {LINES[0]!r}")
    await play(lines[0])
    await quiet_until(bookings.started.is_set, 20)
    await quiet_until(lambda: False, cut_s)
    log(f"[caller] cuts in: {LINES[1]!r}")
    await play(lines[1])
    await quiet_until(lambda: bool(bookings.charged) and answered(agent), 25)
    await quiet_until(lambda: False, 5)  # let the answer play


async def run(args: argparse.Namespace) -> int:
    from livekit import rtc
    from livekit.agents import Agent, AgentSession, TurnHandlingOptions

    silero, smart = model_paths(args.silero, args.smart_turn)
    speech_to_text, text_to_speech = livekit_speech(silero, smart, args.stt_model)
    bookings = Bookings()
    agent_room, caller_room = rtc.Room(), rtc.Room()
    await agent_room.connect(args.url, token(args.key, args.secret, "agent"))
    await caller_room.connect(args.url, token(args.key, args.secret, "caller"))
    stop_playing = play_agent(caller_room, args.silent)
    turns = TurnHandlingOptions(turn_detection="stt")
    session = AgentSession(stt=speech_to_text, tts=text_to_speech, turn_handling=turns)
    await session.start(Agent(instructions="You book show tickets."), room=agent_room, record=False)
    tools = registry(bookings)
    agent = VoiceAgent(scripted_model(), Executor(tools), LiveKitSpeaker(session),
                       instructions="You book show tickets.")  # fmt: skip
    attach_voice_agent(session, agent)
    try:
        await scripted_caller(caller_room, bookings, agent, args.cut)
    finally:
        await agent.close()
        stop_playing()
        await session.aclose()
        await caller_room.disconnect()
        await agent_room.disconnect()
    print(f"\nreserved {bookings.reserved}, charged {bookings.charged}; history:")
    for message in agent.history:
        print(f"  {message.role}: {message.content or [c.name for c in message.tool_calls]}")
    return 0 if bookings.reserved == bookings.charged == 1 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="ws://127.0.0.1:7880")
    parser.add_argument("--key", default="devkey", help="the dev server's default")
    parser.add_argument("--secret", default="secret", help="the dev server's default")
    parser.add_argument("--cut", type=float, default=0.8, help="cut in N s into the booking")
    parser.add_argument("--silent", action="store_true", help="no speaker output")
    parser.add_argument("--stt-model", default="medium-streaming")
    parser.add_argument("--silero", type=Path)
    parser.add_argument("--smart-turn", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
