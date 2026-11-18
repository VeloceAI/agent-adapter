"""AssemblyAI, Soniox, Cartesia Ink and ElevenLabs Scribe against local fakes of their
documented WebSocket messages (format-tested, not called live)."""

import asyncio
import base64
import json

import pytest

from agent_adapter.errors import ConfigError, ModelAuthError, ModelError, ModelRateLimited
from agent_adapter.voice.assemblyai import AssemblyAISTT
from agent_adapter.voice.cartesia import CartesiaInkSTT
from agent_adapter.voice.elevenlabs import ElevenLabsScribeSTT
from agent_adapter.voice.media import StreamingSTT
from agent_adapter.voice.ports import TurnSignal, TurnSignals
from agent_adapter.voice.soniox import SonioxSTT
from tests.voice.fake_ws import frames, pcm, serve

KEY = "test-key-not-a-real-one"


def assembly_turn(order: int, text: str, end: bool, confidence: float = 0.4) -> str:
    words = [
        {"text": w, "start": 0, "end": 1, "confidence": 0.9, "word_is_final": end}
        for w in text.split()
    ]
    return json.dumps(
        {
            "type": "Turn",
            "turn_order": order,
            "turn_is_formatted": end,
            "end_of_turn": end,
            "transcript": text,
            "end_of_turn_confidence": confidence,
            "words": words,
        }
    )


async def test_assemblyai_regroups_audio_signals_the_end_and_finishes_at_once():
    async def assembly(connection, seen):
        await connection.send(json.dumps({"type": "Begin", "id": "s-1", "expires_at": 1}))
        async for frame in frames(connection, seen):
            if isinstance(frame, bytes) and len(seen.frames) == 1:
                await connection.send(assembly_turn(0, "two seats", False))
                await connection.send(assembly_turn(0, "Two seats, please.", True, 0.92))

    async with serve(assembly) as (base, seen):
        stt = AssemblyAISTT(
            api_key=KEY, url=f"{base}/v3/ws", mode="min_latency", keyterms=("Jazz Club",)
        )
        assert isinstance(stt, StreamingSTT)
        stream = stt.open(16_000)
        assert isinstance(stream, TurnSignals)
        got: list[TurnSignal] = []
        stream.on_turn(got.append)
        for _ in range(5):
            stream.feed(pcm(160))  # 10 ms each: too short to send alone
        stream.feed(pcm(40_000))  # 2.5 s: split into chunks of at most 1 s
        stream.feed(pcm(100))
        while not got:
            await asyncio.sleep(0.01)
        assert await stream.finish() == "Two seats, please."
    assert got == [TurnSignal("end", "Two seats, please.", 0.92)]
    sizes = [len(f) for f in seen.frames if isinstance(f, bytes)]
    assert sizes[0] == 1600 and max(sizes) == 32_000 and min(sizes) >= 640
    sent = pcm(160) * 5 + pcm(40_000) + pcm(100)
    assert seen.audio()[: len(sent)] == sent and not seen.audio()[len(sent) :].strip(b"\0")
    assert seen.json() == [{"type": "ForceEndpoint"}, {"type": "Terminate"}]
    assert seen.headers["authorization"] == KEY and KEY not in seen.path
    for part in (
        "sample_rate=16000",
        "encoding=pcm_s16le",
        "speech_model=universal-3-6-pro",
        "mode=min_latency",
    ):
        assert part in seen.path
    assert "keyterms_prompt=%5B%22Jazz+Club%22%5D" in seen.path


async def test_assemblyai_waits_for_termination_and_types_its_errors():
    async def assembly(connection, seen):
        async for frame in frames(connection, seen):
            if frame == json.dumps({"type": "Terminate"}, separators=(",", ":")):
                await connection.send(assembly_turn(0, "hello", False))
                await connection.send(
                    json.dumps({"type": "Termination", "audio_duration_seconds": 1})
                )

    async with serve(assembly) as (base, _):
        stt = AssemblyAISTT(api_key=KEY, url=base)
        stt.realtime_only = False
        assert await stt.transcribe(pcm(3200), 16_000) == "hello"

    async def failing(connection, seen):
        await connection.send(
            json.dumps({"type": "Error", "error_code": 3007, "error": "too fast"})
        )
        await connection.close(3007)

    async with serve(failing) as (base, _):
        stream = AssemblyAISTT(api_key=KEY, url=base).open(16_000)
        stream.feed(pcm(1600))
        with pytest.raises(ModelError, match="too fast"):
            await stream.finish()


async def test_soniox_sends_the_config_without_the_key_and_stops_at_fin():
    def tokens(*items: tuple[str, bool]) -> str:
        return json.dumps(
            {"tokens": [{"text": t, "is_final": f, "confidence": 0.9} for t, f in items]}
        )

    async def soniox(connection, seen):
        async for frame in frames(connection, seen):
            if isinstance(frame, bytes):
                await connection.send(tokens(("Book", True), (" two", False)))
                await connection.send(tokens((" two", True), (" seats", True), ("<end>", True)))
            elif frame.startswith('{"type":"finalize"'):
                await connection.send(tokens((".", True), ("<fin>", True)))
            elif frame == "":
                await connection.send(json.dumps({"tokens": [], "finished": True}))

    async with serve(soniox) as (base, seen):
        stt = SonioxSTT(api_key=KEY, url=base, language_hints=("en",), max_endpoint_delay_ms=800)
        stream = stt.open(16_000)
        got: list[TurnSignal] = []
        stream.on_turn(got.append)
        stream.feed(pcm(1600))
        while not got:
            await asyncio.sleep(0.01)
        assert await stream.finish() == "Book two seats."
    assert got == [TurnSignal("end", "Book two seats")]
    config = json.loads(seen.frames[0])
    assert config == {
        "model": "stt-rt-v6",
        "audio_format": "pcm_s16le",
        "sample_rate": 16000,
        "num_channels": 1,
        "enable_endpoint_detection": True,
        "language_hints": ["en"],
        "max_endpoint_delay_ms": 800,
    }
    assert seen.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in "".join(f for f in seen.frames if isinstance(f, str))
    assert seen.frames[-2:] == ['{"type":"finalize"}', ""]


async def test_soniox_errors_are_typed():
    async def soniox(connection, seen):
        async for _frame in frames(connection, seen):
            message = {
                "error_code": 401,
                "error_type": "unauthenticated",
                "error_message": "bad key",
            }
            await connection.send(json.dumps(message))
            await connection.close()

    async with serve(soniox) as (base, _):
        stream = SonioxSTT(api_key=KEY, url=base).open(16_000)
        with pytest.raises(ModelAuthError, match="bad key"):
            await stream.finish()


async def test_cartesia_ink_joins_final_deltas_and_stops_at_flush_done():
    async def ink(connection, seen):
        async for frame in frames(connection, seen):
            if isinstance(frame, bytes):
                for text, final in (("How ar", False), ("How are", True), (" you?", True)):
                    message = {
                        "type": "transcript",
                        "is_final": final,
                        "text": text,
                        "request_id": "r",
                    }
                    await connection.send(json.dumps(message))
            elif frame == "finalize":
                await connection.send(json.dumps({"type": "flush_done", "request_id": "r"}))

    async with serve(ink) as (base, seen):
        stt = CartesiaInkSTT("ink-whisper", language="en", api_key=KEY, url=base)
        stream = stt.open(16_000)
        stream.feed(pcm(1600))
        await asyncio.sleep(0.05)
        assert await stream.finish() == "How are you?"
    assert seen.headers["x-api-key"] == KEY and seen.headers["cartesia-version"] == "2026-08-14"
    for part in (
        "model=ink-whisper",
        "encoding=pcm_s16le",
        "sample_rate=16000",
        "language=en",
        "cartesia_version=",
    ):
        assert part in seen.path
    assert seen.frames[-2:] == ["finalize", "close"]


async def test_cartesia_ink_errors_are_typed():
    async def ink(connection, seen):
        async for _ in frames(connection, seen):
            message = {
                "type": "error",
                "status_code": 429,
                "title": "Too many",
                "message": "slow down",
            }
            await connection.send(json.dumps(message))

    async with serve(ink) as (base, _):
        stream = CartesiaInkSTT(api_key=KEY, url=base).open(16_000)
        stream.feed(pcm(160))
        with pytest.raises(ModelRateLimited, match="slow down"):
            await stream.finish()


async def test_scribe_sends_base64_chunks_signals_vad_commits_and_commits_at_the_end():
    async def scribe(connection, seen):
        await connection.send(json.dumps({"message_type": "session_started", "session_id": "s"}))
        async for frame in frames(connection, seen):
            message = json.loads(frame)
            if message["commit"]:
                await connection.send(
                    json.dumps({"message_type": "committed_transcript", "text": "please"})
                )
            elif len(seen.frames) == 1:
                await connection.send(
                    json.dumps({"message_type": "partial_transcript", "text": "two"})
                )
                await connection.send(
                    json.dumps({"message_type": "committed_transcript", "text": "Two seats"})
                )

    async with serve(scribe) as (base, seen):
        stt = ElevenLabsScribeSTT(api_key=KEY, url=base, commit_strategy="vad", language="en")
        stream = stt.open(16_000)
        got: list[TurnSignal] = []
        stream.on_turn(got.append)
        stream.feed(pcm(800))
        while not got:
            await asyncio.sleep(0.01)
        stream.feed(pcm(800, 2))
        assert await stream.finish() == "Two seats please"
    first, *_, last = seen.json()
    assert first == {
        "message_type": "input_audio_chunk",
        "audio_base_64": base64.b64encode(pcm(800)).decode(),
        "commit": False,
        "sample_rate": 16000,
    }
    assert last["commit"] is True and last["audio_base_64"] == ""
    assert got[0] == TurnSignal("end", "Two seats")
    assert seen.headers["xi-api-key"] == KEY
    for part in (
        "model_id=scribe_v2_realtime",
        "audio_format=pcm_16000",
        "commit_strategy=vad",
        "language_code=en",
    ):
        assert part in seen.path


async def test_scribe_needs_no_commit_when_nothing_is_new_and_types_errors():
    async def scribe(connection, seen):
        async for _frame in frames(connection, seen):
            await connection.send(
                json.dumps({"message_type": "committed_transcript", "text": "done"})
            )

    async with serve(scribe) as (base, seen):
        stream = ElevenLabsScribeSTT(api_key=KEY, url=base).open(16_000)
        stream.feed(pcm(320))
        while "done" not in stream._result():
            await asyncio.sleep(0.01)
        assert await stream.finish() == "done"
    assert not any(m["commit"] for m in seen.json())

    async def refusing(connection, seen):
        await connection.send(json.dumps({"message_type": "auth_error", "error": "invalid key"}))

    async with serve(refusing) as (base, _):
        stream = ElevenLabsScribeSTT(api_key=KEY, url=base).open(16_000)
        stream.feed(pcm(320))
        with pytest.raises(ModelAuthError, match="invalid key"):
            await stream.finish()
    with pytest.raises(ValueError, match="sample_rate"):
        ElevenLabsScribeSTT(api_key=KEY).open(11_025)


@pytest.mark.parametrize(
    ("make", "env"),
    [
        (lambda: AssemblyAISTT(), "ASSEMBLYAI_API_KEY"),
        (lambda: SonioxSTT(), "SONIOX_API_KEY"),
        (lambda: CartesiaInkSTT(), "CARTESIA_API_KEY"),
        (lambda: ElevenLabsScribeSTT(), "ELEVENLABS_API_KEY"),
    ],
)
async def test_a_missing_key_names_its_variable(monkeypatch, make, env):
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(ConfigError, match=env):
        await make().open(16_000).finish()
