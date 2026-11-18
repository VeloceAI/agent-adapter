"""Cartesia Sonic, ElevenLabs stream-input, Fish Audio and Rime against local fakes of
their documented WebSocket messages (format-tested, not called live)."""

import asyncio
import base64
import json

import pytest

from agent_adapter.errors import ConfigError, ModelError, ModelRateLimited
from agent_adapter.voice.cartesia import CartesiaLiveTTS
from agent_adapter.voice.elevenlabs import ElevenLabsLiveTTS
from agent_adapter.voice.fish import FishAudioTTS
from agent_adapter.voice.media import TTS
from agent_adapter.voice.ports import StreamingTTS, WordTime
from agent_adapter.voice.rime import RimeTTS
from tests.voice.fake_ws import frames, pcm, serve

msgpack = pytest.importorskip("msgpack")
KEY = "test-key-not-a-real-one"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


async def collect(stream) -> bytes:
    return b"".join([chunk async for chunk in stream.audio()])


async def cartesia_fake(connection, seen):
    """Speaks each request's words as one sample per character; `continue: false` ends it."""
    async for frame in frames(connection, seen):
        request = json.loads(frame)
        context = request["context_id"]
        if request.get("cancel"):
            continue
        if text := request["transcript"]:
            chunk = {
                "type": "chunk",
                "data": b64(pcm(len(text))),
                "done": False,
                "status_code": 206,
            }
            await connection.send(json.dumps({**chunk, "context_id": context}))
            stamps = {
                "words": text.split(),
                "start": [0.0] * len(text.split()),
                "end": [0.2] * len(text.split()),
            }
            message = {"type": "timestamps", "context_id": context, "word_timestamps": stamps}
            await connection.send(json.dumps(message))
        if request.get("transcript") == "boom":
            error = {"type": "error", "title": "Busy", "message": "too many", "status_code": 429}
            await connection.send(json.dumps({**error, "done": True, "context_id": context}))
        elif not request["continue"]:
            await connection.send(json.dumps({"type": "done", "done": True, "context_id": context}))


async def test_cartesia_streams_text_into_contexts_on_one_shared_socket():
    async with serve(cartesia_fake) as (base, seen):
        tts = CartesiaLiveTTS(
            "voice-1", api_key=KEY, url=base, speed=1.2, emotion="calm", max_buffer_delay_ms=200
        )
        assert isinstance(tts, TTS) and isinstance(tts, StreamingTTS)
        await tts.warm()
        first, second = tts.open(), tts.open()
        await first.push("Hello ")
        await second.push("Other words")
        await first.push("there.", flush=True)
        await first.end()
        await second.end()
        a, b = await asyncio.gather(collect(first), collect(second))
        assert a == pcm(6) + pcm(6) and b == pcm(11)
        assert [w.word for w in first.words] == ["Hello", "there."]
        assert b"".join([c async for c in tts.synthesize("Hi.")]) == pcm(3)
        await tts.aclose()
    assert seen.connections == 1
    sent = seen.json()
    request = sent[0]
    assert request["model_id"] == "sonic-3" and request["voice"] == {"id": "voice-1"}
    assert request["output_format"] == {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 24000,
    }
    assert request["generation_config"] == {"speed": 1.2, "emotion": "calm"}
    assert request["continue"] is True and request["add_timestamps"] is True
    assert request["max_buffer_delay_ms"] == 200 and request["language"] == "en"
    assert [m.get("flush") for m in sent[:3]] == [None, None, True]
    ends = [m for m in sent if not m.get("continue")]
    assert {m["context_id"] for m in ends} == {first.context_id, second.context_id} | {
        ends[-1]["context_id"]
    }
    assert all(m["transcript"] == "" for m in ends[:2])
    assert seen.headers["x-api-key"] == KEY and KEY not in seen.path
    assert "cartesia_version=2026-08-14" in seen.path


async def test_cartesia_cancel_keeps_the_socket_and_errors_reach_only_their_context():
    async with serve(cartesia_fake) as (base, seen):
        tts = CartesiaLiveTTS("voice-1", api_key=KEY, url=base)
        stream = tts.open()
        await stream.push("Long reply")
        await stream.cancel()
        assert await collect(stream) == b""
        broken = tts.open()
        await broken.push("boom")
        with pytest.raises(ModelRateLimited, match="too many"):
            await collect(broken)
        assert b"".join([c async for c in tts.synthesize("ok")]) == pcm(2)
        await tts.aclose()
    assert seen.connections == 1
    assert {"context_id": stream.context_id, "cancel": True} in seen.json()
    with pytest.raises(ValueError, match="sample_rate"):
        CartesiaLiveTTS("v", sample_rate=11_025)


async def elevenlabs_fake(connection, seen):
    async for frame in frames(connection, seen):
        message = json.loads(frame)
        text = message["text"]
        if text == "":
            await connection.send(json.dumps({"isFinal": True}))
            await connection.close()
            return
        if text.strip():
            chars = list(text)
            alignment = {
                "chars": chars,
                "charStartTimesMs": [i * 10 for i in range(len(chars))],
                "charDurationsMs": [10] * len(chars),
            }
            reply = {
                "audio": b64(pcm(240 * len(text))),
                "alignment": alignment,
                "normalizedAlignment": alignment,
            }
            await connection.send(json.dumps(reply))


async def test_elevenlabs_sends_the_schedule_flushes_and_times_words_across_chunks():
    async with serve(elevenlabs_fake) as (base, seen):
        tts = ElevenLabsLiveTTS(
            "voice/1", api_key=KEY, url=f"{base}/v1/text-to-speech", voice_settings={"speed": 1.1}
        )
        stream = tts.open()
        await stream.push("Hi")
        await stream.push("you there.", flush=True)
        await stream.end()
        audio = await collect(stream)
    assert audio == pcm(240 * 3) + pcm(240 * 11)
    assert stream.words[0] == WordTime("Hi", 0.0, 0.02)
    assert stream.words[1].word == "you" and stream.words[1].start == pytest.approx(0.03)
    assert seen.json() == [
        {
            "text": " ",
            "voice_settings": {"speed": 1.1},
            "generation_config": {"chunk_length_schedule": [50, 120, 160, 290]},
        },
        {"text": "Hi "},
        {"text": "you there. ", "flush": True},
        {"text": ""},
    ]
    assert seen.path.startswith("/v1/text-to-speech/voice%2F1/stream-input?")
    for part in ("model_id=eleven_flash_v2_5", "output_format=pcm_24000", "inactivity_timeout=60"):
        assert part in seen.path
    assert seen.headers["xi-api-key"] == KEY and KEY not in seen.path


async def test_elevenlabs_warm_opens_the_next_socket_ahead_and_cancel_closes_it():
    async with serve(elevenlabs_fake) as (base, seen):
        tts = ElevenLabsLiveTTS("v", api_key=KEY, url=base)
        await tts.warm()
        assert seen.connections == 1 and tts._spare is not None
        assert b"".join([c async for c in tts.synthesize("Hello.")]) == pcm(240 * 7)
        assert seen.connections == 1 and tts._spare is None
        stream = tts.open()
        await stream.push("Never mind")
        await stream.cancel()
        assert await collect(stream) == b""
        await tts.warm()
        tts._spare = (tts._spare[0] - 60, tts._spare[1])  # an old spare is not reused
        assert b"".join([c async for c in tts.synthesize("Hi")]) == pcm(240 * 3)
        await tts.aclose()
    assert seen.connections == 4


async def fish_fake(connection, seen):
    async for frame in frames(connection, seen):
        event = msgpack.unpackb(frame, raw=False)
        seen.frames[-1] = json.dumps({k: v for k, v in event.items() if k != "request"})
        if event["event"] == "start":
            seen.headers["start"] = event["request"]
        elif event["event"] == "text":
            await connection.send(msgpack.packb({"event": "log", "message": "debug"}))
            await connection.send(
                msgpack.packb(
                    {"event": "audio", "audio": pcm(len(event["text"]))}, use_bin_type=True
                )
            )
        elif event["event"] == "stop":
            reason = "error" if seen.headers["start"].get("reference_id") == "broken" else "stop"
            await connection.send(msgpack.packb({"event": "finish", "reason": reason}))
            await connection.close()


async def test_fish_sends_messagepack_frames_with_a_reference_clip():
    async with serve(fish_fake) as (base, seen):
        tts = FishAudioTTS(
            reference=(b"\x01\x02clip", "hello"), api_key=KEY, url=base, speed=1.1, latency="low"
        )
        stream = tts.open()
        await stream.push("Hello ")
        await stream.push("there.", flush=True)
        await stream.end()
        assert await collect(stream) == pcm(6) + pcm(6)
    assert seen.headers["authorization"] == f"Bearer {KEY}" and seen.headers["model"] == "s2.1-pro"
    assert seen.headers["start"] == {
        "text": "",
        "format": "pcm",
        "sample_rate": 24000,
        "latency": "low",
        "references": [{"audio": b"\x01\x02clip", "text": "hello"}],
        "prosody": {"speed": 1.1},
    }
    events = [json.loads(f)["event"] for f in seen.frames]
    assert events == ["start", "text", "text", "flush", "stop"]


async def test_fish_reports_a_failed_synthesis_and_needs_one_voice():
    async with serve(fish_fake) as (base, _):
        tts = FishAudioTTS("broken", api_key=KEY, url=base)
        with pytest.raises(ModelError, match="synthesis failed"):
            b"".join([c async for c in tts.synthesize("Hi")])
    with pytest.raises(ValueError, match="reference_id"):
        FishAudioTTS()


async def rime_fake(connection, seen):
    await connection.send(
        json.dumps({"ready": {"protocol": 1, "languages": ["en"], "defaultLanguage": "en"}})
    )
    async for frame in frames(connection, seen):
        message = json.loads(frame)
        context = message.get("contextId")
        if "start" in message:
            await connection.send(
                json.dumps({"contextId": context, "started": {"requestId": "r1"}})
            )
        elif "text" in message:
            if message["text"] == "boom":
                error = {"kind": "invalid_input", "message": "bad text", "requestId": "r1"}
                await connection.send(json.dumps({"contextId": context, "error": error}))
            else:
                await connection.send(
                    json.dumps({"contextId": context, "audio": b64(pcm(len(message["text"])))})
                )
        elif "end" in message:
            spans = [{"text": "Hi.", "start": "0.080s", "end": "0.520s"}]
            done = {"timestamps": {"status": {}, "spans": spans}}
            await connection.send(json.dumps({"contextId": context, "done": done}))
        elif "cancel" in message:
            await connection.send(json.dumps({"contextId": context, "cancelled": {}}))


async def test_rime_waits_for_ready_then_streams_contexts_with_word_spans():
    async with serve(rime_fake) as (base, seen):
        tts = RimeTTS("lyra", api_key=KEY, url=base, time_scale=0.9)
        stream = tts.open()
        await stream.push("Hi.")
        await stream.end()
        assert await collect(stream) == pcm(3)
        assert stream.words == [WordTime("Hi.", 0.08, 0.52)]
        broken = tts.open()
        await broken.push("boom")
        with pytest.raises(ModelError, match="invalid_input: bad text"):
            await collect(broken)
        cut = tts.open()
        await cut.push("Long reply")
        await cut.cancel()
        await tts.aclose()
    assert seen.connections == 1 and seen.headers["authorization"] == f"Bearer {KEY}"
    start = seen.json()[0]
    assert start == {
        "contextId": stream.context_id,
        "start": {
            "speaker": "lyra",
            "audioParameters": {
                "audioFormat": "audio/pcm",
                "samplingRate": 24000,
                "timeScaleFactor": 0.9,
            },
            "splitStrategy": "SPLIT_STRATEGY_ACCUMULATE_SENTENCE",
            "timestamps": {"enable": True},
            "language": "en",
        },
    }
    assert {"contextId": cut.context_id, "cancel": {}} in seen.json()
    assert RimeTTS("x", model="mistv3").url == "wss://mist.api.rime.ai/ws"
    with pytest.raises(ValueError, match="url="):
        RimeTTS("x", model="mistv2")


async def test_rime_refuses_a_socket_that_is_not_ready():
    async def old(connection, seen):
        await connection.send(json.dumps({"ready": {"protocol": 2}}))

    async with serve(old) as (base, _):
        with pytest.raises(ModelError, match="unsupported protocol"):
            b"".join([c async for c in RimeTTS("lyra", api_key=KEY, url=base).synthesize("Hi")])


async def test_a_dropped_socket_fails_the_reply_and_the_next_one_reconnects():
    async def drop(connection, seen):
        async for _ in frames(connection, seen):
            await connection.close()
            return

    async with serve(drop) as (base, seen):
        tts = CartesiaLiveTTS("v", api_key=KEY, url=base)
        for _ in range(2):
            with pytest.raises(ModelError, match="cartesia"):
                b"".join([c async for c in tts.synthesize("Hi")])
        await tts.aclose()
    assert seen.connections == 2


@pytest.mark.parametrize(
    ("make", "env"),
    [
        (lambda: CartesiaLiveTTS("v"), "CARTESIA_API_KEY"),
        (lambda: ElevenLabsLiveTTS("v"), "ELEVENLABS_API_KEY"),
        (lambda: FishAudioTTS("v"), "FISH_AUDIO_API_KEY"),
        (lambda: RimeTTS("v"), "RIME_API_KEY"),
    ],
)
async def test_a_missing_key_names_its_variable(monkeypatch, make, env):
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(ConfigError, match=env):
        b"".join([c async for c in make().synthesize("Hi")])
