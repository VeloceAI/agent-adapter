import asyncio
import json

import pytest

from agent_adapter.errors import ModelAuthError, ModelError
from agent_adapter.voice.deepgram import DeepgramFluxSTT
from agent_adapter.voice.media import StreamingSTT
from agent_adapter.voice.ports import TurnSignal, TurnSignals
from tests.voice.fake_ws import frames, pcm, serve

KEY = "dg-test-not-a-real-key"


def turn(event: str, text: str, confidence: float = 0.6, index: int = 0) -> str:
    return json.dumps(
        {
            "type": "TurnInfo",
            "request_id": "r-1",
            "sequence_id": 3,
            "event": event,
            "turn_index": index,
            "audio_window_start": 0.0,
            "audio_window_end": 1.2,
            "transcript": text,
            "words": [{"word": w, "confidence": 0.9} for w in text.split()],
            "end_of_turn_confidence": confidence,
        }
    )


def flux(base: str, **options) -> DeepgramFluxSTT:
    return DeepgramFluxSTT(api_key=KEY, url=f"{base}/v2/listen", **options)


async def test_eager_end_resumed_and_end_reach_the_listener_and_finish_is_instant():
    script = [
        turn("StartOfTurn", "book"),
        turn("Update", "book two"),
        turn("EagerEndOfTurn", "book two", 0.55),
        turn("TurnResumed", "book two"),
        turn("Update", "book two seats"),
        turn("EagerEndOfTurn", "book two seats", 0.6),
        turn("EndOfTurn", "Book two seats.", 0.85),
    ]

    async def deepgram(connection, seen):
        await connection.send(json.dumps({"type": "Connected", "request_id": "r-1"}))
        count = 0
        async for frame in frames(connection, seen):
            if isinstance(frame, bytes) and count < len(script):
                await connection.send(script[count])
                count += 1
            # the socket stays open: finish must not wait for it to close

    async with serve(deepgram) as (base, seen):
        stt = flux(base, eot_timeout_ms=3000, keyterms=("Jazz Club",))
        assert isinstance(stt, StreamingSTT)
        stream = stt.open(16_000)
        assert isinstance(stream, TurnSignals)
        got: list[TurnSignal] = []
        stream.on_turn(got.append)
        for _ in script:
            stream.feed(pcm(1280))
        while len(got) < 4:
            await asyncio.sleep(0.01)
        assert await stream.finish() == "Book two seats."
    assert [s.event for s in got] == ["eager_end", "resumed", "eager_end", "end"]
    assert got[0] == TurnSignal("eager_end", "book two", 0.55)
    assert got[-1] == TurnSignal("end", "Book two seats.", 0.85)
    assert seen.headers["authorization"] == f"Token {KEY}"
    for part in (
        "/v2/listen?model=flux-general-en",
        "encoding=linear16",
        "sample_rate=16000",
        "eot_threshold=0.7",
        "eager_eot_threshold=0.5",
        "eot_timeout_ms=3000",
        "keyterm=Jazz+Club",
    ):
        assert part in seen.path
    assert KEY not in seen.path and seen.audio() == pcm(1280) * len(script)


async def test_finish_before_end_of_turn_sends_close_stream_and_keeps_the_words():
    async def deepgram(connection, seen):
        async for frame in frames(connection, seen):
            if isinstance(frame, bytes):
                await connection.send(turn("Update", "two tickets"))
            elif json.loads(frame)["type"] == "ForceEndTurn":
                await connection.send(turn("EndOfTurn", "Two tickets please.", 0.9))
                await connection.close()

    async with serve(deepgram) as (base, seen):
        stream = flux(base, eager_eot_threshold=None).open(8_000)
        stream.feed(pcm(640))
        assert await stream.finish() == "Two tickets please."
    assert "eager_eot_threshold" not in seen.path
    assert seen.json() == [{"type": "ForceEndTurn"}, {"type": "CloseStream"}]


async def test_two_turns_in_one_stream_join_and_a_closed_socket_keeps_the_partial():
    async def deepgram(connection, seen):
        async for frame in frames(connection, seen):
            if isinstance(frame, bytes):
                await connection.send(turn("EndOfTurn", "Hello.", index=0))
                await connection.send(turn("StartOfTurn", "and", index=1))
                await connection.send(turn("Update", "and goodbye", index=1))
            else:
                await connection.close()

    async with serve(deepgram) as (base, _):
        stream = flux(base).open(16_000)
        stream.feed(pcm(320))
        while "goodbye" not in stream._result():
            await asyncio.sleep(0.01)
        assert await stream.finish() == "Hello. and goodbye"


async def test_a_fatal_error_and_a_rejected_key_are_typed():
    async def deepgram(connection, seen):
        error = {"type": "Error", "sequence_id": 1, "code": "BAD", "description": "bad audio"}
        await connection.send(json.dumps(error))
        async for _ in frames(connection, seen):
            pass

    async with serve(deepgram) as (base, _):
        stream = flux(base).open(16_000)
        stream.feed(pcm(320))
        with pytest.raises(ModelError, match="bad audio"):
            await stream.finish()

    class Rejected(Exception):
        class response:
            status_code = 401

    async def rejected(url, headers):
        raise Rejected()

    with pytest.raises(ModelAuthError):
        await DeepgramFluxSTT(api_key=KEY, connect=rejected).open(16_000).finish()


async def test_transcribe_streams_a_finished_turn_in_chunks():
    async def deepgram(connection, seen):
        async for frame in frames(connection, seen):
            if isinstance(frame, str):
                await connection.send(turn("EndOfTurn", "One more.", 0.8))
                await connection.close()

    async with serve(deepgram) as (base, seen):
        assert await flux(base).transcribe(pcm(4000), 16_000) == "One more."
    assert seen.audio() == pcm(4000)
    assert max(len(f) for f in seen.frames if isinstance(f, bytes)) == 3200  # 100 ms
