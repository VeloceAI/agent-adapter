"""Telnyx, Plivo and Vonage streams against the message formats in each network's docs.

Fake sockets only: no account, no network. Inbound messages copy the docs' examples.
"""

import asyncio
import base64
import json
import logging

import pytest

from agent_adapter.errors import ConfigError
from agent_adapter.voice.audio_socket import binary_socket
from agent_adapter.voice.pcm import ulaw_encode
from agent_adapter.voice.playout import PlayoutBuffer, Segment
from agent_adapter.voice.plivo import PlivoStream
from agent_adapter.voice.socket import text_socket
from agent_adapter.voice.telnyx import TelnyxStream
from agent_adapter.voice.testing import settle, tone
from agent_adapter.voice.vonage import VonageStream
from tests.voice.test_transports import Far, collect

STREAM = "32de0dea-53cb-4b21-89a4-9324e3dd7d0f"
ULAW_20MS = base64.b64encode(ulaw_encode(tone(20, rate=8_000))).decode()


def feed(far: Far, *items: object) -> None:
    for item in items:
        far.inbox.put_nowait(item)


async def play_line(transport, rate: int, ms: int = 40) -> tuple[PlayoutBuffer, Segment]:
    buffer = PlayoutBuffer(rate)
    await transport.start(buffer)
    seg = buffer.open()
    buffer.write(seg, tone(ms, rate=rate))
    buffer.finish(seg)
    await asyncio.sleep(0.08)
    return buffer, seg


def telnyx(event: str, **fields) -> str:
    return json.dumps({"event": event, "sequence_number": "2", "stream_id": STREAM, **fields})


def telnyx_stream(**options) -> tuple[Far, TelnyxStream]:
    far = Far()
    return far, TelnyxStream(text_socket(far._receive, far._messages()), **options)


TELNYX_START = {
    "user_id": "3e6f995f-85f7-4705-9741-53b116d28237",
    "call_control_id": "v3:MdI91X4lWFEs7IgbBEOT9M4AigoY08M0WWZFISt1Yw2axZ_IiE4pqg",
    "call_session_id": "e8f7b0e6-7fc8-11ee-a31d-02420a0d4268",
    "from": "+13122010094",
    "to": "+13122123456",
    "tags": ["TAG1"],
    "client_state": "aGF2ZSBhIG5pY2UgZGF5ID1d",
    "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1},
}


async def test_telnyx_reads_start_media_dtmf_and_stop(caplog):
    far, stream = telnyx_stream()
    pressed: list[str] = []
    stream.on_digit.append(pressed.append)
    second = {**TELNYX_START, "call_control_id": "v3:other"}
    error = {"code": 100003, "title": "malformed_frame", "detail": "Received frame was not formed"}
    feed(
        far,
        json.dumps({"event": "connected", "version": "1.0.0"}),
        telnyx("start", start=TELNYX_START),
        telnyx(
            "media",
            media={"track": "inbound", "chunk": "2", "timestamp": "5", "payload": ULAW_20MS},
        ),
        telnyx("media", media={"track": "outbound", "payload": ULAW_20MS}),
        telnyx("media", media={"payload": "!!not base64"}),
        telnyx("dtmf", occurred_at="2025-06-05T08:54:19.698408Z", dtmf={"digit": "1"}),
        telnyx("dtmf", dtmf={"digit": "12"}),
        telnyx("error", payload=error),
        "garbage",
        "[" * 100_000,  # nested past the recursion limit
        json.dumps({"event": "start", "stream_id": "other", "start": second}),
        telnyx("stop", stop={"user_id": "u", "call_control_id": TELNYX_START["call_control_id"]}),
        telnyx("media", media={"payload": ULAW_20MS}),
    )
    with caplog.at_level(logging.WARNING, logger="agent_adapter.voice.telnyx"):
        frames = await collect(stream.frames())
    assert [len(f) for f in frames] == [320]
    assert pressed == ["1"]
    assert stream.stream_id == STREAM and stream.call_control_id == TELNYX_START["call_control_id"]
    assert stream.call_session_id == TELNYX_START["call_session_id"]
    assert stream.client_state == TELNYX_START["client_state"]
    assert stream.start_info["from"] == "+13122010094"
    assert "100003 malformed_frame" in caplog.text and "second 'start'" in caplog.text


async def test_telnyx_sends_media_marks_and_clear_and_waits_for_the_mark():
    far, stream = telnyx_stream(lead_ms=200)
    stream.stream_id = STREAM
    buffer, seg = await play_line(stream, 8_000)
    sent = far.texts()
    assert sent[0].keys() == {"event", "media"} and sent[0]["event"] == "media"
    assert len(base64.b64decode(sent[0]["media"]["payload"])) == 160  # 20 ms of PCMU
    marks = [m for m in sent if m["event"] == "mark"]
    assert marks == [{"event": "mark", "mark": {"name": f"aa-line-{seg.id}"}}]
    assert not seg.done.done()
    feed(far, telnyx("mark", mark={"name": "someone-else"}))
    feed(far, telnyx("mark", mark={"name": marks[0]["mark"]["name"]}), None)
    await collect(stream.frames())
    await settle()
    assert seg.done.done()
    buffer.clear()
    await settle()
    assert far.texts()[-1] == {"event": "clear"}
    await stream.close()


async def test_telnyx_before_start_sends_nothing_and_never_waits():
    far, stream = telnyx_stream()
    _buffer, seg = await play_line(stream, 8_000, ms=20)
    await settle()
    assert far.sent == [] and seg.done.done()
    await stream.close()


async def test_telnyx_hang_up_ends_the_frames_and_survives_a_dead_socket():
    far, stream = telnyx_stream()
    reading = asyncio.ensure_future(collect(stream.frames()))
    await settle()
    await stream.hang_up()
    assert await asyncio.wait_for(reading, 1) == []

    async def broken(message):
        raise ConnectionError("gone")

    dead = TelnyxStream(text_socket(broken, far._messages()))
    dead.stream_id = STREAM
    await dead._send({"event": "clear"})


def plivo(event: str, **fields) -> str:
    return json.dumps({"event": event, "sequenceNumber": 42, "streamId": STREAM, **fields})


def plivo_stream(**options) -> tuple[Far, PlivoStream]:
    far = Far()
    return far, PlivoStream(text_socket(far._receive, far._messages()), **options)


def plivo_start(encoding: str = "audio/x-mulaw", rate: int = 8000) -> str:
    start = {
        "callId": "12345678-1234-1234-1234-123456789abc",
        "streamId": STREAM,
        "accountId": "MAXXXXXXXXXXXXXXXXXX",
        "tracks": ["inbound"],
        "mediaFormat": {"encoding": encoding, "sampleRate": rate},
    }
    return json.dumps(
        {
            "event": "start",
            "sequenceNumber": 1,
            "start": start,
            "extra_headers": "userId=12345;sessionId=abc-xyz;broken",
        }
    )


async def test_plivo_reads_start_mulaw_media_and_dtmf():
    far, stream = plivo_stream(content_type="audio/x-mulaw;rate=8000")
    pressed: list[str] = []
    stream.on_digit.append(pressed.append)
    media = {"track": "inbound", "timestamp": "1705312200000", "chunk": 41, "payload": ULAW_20MS}
    feed(
        far,
        plivo_start(),
        plivo("media", media=media, extra_headers="userId=12345"),
        plivo("media", media={**media, "track": "outbound"}),
        plivo("dtmf", dtmf={"track": "inbound", "digit": "5", "timestamp": "1705312250000"}),
        plivo("dtmf", dtmf={"track": "inbound", "digit": "x"}),
        plivo("clearedAudio"),
        "[" * 100_000,
        None,
    )
    frames = await collect(stream.frames())
    assert [len(f) for f in frames] == [320] and pressed == ["5"]
    assert (stream.stream_id, stream.call_id) == (STREAM, "12345678-1234-1234-1234-123456789abc")
    assert stream.account_id == "MAXXXXXXXXXXXXXXXXXX"
    assert stream.parameters == {"userId": "12345", "sessionId": "abc-xyz"}
    assert (stream.in_rate, stream.out_rate) == (8_000, 8_000)


async def test_plivo_linear_pcm_passes_through_at_16k():
    far, stream = plivo_stream(content_type="audio/x-l16;rate=16000")
    pcm = tone(20)
    payload = base64.b64encode(pcm + b"\x01").decode()  # an odd byte is dropped
    feed(far, plivo_start("audio/x-l16", 16000), plivo("media", media={"payload": payload}), None)
    assert await collect(stream.frames()) == [pcm]
    assert stream.in_rate == 16_000


async def test_plivo_plays_audio_checkpoints_and_clears():
    far, stream = plivo_stream(lead_ms=200)  # Plivo's default: audio/x-l16;rate=8000
    stream.stream_id = STREAM
    buffer, seg = await play_line(stream, 8_000)
    sent = far.texts()
    play = sent[0]
    assert play["event"] == "playAudio" and play.keys() == {"event", "media"}
    assert play["media"]["contentType"] == "audio/x-l16" and play["media"]["sampleRate"] == 8000
    assert len(base64.b64decode(play["media"]["payload"])) == 320  # 20 ms of PCM16 at 8 kHz
    checkpoints = [m for m in sent if m["event"] == "checkpoint"]
    assert checkpoints == [{"event": "checkpoint", "streamId": STREAM, "name": f"aa-line-{seg.id}"}]
    assert not seg.done.done()
    feed(far, plivo("playedStream", name=checkpoints[0]["name"]), None)
    await collect(stream.frames())
    await settle()
    assert seg.done.done()
    buffer.clear()
    await settle()
    assert far.texts()[-1] == {"event": "clearAudio", "streamId": STREAM}
    await stream.close()


async def test_plivo_mulaw_playback_and_content_type_checks(caplog):
    far, stream = plivo_stream(content_type="audio/x-mulaw; rate=8000", lead_ms=200)
    stream.stream_id = STREAM
    _buffer, _seg = await play_line(stream, 8_000, ms=20)
    media = far.texts()[0]["media"]
    assert media["contentType"] == "audio/x-mulaw"
    assert len(base64.b64decode(media["payload"])) == 160
    await stream.close()
    with pytest.raises(ConfigError, match="content_type"):
        PlivoStream(text_socket(far._receive, far._messages()), content_type="audio/wav")
    far, mismatch = plivo_stream()
    feed(far, plivo_start("audio/x-mulaw"), None)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.voice.plivo"):
        await collect(mismatch.frames())
    assert "content_type says audio/x-l16" in caplog.text


def vonage_stream(**options) -> tuple[Far, VonageStream]:
    far = Far()
    return far, VonageStream(binary_socket(far._receive, far._messages()), **options)


async def test_vonage_reads_connected_binary_audio_and_dtmf():
    far, stream = vonage_stream()
    pressed: list[str] = []
    stream.on_digit.append(pressed.append)
    connected = {
        "event": "websocket:connected",
        "content-type": "audio/l16;rate=16000",
        "prop1": "value1",
        "prop2": "value2",
    }
    pcm = tone(20)
    feed(
        far,
        json.dumps(connected),
        pcm,
        pcm + b"\x00",  # an odd byte is dropped
        json.dumps({"event": "websocket:dtmf", "digit": "5", "duration": 260}),
        json.dumps({"event": "websocket:dtmf", "digit": "55"}),
        json.dumps({"event": "websocket:cleared"}),
        json.dumps({**connected, "prop1": "spoofed"}),
        "[" * 100_000,
        None,
    )
    assert await collect(stream.frames()) == [pcm, pcm]
    assert len(pcm) == 640 and pressed == ["5"]
    assert stream.content_type == "audio/l16;rate=16000"
    assert stream.headers == {"prop1": "value1", "prop2": "value2"}


async def test_vonage_sends_binary_frames_notify_and_clear():
    far, stream = vonage_stream(lead_ms=200)
    buffer, seg = await play_line(stream, 16_000, ms=50)
    audio = [m for m in far.sent if isinstance(m, bytes)]
    assert [len(m) for m in audio] == [640, 640, 640]  # the 10 ms tail padded to 20 ms
    assert audio[2][320:] == bytes(320)
    notify = far.texts()
    assert notify == [{"action": "notify", "payload": {"mark": f"aa-line-{seg.id}"}}]
    assert far.sent.index(json.dumps(notify[0])) > far.sent.index(audio[-1])
    assert not seg.done.done()
    feed(far, json.dumps({"event": "websocket:notify", "payload": {"mark": "other"}}))
    feed(far, json.dumps({"event": "websocket:notify", "payload": notify[0]["payload"]}), None)
    await collect(stream.frames())
    await settle()
    assert seg.done.done()
    buffer.clear()
    await settle()
    assert far.texts()[-1] == {"action": "clear"}
    await stream.close()


async def test_vonage_rate_checks(caplog):
    with pytest.raises(ConfigError, match="rate"):
        vonage_stream(rate=44_100)
    far, stream = vonage_stream(rate=8_000)
    assert (stream.in_rate, stream.out_rate) == (8_000, 8_000)
    event = {"event": "websocket:connected", "content-type": "audio/l16;rate=16000"}
    feed(far, json.dumps(event), None)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.voice.vonage"):
        await collect(stream.frames())
    assert "expects rate=8000" in caplog.text
