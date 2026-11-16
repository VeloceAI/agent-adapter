import asyncio
import base64
import json
import sys
import types
from collections.abc import AsyncIterator

import pytest

from agent_adapter.voice.audio_socket import AudioSocket, binary_socket, starlette_messages
from agent_adapter.voice.pcm import ulaw_encode
from agent_adapter.voice.playout import PlayoutBuffer
from agent_adapter.voice.socket import text_socket
from agent_adapter.voice.testing import settle, tone
from agent_adapter.voice.twilio import TwilioStream


class Far:
    """The far end of a socket: what it sends us, and what we sent it."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[object] = asyncio.Queue()
        self.sent: list[str | bytes] = []

    async def _messages(self) -> AsyncIterator:
        while (item := await self.inbox.get()) is not None:
            yield item

    async def _receive(self, message) -> None:
        self.sent.append(message)

    def texts(self) -> list[dict]:
        return [json.loads(m) for m in self.sent if isinstance(m, str)]


async def collect(frames: AsyncIterator[bytes]) -> list[bytes]:
    return [frame async for frame in frames]


async def test_audio_socket_reads_binary_and_hangs_up_on_text():
    far = Far()
    transport = AudioSocket(binary_socket(far._receive, far._messages()))
    for item in (
        b"\x01\x02",
        "not json",
        json.dumps({"type": "other"}),
        "x" * 5000,
        "[" * 4000,  # nested past the recursion limit
        json.dumps([1]),
        json.dumps({"type": "hangup"}),
        b"never",
    ):
        far.inbox.put_nowait(item)
    assert await collect(transport.frames()) == [b"\x01\x02"]


async def test_audio_socket_paces_audio_out_and_sends_clear():
    far = Far()
    transport = AudioSocket(binary_socket(far._receive, far._messages()), lead_ms=40)
    buffer = PlayoutBuffer(16_000)
    await transport.start(buffer)
    seg = buffer.open()
    buffer.write(seg, tone(1000))
    buffer.finish(seg)
    await asyncio.sleep(0.1)
    sent = sum(len(m) for m in far.sent if isinstance(m, bytes)) / 32
    assert 60 <= sent <= 240
    buffer.clear()
    await settle()
    assert {"type": "clear"} in far.texts()
    await transport.close()


async def test_audio_socket_survives_a_dead_client():
    async def broken(message):
        raise ConnectionError("gone")

    transport = AudioSocket(binary_socket(broken, Far()._messages()))
    await transport._send(b"\x00\x00")
    await transport._send_text({"type": "clear"})


async def test_starlette_messages_until_disconnect():
    class WS:
        def __init__(self) -> None:
            self.items = [
                {"type": "websocket.receive", "bytes": b"\x01"},
                {"type": "websocket.receive", "text": "hi"},
                {"type": "websocket.disconnect"},
            ]

        async def receive(self):
            return self.items.pop(0)

    assert [m async for m in starlette_messages(WS())] == [b"\x01", "hi"]


def twilio(event: str, **fields) -> str:
    return json.dumps({"event": event, "streamSid": "MZ1", **fields})


async def test_twilio_decodes_inbound_audio_and_reads_the_start():
    far = Far()
    transport = TwilioStream(text_socket(far._receive, far._messages()))
    payload = base64.b64encode(ulaw_encode(tone(20, rate=8_000))).decode()
    start = {"callSid": "CA1", "customParameters": {"lang": "en"}}
    second_start = {"callSid": "CA2", "customParameters": {"lang": "xx"}}
    for item in (
        twilio("connected"),
        twilio("start", start=start),
        twilio("media", media={"track": "inbound", "payload": payload}),
        twilio("media", media={"track": "outbound", "payload": payload}),
        twilio("media", media={"payload": "!!not base64"}),
        twilio("dtmf"),
        "garbage",
        "[" * 100_000,  # nested past the recursion limit
        json.dumps({"event": "start", "streamSid": "MZ2", "start": second_start}),
        twilio("stop"),
        twilio("media"),
    ):
        far.inbox.put_nowait(item)
    frames = await collect(transport.frames())
    assert len(frames) == 1 and len(frames[0]) == 320
    assert (transport.stream_sid, transport.call_sid) == ("MZ1", "CA1")  # only the first start
    assert transport.parameters == {"lang": "en"}


async def test_twilio_line_counts_as_played_only_when_its_mark_comes_back():
    far = Far()
    transport = TwilioStream(text_socket(far._receive, far._messages()), lead_ms=200)
    transport.stream_sid = "MZ1"
    buffer = PlayoutBuffer(8_000)
    await transport.start(buffer)
    seg = buffer.open()
    buffer.write(seg, tone(40, rate=8_000))
    buffer.finish(seg)
    await asyncio.sleep(0.08)
    sent = far.texts()
    assert sent[0]["event"] == "media" and sent[0]["streamSid"] == "MZ1"
    marks = [m for m in sent if m["event"] == "mark"]
    assert marks and not seg.done.done()
    far.inbox.put_nowait(twilio("mark", mark={"name": marks[0]["mark"]["name"]}))
    far.inbox.put_nowait(twilio("mark", mark={"name": "someone-else"}))
    far.inbox.put_nowait(None)
    await collect(transport.frames())
    await settle()
    assert seg.done.done()
    buffer.clear()
    await settle()
    assert far.texts()[-1] == {"event": "clear", "streamSid": "MZ1"}
    await transport.close()


async def test_twilio_before_start_drops_messages_and_never_waits_on_marks():
    far = Far()
    transport = TwilioStream(text_socket(far._receive, far._messages()))
    buffer = PlayoutBuffer(8_000)
    await transport.start(buffer)
    seg = buffer.open()
    buffer.write(seg, tone(20, rate=8_000))
    buffer.finish(seg)
    await asyncio.sleep(0.05)
    await settle()
    assert far.sent == [] and seg.done.done()
    await transport.close()


async def test_twilio_survives_a_dead_socket():
    async def broken(message):
        raise ConnectionError("gone")

    transport = TwilioStream(text_socket(broken, Far()._messages()))
    transport.stream_sid = "MZ1"
    await transport._send({"event": "clear"})


class FakeStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.state: list[str] = []

    def start(self) -> None:
        self.state.append("started")

    def stop(self) -> None:
        self.state.append("stopped")

    def close(self) -> None:
        self.state.append("closed")


@pytest.fixture
def fake_sounddevice(monkeypatch):
    module = types.SimpleNamespace(RawInputStream=FakeStream, RawOutputStream=FakeStream)
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


async def test_local_audio_wires_the_mic_and_the_speaker_callbacks(fake_sounddevice):
    from agent_adapter.voice.local import LocalAudio

    audio = LocalAudio(out_rate=16_000)
    buffer = PlayoutBuffer(16_000)
    await audio.start(buffer)
    mic, speaker = audio._streams
    assert mic.state == ["started"] and mic.kwargs["samplerate"] == 16_000
    mic.kwargs["callback"](memoryview(b"\x01\x02"), 1, None, None)
    seg = buffer.open()
    buffer.write(seg, b"\x07" * 4)
    out = bytearray(8)
    speaker.kwargs["callback"](out, 4, None, None)
    assert bytes(out) == b"\x07" * 4 + bytes(4)
    await settle()
    audio.hang_up()
    assert await collect(audio.frames()) == [b"\x01\x02"]
    await audio.close()
    assert mic.state == ["started", "stopped", "closed"]


async def test_local_audio_without_speakers_opens_only_the_mic(fake_sounddevice):
    from agent_adapter.voice.local import LocalAudio

    audio = LocalAudio(speakers=False)
    await audio.start(PlayoutBuffer(24_000))
    assert len(audio._streams) == 1
    await audio.close()


async def test_local_audio_names_the_extra_when_sounddevice_is_missing(monkeypatch):
    from agent_adapter.errors import MissingDependency
    from agent_adapter.voice.local import LocalAudio

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(MissingDependency, match="voice-local"):
        await LocalAudio().start(PlayoutBuffer(24_000))
