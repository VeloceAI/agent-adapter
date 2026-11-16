import asyncio
import json
import sys

import pytest

httpx = pytest.importorskip("httpx")

from agent_adapter.errors import ConfigError, MissingDependency, ModelAuthError, ModelError  # noqa: E402
from agent_adapter.voice.deepgram import DeepgramSTT  # noqa: E402
from agent_adapter.voice.media import StreamingSTT  # noqa: E402
from agent_adapter.voice.testing import settle  # noqa: E402

KEY = "dg-test-not-a-real-key"


def results(text: str, *, final: bool = True, from_finalize: bool = False) -> str:
    channel = {"alternatives": [{"transcript": text, "confidence": 0.9}]}
    message = {"type": "Results", "channel": channel, "is_final": final}
    return json.dumps({**message, "from_finalize": from_finalize})


class FakeLive:
    """A fake Deepgram live socket: `script` maps a control message type to replies."""

    def __init__(self, script: dict[str, list[str]] | None = None) -> None:
        self.script = script or {}
        self.sent: list[str | bytes] = []
        self.url = ""
        self.headers: dict[str, str] = {}
        self.closed = False
        self._inbox: asyncio.Queue[str | bytes | None] = asyncio.Queue()

    async def connect(self, url: str, headers: dict[str, str]) -> "FakeLive":
        self.url, self.headers = url, headers
        return self

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)
        if isinstance(message, str):
            for reply in self.script.get(json.loads(message)["type"], []):
                self._inbox.put_nowait(reply)

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        while (message := await self._inbox.get()) is not None:
            yield message

    async def close(self) -> None:
        self.closed = True
        self._inbox.put_nowait(None)


def stt(live: FakeLive, **options) -> DeepgramSTT:
    return DeepgramSTT(api_key=KEY, connect=live.connect, **options)


async def test_a_turn_streams_audio_and_returns_the_final_words():
    live = FakeLive(
        {
            "Finalize": [
                results("book the", final=False),
                results("book the"),
                b"\x00binary is ignored",
                "not json",
                results("jazz show", from_finalize=True),
            ]
        }
    )
    deepgram = stt(live)
    assert isinstance(deepgram, StreamingSTT)
    stream = deepgram.open(16_000)
    stream.feed(b"\x01\x00" * 100)
    stream.feed(b"")
    stream.feed(b"\x02\x00" * 100)
    assert await stream.finish() == "book the jazz show"
    assert live.sent[:2] == [b"\x01\x00" * 100, b"\x02\x00" * 100]
    assert [json.loads(m)["type"] for m in live.sent[2:]] == ["Finalize", "CloseStream"]
    assert live.url.startswith("wss://api.deepgram.com/v1/listen?")
    for part in ("model=nova-3", "encoding=linear16", "sample_rate=16000", "channels=1"):
        assert part in live.url
    assert "smart_format=true" in live.url and KEY not in live.url
    assert live.headers == {"Authorization": f"Token {KEY}"} and live.closed


async def test_metadata_or_a_closed_socket_ends_the_turn():
    live = FakeLive({"CloseStream": [results("hi"), json.dumps({"type": "Metadata"})]})
    stream = stt(live, smart_format=False).open(8_000)
    assert await stream.finish() == "hi"
    assert "smart_format" not in live.url

    quiet = FakeLive({"CloseStream": [results("   ")]})
    stream = stt(quiet).open(16_000)
    finishing = asyncio.ensure_future(stream.finish())
    await settle()
    quiet._inbox.put_nowait(None)  # Deepgram closes the socket
    assert await finishing == ""


async def test_a_deepgram_error_message_fails_the_stream():
    live = FakeLive({"Finalize": [json.dumps({"type": "Error", "description": "bad audio"})]})
    stream = stt(live).open(16_000)
    with pytest.raises(ModelError, match="bad audio"):
        await stream.finish()


async def test_connection_failures_are_typed():
    class Rejected(Exception):
        class response:  # websockets' InvalidStatus carries the HTTP response
            status_code = 401

    async def rejected(url, headers):
        raise Rejected()

    async def broken(url, headers):
        raise OSError("no route")

    with pytest.raises(ModelAuthError):
        await DeepgramSTT(api_key=KEY, connect=rejected).open(16_000).finish()
    with pytest.raises(ModelError, match="OSError"):
        await DeepgramSTT(api_key=KEY, connect=broken).open(16_000).finish()


async def test_a_slow_finish_times_out_and_cancel_closes_the_socket():
    live = FakeLive()
    stream = stt(live, finish_timeout_s=0.05).open(16_000)
    with pytest.raises(TimeoutError):
        await stream.finish()
    assert live.closed

    other = FakeLive()
    stream = stt(other).open(16_000)
    stream.feed(b"\x00\x00")
    await settle()
    await stream.cancel()
    assert other.closed


async def test_a_missing_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="DEEPGRAM_API_KEY"):
        await DeepgramSTT(connect=FakeLive().connect).open(16_000).finish()


async def test_a_finished_turn_is_sent_in_one_request():
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        alternatives = [{"transcript": " two tickets "}]
        return httpx.Response(200, json={"results": {"channels": [{"alternatives": alternatives}]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    deepgram = DeepgramSTT(api_key=KEY, client=client)
    assert await deepgram.transcribe(b"\x01\x00" * 160, 16_000) == "two tickets"
    request = seen[0]
    assert request.url.path == "/v1/listen" and request.url.params["model"] == "nova-3"
    assert request.headers["authorization"] == f"Token {KEY}"
    assert request.headers["content-type"] == "audio/wav" and request.content[:4] == b"RIFF"

    empty = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    assert await DeepgramSTT(api_key=KEY, client=empty).transcribe(b"", 16_000) == ""
    await deepgram.aclose()


async def test_the_default_connection_uses_websockets_with_the_auth_header():
    server_mod = pytest.importorskip("websockets.asyncio.server")
    got: dict[str, object] = {}

    async def handler(connection):
        got["auth"] = connection.request.headers["Authorization"]
        got["path"] = connection.request.path
        async for message in connection:
            if isinstance(message, bytes):
                got["audio"] = message
            elif json.loads(message)["type"] == "Finalize":
                await connection.send(results("hello", from_finalize=True))

    async with server_mod.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        deepgram = DeepgramSTT(api_key=KEY, host=f"127.0.0.1:{port}")
        deepgram._query = lambda **extra: "encoding=linear16"  # plain ws:// for the test
        stream = deepgram.open(16_000)
        stream._url = f"ws://127.0.0.1:{port}/v1/listen?encoding=linear16"
        stream.feed(b"\x05\x00")
        assert await stream.finish() == "hello"
    assert got == {
        "auth": f"Token {KEY}",
        "path": "/v1/listen?encoding=linear16",
        "audio": b"\x05\x00",
    }


async def test_a_missing_websockets_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "websockets.asyncio.client", None)
    with pytest.raises(MissingDependency, match="voice-cloud"):
        await DeepgramSTT(api_key=KEY).open(16_000).finish()
