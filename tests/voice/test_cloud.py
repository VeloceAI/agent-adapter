import json
import sys

import pytest

httpx = pytest.importorskip("httpx")

from agent_adapter.errors import (  # noqa: E402
    ConfigError,
    MissingDependency,
    ModelAuthError,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from agent_adapter.voice.cloud import CartesiaTTS, ElevenLabsTTS, OpenAISTT, OpenAITTS  # noqa: E402
from agent_adapter.voice.media import STT, TTS  # noqa: E402

KEY = "sk-test-not-a-real-key"


class Recorder:
    """A fake provider: records each request and answers with `reply(request)`."""

    def __init__(self, reply):
        self.reply = reply
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.AsyncClient:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self.reply(request)

        return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def pcm_reply(*chunks: bytes):
    return lambda request: httpx.Response(200, stream=httpx.ByteStream(b"".join(chunks)))


async def collect(tts) -> bytes:
    return b"".join([chunk async for chunk in tts.synthesize("Hello there.")])


async def test_openai_stt_uploads_a_wav_and_reads_the_text():
    fake = Recorder(lambda r: httpx.Response(200, json={"text": " booked \n"}))
    stt = OpenAISTT(language="en", prompt="shows", api_key=KEY, client=fake.client())
    assert isinstance(stt, STT)
    assert await stt.transcribe(b"\x01\x00" * 160, 16_000) == "booked"
    request = fake.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/audio/transcriptions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = request.content
    assert b'name="model"' in body and b"gpt-4o-mini-transcribe" in body
    assert b'name="language"' in body and b'name="prompt"' in body
    assert b'filename="turn.wav"' in body and b"RIFF" in body and b"WAVE" in body


async def test_openai_tts_streams_pcm_with_its_options():
    fake = Recorder(pcm_reply(b"\x01\x02\x03\x04"))
    tts = OpenAITTS(
        voice="marin", instructions="Warm.", speed=1.1, api_key=KEY, client=fake.client()
    )
    assert isinstance(tts, TTS) and tts.sample_rate == 24_000
    assert await collect(tts) == b"\x01\x02\x03\x04"
    body = json.loads(fake.requests[0].content)
    assert body == {
        "model": "gpt-4o-mini-tts",
        "input": "Hello there.",
        "voice": "marin",
        "response_format": "pcm",
        "instructions": "Warm.",
        "speed": 1.1,
    }


async def test_elevenlabs_asks_for_raw_pcm_at_its_rate():
    fake = Recorder(pcm_reply(b"\x00\x01" * 8))
    tts = ElevenLabsTTS(
        "voice-1", sample_rate=16_000, language="en", api_key=KEY, client=fake.client()
    )
    assert len(await collect(tts)) == 16
    request = fake.requests[0]
    assert request.url.path == "/v1/text-to-speech/voice-1/stream"
    assert request.url.params["output_format"] == "pcm_16000"
    assert request.headers["xi-api-key"] == KEY
    body = json.loads(request.content)
    assert body == {"text": "Hello there.", "model_id": "eleven_flash_v2_5", "language_code": "en"}


async def test_cartesia_sends_its_version_and_raw_output_format():
    fake = Recorder(pcm_reply(b"\x00\x01"))
    tts = CartesiaTTS("voice-2", sample_rate=22_050, api_key=KEY, client=fake.client())
    await collect(tts)
    request = fake.requests[0]
    assert str(request.url) == "https://api.cartesia.ai/tts/bytes"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["cartesia-version"] == "2026-08-14"
    body = json.loads(request.content)
    assert body["voice"] == {"mode": "id", "id": "voice-2"} and body["language"] == "en"
    assert body["output_format"] == {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 22_050,
    }


async def test_chunks_of_odd_length_are_joined_into_whole_samples():
    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in (b"\x01", b"\x02\x03", b"\x04\x05\x06"):
                yield chunk

    fake = Recorder(lambda r: httpx.Response(200, stream=Chunks()))
    tts = OpenAITTS(api_key=KEY, client=fake.client())
    chunks = [chunk async for chunk in tts.synthesize("hi")]
    assert all(len(c) % 2 == 0 for c in chunks) and b"".join(chunks) == b"\x01\x02\x03\x04\x05\x06"


async def test_errors_are_typed_and_never_carry_the_key():
    def reply(request):
        return httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "2"})

    tts = ElevenLabsTTS("v", api_key=KEY, client=Recorder(reply).client())
    with pytest.raises(ModelRateLimited) as caught:
        await collect(tts)
    assert caught.value.retry_after == 2 and caught.value.provider == "elevenlabs"
    assert KEY not in str(caught.value)

    def denied(request):
        return httpx.Response(401, text="bad key", headers={"x-request-id": "req-9"})

    stt = OpenAISTT(api_key=KEY, client=Recorder(denied).client())
    with pytest.raises(ModelAuthError) as caught_auth:
        await stt.transcribe(b"\x00\x00", 16_000)
    assert caught_auth.value.request_id == "req-9" and KEY not in str(caught_auth.value)


async def test_network_failures_become_timeout_and_unavailable():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ModelTimeout):
        await collect(CartesiaTTS("v", api_key=KEY, client=Recorder(timeout).client()))
    with pytest.raises(ModelUnavailable):
        await OpenAISTT(api_key=KEY, client=Recorder(refused).client()).transcribe(b"", 8_000)


async def test_keys_come_from_the_environment(monkeypatch):
    fake = Recorder(pcm_reply(b"\x00\x00"))
    monkeypatch.setenv("CARTESIA_API_KEY", "env-key")
    await collect(CartesiaTTS("v", client=fake.client()))
    assert fake.requests[0].headers["authorization"] == "Bearer env-key"
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ELEVENLABS_API_KEY"):
        await collect(ElevenLabsTTS("v", client=fake.client()))


def test_unsupported_sample_rates_are_refused():
    with pytest.raises(ValueError, match="sample_rate"):
        ElevenLabsTTS("v", sample_rate=11_025)
    with pytest.raises(ValueError, match="sample_rate"):
        CartesiaTTS("v", sample_rate=32_000)


async def test_an_own_client_is_made_and_closed(monkeypatch):
    tts = OpenAITTS(api_key=KEY, base_url="http://localhost:9/v1/")
    client = tts._http.get()
    assert isinstance(client, httpx.AsyncClient) and tts._http.get() is client
    await tts.aclose()
    assert client.is_closed
    mine = httpx.AsyncClient()
    shared = OpenAISTT(api_key=KEY, client=mine)
    await shared.aclose()
    assert not mine.is_closed
    await mine.aclose()


async def test_a_missing_httpx_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(MissingDependency, match="voice-cloud"):
        await collect(OpenAITTS(api_key=KEY))
