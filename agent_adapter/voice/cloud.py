"""Hosted speech providers for `VoiceSession`: OpenAI, ElevenLabs and Cartesia.

Each class fits the `STT` or `TTS` port and streams over plain HTTP (httpx). Keys come
from the constructor or the provider's usual environment variable, read on the first
call, and never reach a log or an error message. Every voice is 16-bit mono PCM at
`sample_rate`. Deepgram streaming speech-to-text is in `agent_adapter.voice.deepgram`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agent_adapter.voice._http import HttpClient, api_key, post, stream_pcm, wav

_PCM_RATES = frozenset({8_000, 16_000, 22_050, 24_000, 44_100, 48_000})


class _Hosted:
    provider = ""
    env = ""

    def __init__(self, api_key: str | None, client: Any, timeout: float) -> None:
        self._api_key = api_key
        self._http = HttpClient(client, timeout)

    def _key(self) -> str:
        return api_key(self._api_key, self.env, self.provider)

    async def aclose(self) -> None:
        """Close the HTTP client this object made (one you passed in stays open)."""
        await self._http.aclose()


class OpenAISTT(_Hosted):
    """Speech-to-text with OpenAI's transcription endpoint (`OPENAI_API_KEY`).

    `base_url` points it at any OpenAI-compatible server instead.
    """

    provider = "openai"
    env = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str = "gpt-4o-mini-transcribe",
        *,
        language: str | None = None,
        prompt: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        client: Any = None,
        timeout: float = 30,
    ) -> None:
        super().__init__(api_key, client, timeout)
        self.model = model
        self.language = language
        self.prompt = prompt
        self.base_url = base_url.rstrip("/")

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        data = {"model": self.model, "response_format": "json"}
        data.update({k: v for k, v in (("language", self.language), ("prompt", self.prompt)) if v})
        response = await post(
            self._http,
            self.provider,
            self.model,
            f"{self.base_url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {self._key()}"},
            data=data,
            files={"file": ("turn.wav", wav(pcm, sample_rate), "audio/wav")},
        )
        return str(response.json().get("text", "")).strip()


class OpenAITTS(_Hosted):
    """Text-to-speech with OpenAI's speech endpoint, streamed as raw 24 kHz PCM."""

    provider = "openai"
    env = "OPENAI_API_KEY"
    sample_rate = 24_000  # fixed by the API for response_format "pcm"

    def __init__(
        self,
        model: str = "gpt-4o-mini-tts",
        *,
        voice: str = "coral",
        instructions: str | None = None,
        speed: float | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        client: Any = None,
        timeout: float = 30,
    ) -> None:
        super().__init__(api_key, client, timeout)
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.speed = speed
        self.base_url = base_url.rstrip("/")

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        body: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
        }
        if self.instructions:
            body["instructions"] = self.instructions
        if self.speed is not None:
            body["speed"] = self.speed
        headers = {"Authorization": f"Bearer {self._key()}"}
        url = f"{self.base_url}/audio/speech"
        async for chunk in stream_pcm(
            self._http, self.provider, self.model, url, headers=headers, body=body
        ):
            yield chunk


class ElevenLabsTTS(_Hosted):
    """Text-to-speech with ElevenLabs' streaming endpoint (`ELEVENLABS_API_KEY`).

    `voice_id` is a voice from your ElevenLabs library.
    """

    provider = "elevenlabs"
    env = "ELEVENLABS_API_KEY"

    def __init__(
        self,
        voice_id: str,
        *,
        model: str = "eleven_flash_v2_5",
        sample_rate: int = 24_000,
        language: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.elevenlabs.io",
        client: Any = None,
        timeout: float = 30,
    ) -> None:
        super().__init__(api_key, client, timeout)
        self.voice_id = voice_id
        self.model = model
        self.sample_rate = _rate(sample_rate, _PCM_RATES | {32_000})
        self.language = language
        self.base_url = base_url.rstrip("/")

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        body: dict[str, Any] = {"text": text, "model_id": self.model}
        if self.language:
            body["language_code"] = self.language
        url = f"{self.base_url}/v1/text-to-speech/{self.voice_id}/stream"
        async for chunk in stream_pcm(
            self._http,
            self.provider,
            self.model,
            url,
            headers={"xi-api-key": self._key()},
            body=body,
            params={"output_format": f"pcm_{self.sample_rate}"},
        ):
            yield chunk


class CartesiaTTS(_Hosted):
    """Text-to-speech with Cartesia Sonic over its bytes endpoint (`CARTESIA_API_KEY`).

    `voice_id` is a Cartesia voice id; `version` is the `Cartesia-Version` header.
    """

    provider = "cartesia"
    env = "CARTESIA_API_KEY"

    def __init__(
        self,
        voice_id: str,
        *,
        model: str = "sonic-3",
        sample_rate: int = 24_000,
        language: str | None = "en",
        version: str = "2026-08-14",
        api_key: str | None = None,
        base_url: str = "https://api.cartesia.ai",
        client: Any = None,
        timeout: float = 30,
    ) -> None:
        super().__init__(api_key, client, timeout)
        self.voice_id = voice_id
        self.model = model
        self.sample_rate = _rate(sample_rate, _PCM_RATES)
        self.language = language
        self.version = version
        self.base_url = base_url.rstrip("/")

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        body: dict[str, Any] = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
        }
        if self.language:
            body["language"] = self.language
        headers = {"Authorization": f"Bearer {self._key()}", "Cartesia-Version": self.version}
        url = f"{self.base_url}/tts/bytes"
        async for chunk in stream_pcm(
            self._http, self.provider, self.model, url, headers=headers, body=body
        ):
            yield chunk


def _rate(sample_rate: int, allowed: frozenset[int]) -> int:
    if sample_rate not in allowed:
        raise ValueError(f"sample_rate must be one of {sorted(allowed)}")
    return sample_rate


__all__ = ["CartesiaTTS", "ElevenLabsTTS", "OpenAISTT", "OpenAITTS"]
