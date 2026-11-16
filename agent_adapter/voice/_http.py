"""Shared pieces of the hosted speech providers and the Twilio REST helper."""

from __future__ import annotations

import contextlib
import io
import os
import wave
from collections.abc import AsyncIterator, Mapping
from typing import Any

from agent_adapter.core.errors import (
    ConfigError,
    MissingDependency,
    ModelTimeout,
    ModelUnavailable,
)
from agent_adapter.models.errors import MAX_MESSAGE, class_for_status, retry_after


def httpx_module() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise MissingDependency(
            'hosted speech needs httpx: pip install "agent-adapter[voice-cloud]"'
        ) from exc
    return httpx


def api_key(given: str | None, env: str, provider: str) -> str:
    key = given or os.environ.get(env, "")
    if not key:
        raise ConfigError(f"{provider} needs an API key: pass api_key= or set {env}")
    return key


def wav(pcm: bytes, sample_rate: int) -> bytes:
    """16-bit mono PCM in a WAV container, for upload APIs that take a file."""
    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(pcm)
    return out.getvalue()


class HttpClient:
    """One lazily made `httpx.AsyncClient` per provider object, or the one you pass in."""

    def __init__(self, client: Any, timeout: float) -> None:
        self._client = client
        self._own = client is None
        self._timeout = timeout

    def get(self) -> Any:
        if self._client is None:
            httpx = httpx_module()
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=10))
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and self._own:
            await client.aclose()


async def check(response: Any, provider: str, model: str = "") -> None:
    """Raise the typed `ModelError` for a failed response; the body is read if streamed."""
    if response.status_code < 400:
        return
    await response.aread()
    text = response.text[:MAX_MESSAGE]
    raise class_for_status(response.status_code)(
        f"{provider} returned HTTP {response.status_code}: {text}",
        provider=provider,
        model=model,
        status_code=response.status_code,
        retry_after=retry_after(response.headers),
        request_id=_request_id(response.headers),
    )


def _request_id(headers: Mapping[str, str]) -> str:
    for name in ("x-request-id", "request-id", "dg-request-id", "twilio-request-id"):
        if headers.get(name):
            return str(headers[name])
    return ""


@contextlib.asynccontextmanager
async def network_errors(provider: str, model: str = "") -> AsyncIterator[None]:
    """Turn httpx timeouts and connection errors into `ModelTimeout` and `ModelUnavailable`."""
    httpx = httpx_module()
    try:
        yield
    except httpx.TimeoutException as exc:
        raise ModelTimeout(f"{provider} timed out", provider=provider, model=model) from exc
    except httpx.TransportError as exc:
        message = f"{provider} connection failed ({type(exc).__name__})"
        raise ModelUnavailable(message, provider=provider, model=model) from exc


async def post(http: HttpClient, provider: str, model: str, url: str, **kwargs: Any) -> Any:
    """POST and return the response, raising the typed error for a failure."""
    async with network_errors(provider, model):
        response = await http.get().post(url, **kwargs)
    await check(response, provider, model)
    return response


async def stream_pcm(
    http: HttpClient,
    provider: str,
    model: str,
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
    params: dict[str, str] | None = None,
) -> AsyncIterator[bytes]:
    """POST `body` and yield the raw PCM16 reply as it streams, in even-length chunks."""
    odd = b""
    async with network_errors(provider, model):
        request = http.get().stream("POST", url, headers=headers, json=body, params=params)
        async with request as response:
            await check(response, provider, model)
            async for chunk in response.aiter_bytes():
                data, odd = odd + chunk, b""
                if len(data) % 2:
                    data, odd = data[:-1], data[-1:]
                if data:
                    yield data


__all__ = [
    "HttpClient",
    "api_key",
    "check",
    "httpx_module",
    "network_errors",
    "post",
    "stream_pcm",
    "wav",
]
