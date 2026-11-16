"""`DeepgramSTT`: Deepgram speech-to-text that hears each turn as it is spoken.

`open()` streams the turn over Deepgram's live WebSocket while the caller talks; at the
end of the turn we send `Finalize` and `CloseStream`, so only the last moment is left to
transcribe. `transcribe()` sends a finished turn in one HTTP request. The key comes from
`api_key=` or `DEEPGRAM_API_KEY` and travels only in the `Authorization` header.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol
from urllib.parse import urlencode

from agent_adapter.core.errors import AdapterError, MissingDependency, ModelError
from agent_adapter.models.errors import class_for_status
from agent_adapter.voice._http import HttpClient, api_key, post, wav


class LiveSocket(Protocol):
    """The part of a WebSocket client connection the live stream uses."""

    async def send(self, message: str | bytes) -> None: ...

    def __aiter__(self) -> AsyncIterator[str | bytes]: ...

    async def close(self) -> None: ...


Connect = Callable[[str, dict[str, str]], Awaitable[LiveSocket]]


async def _websocket(url: str, headers: dict[str, str]) -> LiveSocket:
    try:
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise MissingDependency(
            'Deepgram streaming needs websockets: pip install "agent-adapter[voice-cloud]"'
        ) from exc
    socket: LiveSocket = await connect(url, additional_headers=headers)
    return socket


class DeepgramSTT:
    """A `StreamingSTT` on Deepgram (`model` such as `nova-3`).

    `finish_timeout_s` bounds the wait for the final words after the turn ends; past it
    the session falls back to `transcribe()` on the whole turn.
    """

    provider = "deepgram"

    def __init__(
        self,
        model: str = "nova-3",
        *,
        language: str = "en",
        smart_format: bool = True,
        api_key: str | None = None,
        host: str = "api.deepgram.com",
        finish_timeout_s: float = 5.0,
        client: Any = None,
        connect: Connect | None = None,
        timeout: float = 30,
    ) -> None:
        self.model = model
        self.language = language
        self.smart_format = smart_format
        self.host = host
        self.finish_timeout_s = finish_timeout_s
        self._api_key = api_key
        self._http = HttpClient(client, timeout)
        self._connect = connect or _websocket

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {api_key(self._api_key, 'DEEPGRAM_API_KEY', 'Deepgram')}"}

    def _query(self, **extra: str | int) -> str:
        params: dict[str, str | int] = {"model": self.model, "language": self.language}
        if self.smart_format:
            params["smart_format"] = "true"
        return urlencode({**params, **extra})

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        response = await post(
            self._http,
            self.provider,
            self.model,
            f"https://{self.host}/v1/listen?{self._query()}",
            headers={**self._headers(), "Content-Type": "audio/wav"},
            content=wav(pcm, sample_rate),
        )
        channels = (response.json().get("results") or {}).get("channels") or [{}]
        return _transcript(channels[0])

    def open(self, sample_rate: int) -> _DeepgramStream:
        query = self._query(encoding="linear16", sample_rate=sample_rate, channels=1)
        return _DeepgramStream(self, f"wss://{self.host}/v1/listen?{query}")

    async def aclose(self) -> None:
        """Close the HTTP client this object made (one you passed in stays open)."""
        await self._http.aclose()


class _DeepgramStream:
    """One turn on the live socket: audio goes out as it comes; final results come back."""

    def __init__(self, stt: DeepgramSTT, url: str) -> None:
        self._stt = stt
        self._url = url
        self._audio: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._finals: list[str] = []
        self._task = asyncio.ensure_future(self._run())

    def feed(self, pcm: bytes) -> None:
        if pcm:
            self._audio.put_nowait(pcm)

    async def finish(self) -> str:
        self._audio.put_nowait(None)
        return await asyncio.wait_for(self._task, self._stt.finish_timeout_s)

    async def cancel(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task

    async def _run(self) -> str:
        socket = await self._open()
        reader = asyncio.ensure_future(self._read(socket))
        try:
            while (chunk := await self._audio.get()) is not None:
                await socket.send(chunk)
            await socket.send(json.dumps({"type": "Finalize"}))
            await socket.send(json.dumps({"type": "CloseStream"}))
            await reader
            return " ".join(self._finals)
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            with contextlib.suppress(Exception):
                await socket.close()

    async def _open(self) -> LiveSocket:
        stt = self._stt
        headers = stt._headers()
        try:
            return await stt._connect(self._url, headers)
        except AdapterError:
            raise
        except Exception as exc:  # websockets raises InvalidStatus with the HTTP response
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise class_for_status(status)(
                f"Deepgram live connection failed ({type(exc).__name__})",
                provider=stt.provider,
                model=stt.model,
                status_code=status,
            ) from exc

    async def _read(self, socket: LiveSocket) -> None:
        async for raw in socket:
            message = _parse(raw)
            kind = message.get("type")
            if kind == "Results":
                if message.get("is_final"):
                    text = _transcript(message.get("channel"))
                    if text:
                        self._finals.append(text)
                if message.get("from_finalize"):
                    return
            elif kind == "Metadata":
                return
            elif kind == "Error":
                detail = str(message.get("description") or message.get("message") or "")[:300]
                raise ModelError(
                    f"Deepgram error: {detail}", provider="deepgram", model=self._stt.model
                )


def _parse(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        message = json.loads(raw)
    except ValueError:
        return {}
    return message if isinstance(message, dict) else {}


def _transcript(channel: Any) -> str:
    alternatives = channel.get("alternatives") if isinstance(channel, dict) else None
    if not isinstance(alternatives, list) or not alternatives:
        return ""
    first = alternatives[0]
    return str(first.get("transcript", "")).strip() if isinstance(first, dict) else ""


__all__ = ["Connect", "DeepgramSTT", "LiveSocket"]
