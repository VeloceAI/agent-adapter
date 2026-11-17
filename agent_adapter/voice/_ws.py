"""WebSocket plumbing shared by the hosted speech providers: connect, type the errors,
and run one speech-to-text turn as a stream."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from agent_adapter.core.errors import (
    AdapterError,
    MissingDependency,
    ModelError,
    ModelUnavailable,
)
from agent_adapter.models.errors import MAX_MESSAGE, class_for_status
from agent_adapter.voice._http import api_key
from agent_adapter.voice.ports import TurnEvent, TurnSignal

Frame = str | bytes


class LiveSocket(Protocol):
    """The part of a WebSocket client connection the providers use."""

    async def send(self, message: Frame) -> None: ...

    def __aiter__(self) -> AsyncIterator[Frame]: ...

    async def close(self) -> None: ...


Connect = Callable[[str, dict[str, str]], Awaitable[LiveSocket]]


_SECRET_HEADERS = frozenset({"authorization", "x-api-key", "xi-api-key"})


class _Redacting(logging.LoggerAdapter):  # type: ignore[type-arg]
    """websockets logs every handshake header at DEBUG, the API key among them."""

    def log(self, level: int, msg: object, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and str(args[0]).lower() in _SECRET_HEADERS:
            args = (args[0], "[redacted]")
        super().log(level, msg, *args, **kwargs)


_LOGGER = _Redacting(logging.getLogger("websockets.client"), {})


async def websocket(url: str, headers: dict[str, str]) -> LiveSocket:
    try:
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise MissingDependency(
            'hosted speech streaming needs websockets: pip install "agent-adapter[voice-cloud]"'
        ) from exc
    socket: LiveSocket = await connect(
        url, additional_headers=headers, max_size=2**24, logger=_LOGGER
    )
    return socket


async def open_socket(
    connect: Connect, url: str, headers: dict[str, str], provider: str, model: str
) -> LiveSocket:
    try:
        return await connect(url, headers)
    except AdapterError:
        raise
    except Exception as exc:  # websockets raises InvalidStatus with the HTTP response
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise class_for_status(status)(
            f"{provider} live connection failed ({type(exc).__name__})",
            provider=provider,
            model=model,
            status_code=status,
        ) from exc


async def send(socket: LiveSocket, frame: Frame, provider: str, model: str) -> None:
    try:
        await socket.send(frame)
    except AdapterError:
        raise
    except Exception as exc:  # the provider closed the socket
        message = f"{provider} live connection lost ({type(exc).__name__})"
        raise ModelUnavailable(message, provider=provider, model=model) from exc


def parse(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str | bytes):
        return {}
    try:
        message = json.loads(raw)
    except (ValueError, RecursionError):  # RecursionError: JSON nested too deep
        return {}
    return message if isinstance(message, dict) else {}


def failure(provider: str, model: str, detail: Any, status: Any = None) -> ModelError:
    code = status if isinstance(status, int) else None
    text = str(detail or "unknown error")[:MAX_MESSAGE]
    return class_for_status(code)(
        f"{provider} error: {text}", provider=provider, model=model, status_code=code
    )


def dumps(message: dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"))


class LiveSTT:
    """Base of a streaming speech-to-text provider: one WebSocket per turn.

    `transcribe()` streams a finished turn the same way, in `chunk_ms` pieces.
    """

    provider = ""
    env = ""
    chunk_ms = 100
    realtime_only = False  # the provider refuses audio sent faster than it plays

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        url: str,
        connect: Connect | None,
        finish_timeout_s: float,
    ) -> None:
        self.model = model
        self.url = url
        self.finish_timeout_s = finish_timeout_s
        self._api_key = api_key
        self._connect = connect or websocket

    def _key(self) -> str:
        return api_key(self._api_key, self.env, self.provider)

    def _headers(self) -> dict[str, str]:
        return {}

    def open(self, sample_rate: int) -> LiveSTTStream:
        raise NotImplementedError

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        stream = self.open(sample_rate)
        step = max(2, sample_rate * self.chunk_ms // 1000 * 2)
        for start in range(0, len(pcm), step):
            stream.feed(pcm[start : start + step])
            if self.realtime_only:
                await asyncio.sleep(self.chunk_ms / 1000)
        return await stream.finish()

    async def aclose(self) -> None:
        """Nothing to close: each turn opens and closes its own socket."""


class LiveSTTStream:
    """One turn on a provider socket: audio goes out as it comes, text comes back.

    Subclasses say what to send (`_opening`, `_frame`, `_closing`) and read each reply
    in `_handle`, which returns True when the provider has said its last word.
    """

    def __init__(self, stt: LiveSTT, url: str) -> None:
        self._stt = stt
        self._url = url
        self._audio: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._listener: Callable[[TurnSignal], None] | None = None
        self._task = asyncio.ensure_future(self._run())

    def feed(self, pcm: bytes) -> None:
        if pcm:
            self._audio.put_nowait(pcm)

    def on_turn(self, callback: Callable[[TurnSignal], None]) -> None:
        self._listener = callback

    async def finish(self) -> str:
        self._audio.put_nowait(None)
        return await asyncio.wait_for(self._task, self._stt.finish_timeout_s)

    async def cancel(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task

    def _emit(self, event: TurnEvent, text: str, confidence: float | None = None) -> None:
        if self._listener is not None:
            self._listener(TurnSignal(event, text, confidence))

    def _fail(self, detail: Any, status: Any = None) -> ModelError:
        return failure(self._stt.provider, self._stt.model, detail, status)

    def _opening(self) -> list[Frame]:
        return []

    def _frame(self, pcm: bytes) -> Frame:
        return pcm

    def _closing(self) -> list[Frame]:
        return []

    def _settled(self) -> bool:
        """The final text is in already, so `finish` need not wait for the provider."""
        return False

    def _handle(self, message: Frame) -> bool:
        raise NotImplementedError

    def _result(self) -> str:
        raise NotImplementedError

    async def _run(self) -> str:
        stt = self._stt
        headers = stt._headers()
        socket = await open_socket(stt._connect, self._url, headers, stt.provider, stt.model)
        reader = asyncio.ensure_future(self._read(socket))
        try:
            for frame in self._opening():
                await self._send(socket, frame, reader)
            while (chunk := await self._audio.get()) is not None:
                await self._send(socket, self._frame(chunk), reader)
            for frame in self._closing():
                await self._send(socket, frame, reader)
            if not self._settled():
                await reader
            return self._result()
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            with contextlib.suppress(Exception):
                await socket.close()

    async def _send(self, socket: LiveSocket, frame: Frame, reader: asyncio.Future[None]) -> None:
        if reader.done():  # the provider has had its last word, or failed
            if (error := reader.exception()) is not None:
                raise error
            return
        try:
            await send(socket, frame, self._stt.provider, self._stt.model)
        except ModelUnavailable:
            await asyncio.wait({reader}, timeout=1.0)  # the provider's reason, if it sent one
            if not reader.done():
                raise
            if (error := reader.exception()) is not None:
                raise error from None

    async def _read(self, socket: LiveSocket) -> None:
        async for raw in socket:
            if self._handle(raw):
                return


__all__ = [
    "Connect",
    "Frame",
    "LiveSTT",
    "LiveSTTStream",
    "LiveSocket",
    "dumps",
    "failure",
    "open_socket",
    "parse",
    "send",
    "websocket",
]
