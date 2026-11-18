"""The base of the WebSocket text-to-speech providers: text goes in as the model writes it,
audio comes back as soon as each piece is ready.

A provider that multiplexes (`shared = True`, such as Cartesia) keeps one socket open and
routes each reply by context id; the others open one socket per reply, and `warm()`
opens the next one ahead so a reply does not wait for the handshake.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from agent_adapter.core.errors import ModelError
from agent_adapter.voice._http import api_key
from agent_adapter.voice._ws import (
    Connect,
    Frame,
    LiveSocket,
    failure,
    open_socket,
    send,
    websocket,
)
from agent_adapter.voice.ports import WordTime

DONE = object()  # the reply is complete
Reply = Any  # bytes (audio), list[WordTime], ModelError, DONE, or a tuple of these


class _Line:
    """One open socket and its reader, routing each reply to its stream."""

    def __init__(self, tts: LiveTTS, socket: LiveSocket) -> None:
        self.tts = tts
        self.socket = socket
        self.routes: dict[str, asyncio.Queue[Reply]] = {}
        self.closed = False
        self.reader = asyncio.ensure_future(self._read())

    async def send(self, frame: Frame) -> None:
        await send(self.socket, frame, self.tts.provider, self.tts.model)

    async def _read(self) -> None:
        error: ModelError | None = None
        try:
            async for raw in self.socket:
                context, reply = self.tts._decode(raw)
                if reply is None:
                    continue
                queue = self.routes.get(context) if context else self._only()
                if queue is not None:
                    for item in reply if isinstance(reply, tuple) else (reply,):
                        queue.put_nowait(item)
        except Exception as exc:  # the socket dropped
            error = failure(*self._names(), f"live connection lost ({type(exc).__name__})")
        finally:
            self.closed = True
            closed = error or failure(*self._names(), "the provider closed the connection")
            for queue in self.routes.values():
                queue.put_nowait(closed)

    def _names(self) -> tuple[str, str]:
        return self.tts.provider, self.tts.model

    def _only(self) -> asyncio.Queue[Reply] | None:
        return next(iter(self.routes.values())) if len(self.routes) == 1 else None

    async def close(self) -> None:
        self.closed = True
        self.reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self.reader
        with contextlib.suppress(Exception):
            await self.socket.close()


class LiveTTSStream:
    """A `TTSStream` on one provider context."""

    def __init__(self, tts: LiveTTS) -> None:
        self.words: list[WordTime] = []
        self.context_id = uuid.uuid4().hex
        self._tts = tts
        self._queue: asyncio.Queue[Reply] = asyncio.Queue()
        self._line: _Line | None = None
        self._opening = asyncio.Lock()
        self._ended = False
        self._finished = False

    async def _open(self) -> _Line:
        async with self._opening:
            if self._line is None:
                line = await self._tts._take_line()
                line.routes[self.context_id] = self._queue
                self._line = line
                for frame in self._tts._first(self.context_id):
                    await line.send(frame)
            return self._line

    async def push(self, text: str, *, flush: bool = False) -> None:
        if self._ended or (not text and not flush):
            return
        line = await self._open()
        for frame in self._tts._text(self.context_id, text, flush):
            await line.send(frame)

    async def end(self) -> None:
        if self._ended:
            return
        line = await self._open()
        self._ended = True
        for frame in self._tts._end(self.context_id):
            await line.send(frame)

    async def audio(self) -> AsyncIterator[bytes]:
        if not self._finished:  # a cancelled stream only drains what is queued
            await self._open()
        odd, complete, played = b"", False, 0
        per_second = 2 * self._tts.sample_rate
        try:
            while True:
                reply = await self._queue.get()
                if isinstance(reply, bytes):
                    data, odd = odd + reply, b""
                    if len(data) % 2:
                        data, odd = data[:-1], data[-1:]
                    if data:
                        played += len(data)
                        yield data
                elif isinstance(reply, list):
                    shift = played / per_second if self._tts.chunk_relative_words else 0.0
                    self.words += [WordTime(w.word, w.start + shift, w.end + shift) for w in reply]
                elif isinstance(reply, Exception):
                    raise reply
                else:
                    complete = True
                    return
        finally:
            await self._release(cancel=not complete)

    async def cancel(self) -> None:
        """Stop this reply now; a shared socket stays open for the next one."""
        await self._release(cancel=True)
        self._queue.put_nowait(DONE)

    async def _release(self, *, cancel: bool) -> None:
        if self._finished:
            return
        self._finished = self._ended = True
        line, self._line = self._line, None
        if line is None:
            return
        line.routes.pop(self.context_id, None)
        if self._tts.shared and cancel and not line.closed:
            with contextlib.suppress(Exception):
                for frame in self._tts._cancel(self.context_id):
                    await line.send(frame)
        if not self._tts.shared:
            await line.close()


class LiveTTS:
    """Base of a WebSocket TTS. Subclasses build the frames and read the replies."""

    provider = ""
    env = ""
    shared = False
    chunk_relative_words = False  # word times count from the audio chunk they come with
    spare_ttl_s = 15.0  # providers drop idle sockets; an older spare is not reused

    def __init__(
        self,
        model: str,
        *,
        sample_rate: int,
        api_key: str | None,
        url: str,
        connect: Connect | None,
    ) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.url = url
        self._api_key = api_key
        self._connect_fn = connect
        self._line: _Line | None = None
        self._spare: tuple[float, LiveSocket] | None = None

    def _key(self) -> str:
        return api_key(self._api_key, self.env, self.provider)

    def open(self) -> LiveTTSStream:
        return LiveTTSStream(self)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        stream = self.open()
        try:
            await stream.push(text, flush=True)
            await stream.end()
            async for chunk in stream.audio():
                yield chunk
        finally:
            await stream.cancel()

    async def warm(self) -> None:
        """Open the socket now, so the next reply skips the connection handshake."""
        if self.shared:
            await self._shared_line()
        elif self._spare is None:
            self._spare = (time.monotonic(), await self._dial())

    async def aclose(self) -> None:
        """Close the open sockets."""
        line, self._line = self._line, None
        if line is not None:
            await line.close()
        spare, self._spare = self._spare, None
        if spare is not None:
            with contextlib.suppress(Exception):
                await spare[1].close()

    async def _dial(self) -> LiveSocket:
        connect = self._connect_fn or websocket
        url, headers = self._url(), self._headers()
        socket = await open_socket(connect, url, headers, self.provider, self.model)
        try:
            await self._ready(socket)
        except BaseException:
            with contextlib.suppress(Exception):
                await socket.close()
            raise
        return socket

    async def _ready(self, socket: LiveSocket) -> None:
        """Wait for the provider's greeting, if it sends one before taking requests."""

    async def _shared_line(self) -> _Line:
        if self._line is None or self._line.closed:
            self._line = _Line(self, await self._dial())
        return self._line

    async def _take_line(self) -> _Line:
        if self.shared:
            return await self._shared_line()
        spare, self._spare = self._spare, None
        if spare is not None and time.monotonic() - spare[0] < self.spare_ttl_s:
            return _Line(self, spare[1])
        if spare is not None:
            with contextlib.suppress(Exception):
                await spare[1].close()
        return _Line(self, await self._dial())

    def _url(self) -> str:
        return self.url

    def _headers(self) -> dict[str, str]:
        return {}

    def _first(self, context: str) -> list[Frame]:
        return []

    def _text(self, context: str, text: str, flush: bool) -> list[Frame]:
        raise NotImplementedError

    def _end(self, context: str) -> list[Frame]:
        raise NotImplementedError

    def _cancel(self, context: str) -> list[Frame]:
        return []

    def _decode(self, raw: Frame) -> tuple[str | None, Reply]:
        """Read one reply: (its context id or None, audio / words / error / DONE / None)."""
        raise NotImplementedError


def words_from_chars(chars: list[Any], starts: list[Any], ends: list[Any]) -> list[WordTime]:
    """Join per-character times into words (ElevenLabs sends characters)."""
    words: list[WordTime] = []
    word, start, end = "", 0.0, 0.0
    for char, s, e in zip(chars, starts, ends, strict=False):
        if str(char).isspace():
            if word:
                words.append(WordTime(word, start, end))
            word = ""
            continue
        if not word:
            start = float(s)
        word, end = word + str(char), float(e)
    if word:
        words.append(WordTime(word, start, end))
    return words


def b64_audio(data: Any) -> bytes | None:
    if not isinstance(data, str):
        return None
    try:
        return base64.b64decode(data)
    except (binascii.Error, ValueError):
        return None


def word_times(stamps: Any) -> list[WordTime] | None:
    """`{"words": [...], "start": [...], "end": [...]}` (seconds) as `WordTime`s."""
    if not isinstance(stamps, dict):
        return None
    words, starts, ends = (stamps.get(k) or [] for k in ("words", "start", "end"))
    try:
        return [
            WordTime(str(w), float(s), float(e))
            for w, s, e in zip(words, starts, ends, strict=False)
        ]
    except (TypeError, ValueError):
        return None


__all__ = [
    "DONE",
    "LiveTTS",
    "LiveTTSStream",
    "Reply",
    "b64_audio",
    "word_times",
    "words_from_chars",
]
