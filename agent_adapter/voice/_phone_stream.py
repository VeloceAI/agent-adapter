"""What every phone media stream shares: paced playout, marks, cuts, keypad and hang-up.

Each network module (`twilio`, `telnyx`, `plivo`, `vonage`) only reads and writes its own
message format. Audio goes out at the pace it plays; after each line a mark (named
`aa-line-<id>`) is queued behind it, and the line counts as played only when the network
echoes the mark back, so the history holds what the caller really heard.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import binascii
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, ClassVar

from agent_adapter.core.errors import log_failure
from agent_adapter.voice.playout import Pacer, PlayoutBuffer, Segment, fire

MAX_MESSAGE = 256_000
DIGITS = frozenset("0123456789*#ABCD")
_MARK = "aa-line-"


class PhoneStream(abc.ABC):
    """An `AudioTransport` over one network's media-stream WebSocket."""

    network: ClassVar[str]
    in_rate: int
    out_rate: int

    def __init__(self, *, frame_ms: int, lead_ms: int) -> None:
        self.on_digit: list[Callable[[str], None]] = []
        self._log = logging.getLogger(type(self).__module__)
        self._frame_ms = frame_ms
        self._lead_ms = lead_ms
        self._has_started = False
        self._buffer: PlayoutBuffer | None = None
        self._pacer: asyncio.Task[None] | None = None
        self._ended = asyncio.Event()

    async def start(self, buffer: PlayoutBuffer) -> None:
        self._buffer = buffer
        buffer.confirm_playout = True
        buffer.on_drained.append(self._mark)
        buffer.on_clear.append(self._clear)
        pacer = Pacer(buffer, self._media, frame_ms=self._frame_ms, lead_ms=self._lead_ms)
        self._pacer = asyncio.create_task(pacer.run())

    async def frames(self) -> AsyncIterator[bytes]:
        messages = self._messages()
        while (raw := await self._next(messages)) is not None:
            pcm = self._handle(raw)
            if pcm is None:
                return
            if pcm:
                yield pcm

    async def hang_up(self) -> None:
        """End our side of the call: `frames()` stops; the host then closes the socket."""
        self._ended.set()

    async def close(self) -> None:
        task, self._pacer = self._pacer, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @property
    @abc.abstractmethod
    def _ready(self) -> bool:
        """The network told us what it needs to address messages (a stream id)."""

    @abc.abstractmethod
    def _messages(self) -> AsyncIterator[Any]: ...

    @abc.abstractmethod
    def _handle(self, raw: Any) -> bytes | None:
        """Caller audio as PCM16 (b"" for none), or None once the stream has stopped."""

    @abc.abstractmethod
    async def _media(self, pcm: bytes) -> None: ...

    @abc.abstractmethod
    def _mark_message(self, name: str) -> dict[str, Any]: ...

    @abc.abstractmethod
    def _clear_message(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    async def _write(self, data: str | bytes) -> None: ...

    def _first_start(self) -> bool:
        if self._has_started:
            self._log.warning("%s: a second 'start' on one stream was ignored", self.network)
            return False
        self._has_started = True
        return True

    async def _next(self, messages: AsyncIterator[Any]) -> Any:
        """The next message, or None once the socket ends or `hang_up()` is called."""
        if self._ended.is_set():
            return None
        receive = asyncio.ensure_future(anext(messages, None))
        ended = asyncio.ensure_future(self._ended.wait())
        try:
            await asyncio.wait({receive, ended}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            ended.cancel()
            if not receive.done():
                receive.cancel()
                with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                    await receive
        return None if receive.cancelled() else receive.result()

    def _digit(self, digit: Any) -> None:
        if not isinstance(digit, str) or digit not in DIGITS:
            return
        for callback in list(self.on_digit):
            try:
                callback(digit)
            except Exception as exc:
                log_failure(self._log, logging.WARNING, exc, "keypad callback failed")

    def _confirmed(self, name: Any) -> None:
        text = name if isinstance(name, str) else ""
        line = text[len(_MARK) :]
        if text.startswith(_MARK) and line.isdigit() and self._buffer is not None:
            self._buffer.confirm(int(line))

    def _mark(self, segment: Segment) -> None:
        if not self._ready and self._buffer is not None:
            self._buffer.confirm(segment.id)  # nothing reached the network, so nothing to wait for
            return
        fire(self._send(self._mark_message(f"{_MARK}{segment.id}")))

    def _clear(self) -> None:
        fire(self._send(self._clear_message()))

    async def _send(self, message: dict[str, Any]) -> None:
        what = str(message.get("event") or message.get("action"))
        if not self._ready:
            self._log.debug("%s: %s dropped before the stream started", self.network, what)
            return
        await self._deliver(json.dumps(message), what)

    async def _deliver(self, data: str | bytes, what: str) -> None:
        try:
            await self._write(data)
        except Exception as exc:  # the network hung up; frames() ends the call
            self._log.debug("%s: %s not sent (%s)", self.network, what, type(exc).__name__)


def parse_json(raw: Any) -> dict[str, Any]:
    """A JSON object from a text message, or {} for anything else."""
    if not isinstance(raw, str) or len(raw) > MAX_MESSAGE:
        return {}
    try:
        message = json.loads(raw)
    except (ValueError, RecursionError):  # RecursionError: JSON nested too deep
        return {}
    return message if isinstance(message, dict) else {}


def field(message: dict[str, Any], key: str) -> dict[str, Any]:
    """`message[key]` when it is an object, else {}."""
    value = message.get(key)
    return value if isinstance(value, dict) else {}


def b64(payload: Any) -> bytes:
    """Strict base64 to bytes; b"" when it is not valid base64."""
    try:
        return base64.b64decode(str(payload or ""), validate=True)
    except (binascii.Error, ValueError):
        return b""


def to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
