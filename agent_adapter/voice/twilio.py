"""`TwilioStream`: a phone call through Twilio Media Streams (bidirectional `<Connect><Stream>`).

Twilio sends the caller's audio as 8 kHz μ-law in JSON `media` messages over a WebSocket,
and plays back what we send the same way. On a cut we send `clear`, which empties Twilio's
playback buffer. After each line we send a `mark`; Twilio echoes it once the line has
really played, so the history holds what the caller heard on the phone.

Keypad presses arrive as `dtmf` messages; each digit goes to the `on_digit` callbacks
(`VoiceSession` turns them into words for the agent). `hang_up()` and `transfer()` use the
Calls REST API through the `TwilioCalls` you pass as `calls=`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.voice.pcm import ulaw_decode, ulaw_encode
from agent_adapter.voice.playout import Pacer, PlayoutBuffer, Segment, fire
from agent_adapter.voice.socket import TextSocket

if TYPE_CHECKING:
    from agent_adapter.voice.twilio_calls import TwilioCalls

logger = logging.getLogger(__name__)
_MAX_MESSAGE = 256_000
_MARK = "aa-line-"
_DIGITS = frozenset("0123456789*#ABCD")


class TwilioStream:
    """An `AudioTransport` over a Twilio Media Streams WebSocket (any `TextSocket`)."""

    in_rate = 8_000
    out_rate = 8_000

    def __init__(
        self,
        socket: TextSocket,
        *,
        calls: TwilioCalls | None = None,
        frame_ms: int = 20,
        lead_ms: int = 80,
    ) -> None:
        self.socket = socket
        self.calls = calls
        self.on_digit: list[Callable[[str], None]] = []
        self.stream_sid = ""
        self.call_sid = ""
        self.parameters: dict[str, Any] = {}  # the <Parameter> values from your TwiML
        self._frame_ms = frame_ms
        self._lead_ms = lead_ms
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
        messages = aiter(self.socket)
        while (raw := await self._next(messages)) is not None:
            message = _parse(raw)
            event = message.get("event")
            if event == "start":
                self._started(message)
            elif event == "media":
                pcm = self._audio(message)
                if pcm:
                    yield pcm
            elif event == "mark":
                self._marked(message)
            elif event == "dtmf":
                self._digit(message)
            elif event == "stop":
                return

    async def hang_up(self) -> None:
        """End the call: through the Calls API when `calls` is set, and end `frames()`.

        With no `calls`, only our side ends; Twilio then runs the TwiML after `<Connect>`,
        which hangs up when there is none.
        """
        try:
            if self.calls is not None and self.call_sid:
                await self.calls.hang_up(self.call_sid)
        finally:
            self._ended.set()

    async def transfer(self, to: str, *, caller_id: str | None = None) -> None:
        """Hand the caller to a person at `to`; Twilio ends this stream as it dials."""
        if self.calls is None:
            raise ConfigError("transfer() needs TwilioStream(..., calls=TwilioCalls())")
        await self.calls.transfer(self.call_sid, to, caller_id=caller_id)

    async def close(self) -> None:
        task, self._pacer = self._pacer, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _started(self, message: dict[str, Any]) -> None:
        start = message.get("start")
        start = start if isinstance(start, dict) else {}
        self.stream_sid = str(message.get("streamSid") or start.get("streamSid") or "")
        self.call_sid = str(start.get("callSid") or "")
        params = start.get("customParameters")
        self.parameters = dict(params) if isinstance(params, dict) else {}

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

    def _digit(self, message: dict[str, Any]) -> None:
        dtmf = message.get("dtmf")
        digit = str(dtmf.get("digit", "")) if isinstance(dtmf, dict) else ""
        if digit not in _DIGITS:
            return
        for callback in list(self.on_digit):
            try:
                callback(digit)
            except Exception as exc:
                log_failure(logger, logging.WARNING, exc, "keypad callback failed")

    def _audio(self, message: dict[str, Any]) -> bytes:
        media = message.get("media")
        if not isinstance(media, dict) or media.get("track", "inbound") != "inbound":
            return b""
        try:
            return ulaw_decode(base64.b64decode(str(media.get("payload", "")), validate=True))
        except (binascii.Error, ValueError):
            return b""

    def _marked(self, message: dict[str, Any]) -> None:
        mark = message.get("mark")
        name = str(mark.get("name", "")) if isinstance(mark, dict) else ""
        if name.startswith(_MARK) and name[len(_MARK) :].isdigit() and self._buffer is not None:
            self._buffer.confirm(int(name[len(_MARK) :]))

    async def _media(self, pcm: bytes) -> None:
        payload = base64.b64encode(ulaw_encode(pcm)).decode("ascii")
        await self._send({"event": "media", "media": {"payload": payload}})

    def _mark(self, segment: Segment) -> None:
        if not self.stream_sid and self._buffer is not None:
            self._buffer.confirm(segment.id)  # nothing reached Twilio, so nothing to wait for
            return
        fire(self._send({"event": "mark", "mark": {"name": f"{_MARK}{segment.id}"}}))

    def _clear(self) -> None:
        fire(self._send({"event": "clear"}))

    async def _send(self, message: dict[str, Any]) -> None:
        if not self.stream_sid:
            logger.debug("twilio: %s dropped before the stream started", message["event"])
            return
        try:
            await self.socket.send(json.dumps({**message, "streamSid": self.stream_sid}))
        except Exception as exc:  # Twilio hung up; frames() ends the call
            logger.debug("twilio: %s not sent (%s)", message["event"], type(exc).__name__)


def _parse(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or len(raw) > _MAX_MESSAGE:
        return {}
    try:
        message = json.loads(raw)
    except ValueError:
        return {}
    return message if isinstance(message, dict) else {}


__all__ = ["TwilioStream"]
