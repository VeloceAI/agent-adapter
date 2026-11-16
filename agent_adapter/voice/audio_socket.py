"""`AudioSocket`: raw audio over one WebSocket, for browsers, apps and phone gateways.

| From the client | Meaning |
|---|---|
| binary message | the caller's audio: 16-bit little-endian mono PCM at `in_rate` |
| `{"type": "hangup"}` | the call is over |

| To the client | Meaning |
|---|---|
| binary message | the agent's audio: PCM16 at `out_rate`; play it as it arrives |
| `{"type": "clear"}` | drop any audio not yet played (the caller cut in) |

Audio goes out at the pace it plays, `lead_ms` ahead, so the client needs only a small
jitter buffer and a cut stops what the caller hears within that much.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from agent_adapter.voice.playout import Pacer, PlayoutBuffer, fire

logger = logging.getLogger(__name__)
_MAX_TEXT = 4_096


class BinarySocket(Protocol):
    """A WebSocket that sends and receives both text and binary messages.

    A `websockets` connection is one as it is; wrap other servers with `binary_socket`.
    """

    async def send(self, message: str | bytes) -> None: ...

    def __aiter__(self) -> AsyncIterator[str | bytes]: ...


class _Wrapped:
    def __init__(
        self,
        send: Callable[[str | bytes], Awaitable[Any]],
        messages: AsyncIterator[str | bytes],
    ) -> None:
        self._send = send
        self._messages = messages

    async def send(self, message: str | bytes) -> None:
        await self._send(message)

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self._messages


def binary_socket(
    send: Callable[[str | bytes], Awaitable[Any]], messages: AsyncIterator[str | bytes]
) -> BinarySocket:
    """A `BinarySocket` from a send function and an async iterator of received messages."""
    return _Wrapped(send, messages)


async def starlette_messages(websocket: Any) -> AsyncIterator[str | bytes]:
    """The messages of a Starlette or FastAPI WebSocket, text or binary, until it closes."""
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return
        data = message.get("bytes")
        yield data if data is not None else str(message.get("text") or "")


class AudioSocket:
    """An `AudioTransport` over a `BinarySocket`."""

    def __init__(
        self,
        socket: BinarySocket,
        *,
        in_rate: int = 16_000,
        out_rate: int = 16_000,
        frame_ms: int = 20,
        lead_ms: int = 60,
    ) -> None:
        self.socket = socket
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._frame_ms = frame_ms
        self._lead_ms = lead_ms
        self._pacer: asyncio.Task[None] | None = None

    async def start(self, buffer: PlayoutBuffer) -> None:
        pacer = Pacer(buffer, self._send, frame_ms=self._frame_ms, lead_ms=self._lead_ms)
        buffer.on_clear.append(self._clear)
        self._pacer = asyncio.create_task(pacer.run())

    async def frames(self) -> AsyncIterator[bytes]:
        async for message in self.socket:
            if isinstance(message, bytes | bytearray):
                yield bytes(message)
            elif _control(message).get("type") == "hangup":
                return

    async def close(self) -> None:
        task, self._pacer = self._pacer, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _send(self, pcm: bytes) -> None:
        try:
            await self.socket.send(pcm)
        except Exception as exc:  # the client went away; frames() ends the call
            logger.debug("audio socket: send failed (%s)", type(exc).__name__)

    def _clear(self) -> None:
        fire(self._send_text({"type": "clear"}))

    async def _send_text(self, message: dict[str, Any]) -> None:
        try:
            await self.socket.send(json.dumps(message))
        except Exception as exc:
            logger.debug("audio socket: %s not sent (%s)", message.get("type"), type(exc).__name__)


def _control(message: Any) -> dict[str, Any]:
    if not isinstance(message, str) or len(message) > _MAX_TEXT:
        return {}
    try:
        parsed = json.loads(message)
    except (ValueError, RecursionError):  # RecursionError: JSON nested too deep
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = ["AudioSocket", "BinarySocket", "binary_socket", "starlette_messages"]
