"""A voice agent over one WebSocket, at the transcript level: the client hears and speaks.

The client (a browser with the Web Speech API, a phone gateway, a native app) does speech
to text and text to speech; this side decides. Messages are JSON text:

| From the client | Meaning |
|---|---|
| `{"type": "speech_started"}` | the caller started talking |
| `{"type": "transcript", "text": "...", "final": true}` | a transcript; only final ones count |
| `{"type": "played", "id": 3}` | line 3 played to the end |
| `{"type": "hangup"}` | the call is over |

| To the client | Meaning |
|---|---|
| `{"type": "say", "id": 3, "text": "..."}` | speak this line, then send `played` |
| `{"type": "stop"}` | stop speaking now and drop queued lines |

A line counts as played only after its `played` message, so the history after a cut holds
exactly what the caller heard. A client that never acknowledges is trusted after
`played_timeout_s`. Send `speech_started` as each utterance begins: an answer to a spoken
yes/no question counts only if its speech started after the question played.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from agent_adapter.voice.pump import EventPump

if TYPE_CHECKING:
    from agent_adapter.voice.agent import VoiceAgent

logger = logging.getLogger(__name__)
_MAX_MESSAGE = 64_000


class TextSocket(Protocol):
    """A text WebSocket: `send(text)` and async iteration over received texts.

    The `websockets` client and server connections are one as they are; wrap Starlette or
    FastAPI sockets with `text_socket(ws.send_text, ws.iter_text())`.
    """

    async def send(self, text: str) -> None: ...

    def __aiter__(self) -> AsyncIterator[str]: ...


class _Wrapped:
    def __init__(self, send: Callable[[str], Awaitable[Any]], messages: AsyncIterator[str]):
        self._send = send
        self._messages = messages

    async def send(self, text: str) -> None:
        await self._send(text)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._messages


def text_socket(send: Callable[[str], Awaitable[Any]], messages: AsyncIterator[str]) -> TextSocket:
    """A `TextSocket` from a send function and an async iterator of received texts."""
    return _Wrapped(send, messages)


class VoiceSocket:
    """Runs a `VoiceAgent` over a `TextSocket`; give the agent `voice_socket.speaker`."""

    def __init__(self, socket: TextSocket, *, played_timeout_s: float = 30.0) -> None:
        self.socket = socket
        self.speaker = _SocketSpeaker(socket, played_timeout_s)

    async def run(self, agent: VoiceAgent) -> None:
        """Read the client until it hangs up or disconnects; then close the agent."""
        pump = EventPump(agent)
        try:
            async for raw in self.socket:
                if self._handle(raw, pump) == "hangup":
                    break
            await pump.idle()
        finally:
            await pump.aclose()
            await agent.close()

    def _handle(self, raw: Any, pump: EventPump) -> str:
        message = _parse(raw)
        kind = message.get("type")
        if kind == "speech_started":
            pump.started_speaking()
        elif kind == "transcript" and message.get("final") is True:
            pump.said(str(message.get("text", "")))
        elif kind == "played":
            self.speaker.played(message.get("id"))
        elif kind not in ("transcript", "hangup"):
            logger.debug("voice socket: ignored message type %r", kind)
        return str(kind)


def _parse(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or len(raw) > _MAX_MESSAGE:
        return {}
    try:
        message = json.loads(raw)
    except (ValueError, RecursionError):  # RecursionError: JSON nested too deep
        return {}
    return message if isinstance(message, dict) else {}


class _SocketSpeaker:
    """Sends `say` and waits for the client's `played`; `stop` tells the client to hush."""

    def __init__(self, socket: TextSocket, timeout_s: float) -> None:
        self._socket = socket
        self._timeout_s = timeout_s
        self._next_id = 0
        self._waiting: dict[int, asyncio.Future[None]] = {}

    async def say(self, text: str) -> None:
        self._next_id += 1
        line = self._next_id
        done = asyncio.get_running_loop().create_future()
        self._waiting[line] = done
        try:
            await self._socket.send(json.dumps({"type": "say", "id": line, "text": text}))
            await asyncio.wait_for(done, self._timeout_s)
        except TimeoutError:
            logger.debug("voice socket: no 'played' for line %s; counted as played", line)
        finally:
            self._waiting.pop(line, None)

    def played(self, line: Any) -> None:
        done = self._waiting.get(line) if isinstance(line, int) else None
        if done is not None and not done.done():
            done.set_result(None)

    async def stop(self) -> None:
        for done in self._waiting.values():
            done.cancel()
        try:
            await self._socket.send(json.dumps({"type": "stop"}))
        except Exception as exc:  # the client may already be gone
            logger.debug("voice socket: stop not sent (%s)", type(exc).__name__)


__all__ = ["TextSocket", "VoiceSocket", "text_socket"]
