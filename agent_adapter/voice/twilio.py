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

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.voice._phone_stream import PhoneStream, b64, field, parse_json, to_b64
from agent_adapter.voice.pcm import ulaw_decode, ulaw_encode
from agent_adapter.voice.socket import TextSocket

if TYPE_CHECKING:
    from agent_adapter.voice.twilio_calls import TwilioCalls


class TwilioStream(PhoneStream):
    """An `AudioTransport` over a Twilio Media Streams WebSocket (any `TextSocket`).

    The identity fields (`call_sid`, `stream_sid`, `parameters`) come from the first `start`
    message only; trust them only if the host verified `X-Twilio-Signature` on the upgrade.
    """

    network = "twilio"
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
        super().__init__(frame_ms=frame_ms, lead_ms=lead_ms)
        self.socket = socket
        self.calls = calls
        self.stream_sid = ""
        self.call_sid = ""
        self.parameters: dict[str, Any] = {}  # the <Parameter> values from your TwiML

    async def hang_up(self) -> None:
        """End the call: through the Calls API when `calls` is set, and end `frames()`.

        With no `calls`, only our side ends; Twilio then runs the TwiML after `<Connect>`,
        which hangs up when there is none.
        """
        try:
            if self.calls is not None and self.call_sid:
                await self.calls.hang_up(self.call_sid)
        finally:
            await super().hang_up()

    async def transfer(self, to: str, *, caller_id: str | None = None) -> None:
        """Hand the caller to a person at `to`; Twilio ends this stream as it dials."""
        if self.calls is None:
            raise ConfigError("transfer() needs TwilioStream(..., calls=TwilioCalls())")
        await self.calls.transfer(self.call_sid, to, caller_id=caller_id)

    @property
    def _ready(self) -> bool:
        return bool(self.stream_sid)

    def _messages(self) -> AsyncIterator[Any]:
        return aiter(self.socket)

    def _handle(self, raw: Any) -> bytes | None:
        message = parse_json(raw)
        event = message.get("event")
        if event == "start":
            self._started(message)
        elif event == "media":
            media = field(message, "media")
            if media and media.get("track", "inbound") == "inbound":
                return ulaw_decode(b64(media.get("payload")))
        elif event == "mark":
            self._confirmed(field(message, "mark").get("name"))
        elif event == "dtmf":
            self._digit(str(field(message, "dtmf").get("digit", "")))
        elif event == "stop":
            return None
        return b""

    def _started(self, message: dict[str, Any]) -> None:
        if not self._first_start():
            return
        start = field(message, "start")
        self.stream_sid = str(message.get("streamSid") or start.get("streamSid") or "")
        self.call_sid = str(start.get("callSid") or "")
        params = start.get("customParameters")
        self.parameters = dict(params) if isinstance(params, dict) else {}

    async def _media(self, pcm: bytes) -> None:
        await self._send(
            self._stamped({"event": "media", "media": {"payload": to_b64(ulaw_encode(pcm))}})
        )

    def _mark_message(self, name: str) -> dict[str, Any]:
        return self._stamped({"event": "mark", "mark": {"name": name}})

    def _clear_message(self) -> dict[str, Any]:
        return self._stamped({"event": "clear"})

    def _stamped(self, message: dict[str, Any]) -> dict[str, Any]:
        return {**message, "streamSid": self.stream_sid}

    async def _write(self, data: str | bytes) -> None:
        await self.socket.send(str(data))


__all__ = ["TwilioStream"]
