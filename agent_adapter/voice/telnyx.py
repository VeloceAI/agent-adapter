"""`TelnyxStream`: a phone call through Telnyx bidirectional Media Streaming.

Start the stream from Call Control (`stream_url`, `stream_bidirectional_mode="rtp"`,
`stream_bidirectional_codec="PCMU"`) or TeXML (`<Stream bidirectionalMode="rtp">`). Telnyx
sends the caller's audio as base64 8 kHz μ-law in JSON `media` messages; we send ours back
the same way, `clear` on a cut, and a `mark` after each line, which Telnyx echoes when the
line has played. Keypad presses arrive as `dtmf` messages and reach `on_digit`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agent_adapter.voice._phone_stream import PhoneStream, b64, field, parse_json, to_b64
from agent_adapter.voice.pcm import ulaw_decode, ulaw_encode
from agent_adapter.voice.socket import TextSocket


class TelnyxStream(PhoneStream):
    """An `AudioTransport` over a Telnyx Media Streaming WebSocket (any `TextSocket`).

    The identity fields come from the first `start` message only; trust them only if the
    host verified the `telnyx-signature-ed25519` header on the upgrade.
    """

    network = "telnyx"
    in_rate = 8_000
    out_rate = 8_000

    def __init__(self, socket: TextSocket, *, frame_ms: int = 20, lead_ms: int = 80) -> None:
        super().__init__(frame_ms=frame_ms, lead_ms=lead_ms)
        self.socket = socket
        self.stream_id = ""
        self.call_control_id = ""
        self.call_session_id = ""
        self.client_state = ""  # base64, as you passed it to Telnyx
        self.start_info: dict[str, Any] = {}  # the whole `start` object (from, to, tags...)

    @property
    def _ready(self) -> bool:
        return bool(self.stream_id)

    def _messages(self) -> AsyncIterator[Any]:
        return aiter(self.socket)

    def _handle(self, raw: Any) -> bytes | None:
        message = parse_json(raw)
        event = message.get("event")
        if event == "start":
            self._started(message)
        elif event == "media":
            media = field(message, "media")
            if media and media.get("track", "inbound") not in ("outbound", "outbound_track"):
                return ulaw_decode(b64(media.get("payload")))
        elif event == "mark":
            self._confirmed(field(message, "mark").get("name"))
        elif event == "dtmf":
            self._digit(str(field(message, "dtmf").get("digit", "")))
        elif event == "error":
            error = field(message, "payload")
            self._log.warning("telnyx: stream error %s %s", error.get("code"), error.get("title"))
        elif event == "stop":
            return None
        return b""

    def _started(self, message: dict[str, Any]) -> None:
        if not self._first_start():
            return
        start = field(message, "start")
        self.stream_id = str(message.get("stream_id") or "")
        self.call_control_id = str(start.get("call_control_id") or "")
        self.call_session_id = str(start.get("call_session_id") or "")
        self.client_state = str(start.get("client_state") or "")
        self.start_info = start
        encoding = field(start, "media_format").get("encoding", "PCMU")
        if encoding != "PCMU":
            self._log.warning("telnyx: stream is %s; TelnyxStream reads PCMU only", encoding)

    async def _media(self, pcm: bytes) -> None:
        await self._send({"event": "media", "media": {"payload": to_b64(ulaw_encode(pcm))}})

    def _mark_message(self, name: str) -> dict[str, Any]:
        return {"event": "mark", "mark": {"name": name}}

    def _clear_message(self) -> dict[str, Any]:
        return {"event": "clear"}

    async def _write(self, data: str | bytes) -> None:
        await self.socket.send(str(data))


__all__ = ["TelnyxStream"]
