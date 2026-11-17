"""`PlivoStream`: a phone call through Plivo Audio Streaming (`<Stream bidirectional="true">`).

Plivo sends the caller's audio as base64 in JSON `media` messages over a WebSocket; we send
ours back as `playAudio`, `clearAudio` on a cut, and a `checkpoint` after each line, which
Plivo answers with `playedStream` once the line has played. Keypad presses arrive as `dtmf`
messages and reach `on_digit`. Pass the same `content_type` as the `<Stream>` element.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.voice._phone_stream import PhoneStream, b64, field, parse_json, to_b64
from agent_adapter.voice.pcm import ulaw_decode, ulaw_encode
from agent_adapter.voice.socket import TextSocket

_FORMATS = {
    "audio/x-l16;rate=8000": ("audio/x-l16", 8_000),
    "audio/x-l16;rate=16000": ("audio/x-l16", 16_000),
    "audio/x-mulaw;rate=8000": ("audio/x-mulaw", 8_000),
}


class PlivoStream(PhoneStream):
    """An `AudioTransport` over a Plivo Audio Streaming WebSocket (any `TextSocket`).

    `content_type` is one of Plivo's: `audio/x-l16;rate=8000` (Plivo's default),
    `audio/x-l16;rate=16000` or `audio/x-mulaw;rate=8000`. The identity fields come from
    the first `start` message only; trust them only if the host verified
    `X-Plivo-Signature-V3` on the upgrade.
    """

    network = "plivo"

    def __init__(
        self,
        socket: TextSocket,
        *,
        content_type: str = "audio/x-l16;rate=8000",
        frame_ms: int = 20,
        lead_ms: int = 80,
    ) -> None:
        if content_type.replace(" ", "") not in _FORMATS:
            raise ConfigError(f"PlivoStream content_type must be one of {sorted(_FORMATS)}")
        super().__init__(frame_ms=frame_ms, lead_ms=lead_ms)
        self.socket = socket
        self._encoding, rate = _FORMATS[content_type.replace(" ", "")]
        self.in_rate = self.out_rate = rate
        self.stream_id = ""
        self.call_id = ""
        self.account_id = ""
        self.parameters: dict[str, str] = {}  # from the <Stream extraHeaders="k=v;..."> attribute

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
            if media and media.get("track", "inbound") == "inbound":
                return self._decode(b64(media.get("payload")))
        elif event == "playedStream":
            self._confirmed(message.get("name"))
        elif event == "dtmf":
            self._digit(str(field(message, "dtmf").get("digit", "")))
        elif event == "stop":
            return None
        return b""

    def _started(self, message: dict[str, Any]) -> None:
        if not self._first_start():
            return
        start = field(message, "start")
        self.stream_id = str(start.get("streamId") or message.get("streamId") or "")
        self.call_id = str(start.get("callId") or "")
        self.account_id = str(start.get("accountId") or "")
        self.parameters = _headers(message.get("extra_headers"))
        encoding = field(start, "mediaFormat").get("encoding")
        if encoding and encoding != self._encoding:
            self._log.warning(
                "plivo: stream is %s but content_type says %s", encoding, self._encoding
            )

    def _decode(self, data: bytes) -> bytes:
        if self._encoding == "audio/x-mulaw":
            return ulaw_decode(data)
        return data[: len(data) & ~1]

    async def _media(self, pcm: bytes) -> None:
        data = ulaw_encode(pcm) if self._encoding == "audio/x-mulaw" else pcm
        media = {
            "contentType": self._encoding,
            "sampleRate": self.out_rate,
            "payload": to_b64(data),
        }
        await self._send({"event": "playAudio", "media": media})

    def _mark_message(self, name: str) -> dict[str, Any]:
        return {"event": "checkpoint", "streamId": self.stream_id, "name": name}

    def _clear_message(self) -> dict[str, Any]:
        return {"event": "clearAudio", "streamId": self.stream_id}

    async def _write(self, data: str | bytes) -> None:
        await self.socket.send(str(data))


def _headers(raw: Any) -> dict[str, str]:
    """`"k=v;k2=v2"` to a dict; parts without `=` are skipped."""
    pairs = (part.split("=", 1) for part in str(raw or "").split(";") if "=" in part)
    return {key.strip(): value.strip() for key, value in pairs if key.strip()}


__all__ = ["PlivoStream"]
