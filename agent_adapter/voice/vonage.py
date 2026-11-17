"""`VonageStream`: a phone call through a Vonage Voice API WebSocket (NCCO `connect`).

Vonage sends the caller's audio as binary messages of 16-bit little-endian PCM at the rate
in the NCCO `content-type` (`audio/l16;rate=16000`), and plays back binary audio we send the
same way. Control is JSON text: `{"action": "clear"}` on a cut, and after each line a
`{"action": "notify"}`, which Vonage answers with `websocket:notify` once its audio has
played. Keypad presses arrive as `websocket:dtmf` and reach `on_digit`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.voice._phone_stream import PhoneStream, field, parse_json
from agent_adapter.voice.audio_socket import BinarySocket

_RATES = (8_000, 16_000, 24_000)


class VonageStream(PhoneStream):
    """An `AudioTransport` over a Vonage Voice API WebSocket (any `BinarySocket`).

    `rate` must match the NCCO endpoint's `content-type`. `headers` holds the first
    `websocket:connected` message's fields (your NCCO `headers`); trust them only if the
    host checked the `Authorization` header on the upgrade.
    """

    network = "vonage"

    def __init__(
        self, socket: BinarySocket, *, rate: int = 16_000, frame_ms: int = 20, lead_ms: int = 80
    ) -> None:
        if rate not in _RATES:
            raise ConfigError(f"VonageStream rate must be one of {_RATES}")
        super().__init__(frame_ms=frame_ms, lead_ms=lead_ms)
        self.socket = socket
        self.in_rate = self.out_rate = rate
        self.content_type = ""
        self.headers: dict[str, Any] = {}
        self._frame = rate * 2 * frame_ms // 1000

    @property
    def _ready(self) -> bool:
        return True  # Vonage needs no stream id to address what we send

    def _messages(self) -> AsyncIterator[Any]:
        return aiter(self.socket)

    def _handle(self, raw: Any) -> bytes | None:
        if isinstance(raw, bytes | bytearray):
            return bytes(raw[: len(raw) & ~1])
        message = parse_json(raw)
        event = message.get("event")
        if event == "websocket:connected":
            self._connected(message)
        elif event == "websocket:notify":
            self._confirmed(field(message, "payload").get("mark"))
        elif event == "websocket:dtmf":
            self._digit(str(message.get("digit", "")))
        return b""

    def _connected(self, message: dict[str, Any]) -> None:
        if not self._first_start():
            return
        self.content_type = str(message.get("content-type") or "")
        self.headers = {k: v for k, v in message.items() if k not in ("event", "content-type")}
        if self.content_type and f"rate={self.in_rate}" not in self.content_type:
            self._log.warning(
                "vonage: stream is %s; VonageStream expects rate=%s",
                self.content_type,
                self.in_rate,
            )

    async def _media(self, pcm: bytes) -> None:
        # Vonage documents ~20 ms packets; pad a short last frame with silence to keep that size.
        await self._deliver(pcm + bytes(max(0, self._frame - len(pcm))), "audio")

    def _mark_message(self, name: str) -> dict[str, Any]:
        return {"action": "notify", "payload": {"mark": name}}

    def _clear_message(self) -> dict[str, Any]:
        return {"action": "clear"}

    async def _write(self, data: str | bytes) -> None:
        await self.socket.send(data)


__all__ = ["VonageStream"]
