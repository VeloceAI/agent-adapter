"""`FishAudioTTS`: Fish Audio live text-to-speech over `wss://api.fish.audio/v1/tts/live`.

Every frame both ways is a MessagePack map: `start` once, `text` as the model writes it,
`flush` to speak what is buffered, `stop` at the end; audio comes back in `audio`
events until `finish`. The voice is a saved model (`reference_id`) or a clip with its
transcript (`reference=(audio, text)`). The key comes from `api_key=` or
`FISH_AUDIO_API_KEY` and travels only in the `Authorization` header.
"""

from __future__ import annotations

from typing import Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice._ws import Connect, Frame, failure
from agent_adapter.voice._ws_tts import DONE, LiveTTS, Reply


def _msgpack() -> Any:
    try:
        import msgpack
    except ImportError as exc:
        raise MissingDependency(
            'Fish Audio live TTS needs msgpack: pip install "agent-adapter[voice-cloud]"'
        ) from exc
    return msgpack


class FishAudioTTS(LiveTTS):
    """A `StreamingTTS` on Fish Audio (`model` such as `s2.1-pro`), one socket per reply.

    `latency` is `low`, `normal` or `balanced`; `speed` (0.5 to 2) and `volume` (dB,
    -20 to 20) go in `prosody`; `chunk_length` (100 to 300) is how much text Fish waits
    for before it speaks. `warm()` opens the next socket ahead.
    """

    provider = "fish-audio"
    env = "FISH_AUDIO_API_KEY"

    def __init__(
        self,
        reference_id: str | None = None,
        *,
        reference: tuple[bytes, str] | None = None,
        model: str = "s2.1-pro",
        sample_rate: int = 24_000,
        latency: str = "balanced",
        speed: float | None = None,
        volume: float | None = None,
        chunk_length: int | None = None,
        language: str | None = None,
        api_key: str | None = None,
        url: str = "wss://api.fish.audio/v1/tts/live",
        connect: Connect | None = None,
    ) -> None:
        if (reference_id is None) == (reference is None):
            raise ValueError("pass a voice: reference_id= or reference=(audio, text), not both")
        super().__init__(model, sample_rate=sample_rate, api_key=api_key, url=url, connect=connect)
        self.reference_id = reference_id
        self.reference = reference
        self.latency = latency
        self.prosody = {k: v for k, v in (("speed", speed), ("volume", volume)) if v is not None}
        self.chunk_length = chunk_length
        self.language = language
        self._pack = _msgpack()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key()}", "model": self.model}

    def _frame(self, message: dict[str, Any]) -> bytes:
        frame: bytes = self._pack.packb(message, use_bin_type=True)
        return frame

    def _first(self, context: str) -> list[Frame]:
        request: dict[str, Any] = {
            "text": "",
            "format": "pcm",
            "sample_rate": self.sample_rate,
            "latency": self.latency,
        }
        if self.reference_id is not None:
            request["reference_id"] = self.reference_id
        if self.reference is not None:
            audio, text = self.reference
            request["references"] = [{"audio": audio, "text": text}]
        if self.prosody:
            request["prosody"] = self.prosody
        if self.chunk_length is not None:
            request["chunk_length"] = self.chunk_length
        if self.language:
            request["language"] = self.language
        return [self._frame({"event": "start", "request": request})]

    def _text(self, context: str, text: str, flush: bool) -> list[Frame]:
        frames: list[Frame] = [self._frame({"event": "text", "text": text})] if text else []
        if flush:
            frames.append(self._frame({"event": "flush"}))
        return frames

    def _end(self, context: str) -> list[Frame]:
        return [self._frame({"event": "stop"})]

    def _decode(self, raw: Frame) -> tuple[str | None, Reply]:
        if not isinstance(raw, bytes):
            return None, None
        try:
            message = self._pack.unpackb(raw, raw=False)
        except Exception:  # not MessagePack: ignore it, as for unknown events
            return None, None
        if not isinstance(message, dict):
            return None, None
        event = message.get("event")
        if event == "audio":
            audio = message.get("audio")
            return None, bytes(audio) if isinstance(audio, bytes | bytearray) else None
        if event == "finish":
            if message.get("reason") == "error":
                return None, failure(self.provider, self.model, "synthesis failed")
            return None, DONE
        return None, None


__all__ = ["FishAudioTTS"]
