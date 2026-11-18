"""Cartesia over WebSocket: `CartesiaLiveTTS` (Sonic) and `CartesiaInkSTT` (Ink).

`CartesiaLiveTTS` keeps one socket open and speaks each reply on its own `context_id`:
text is sent as the model writes it with `continue: true`, and a last empty request
with `continue: false` closes the context. Word timestamps come back with the audio.
The key comes from `api_key=` or `CARTESIA_API_KEY` and travels only in the `X-API-Key`
header. Cartesia over plain HTTP is `agent_adapter.voice.cloud.CartesiaTTS`.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from agent_adapter.voice._ws import Connect, Frame, LiveSTT, LiveSTTStream, dumps, failure, parse
from agent_adapter.voice._ws_tts import DONE, LiveTTS, Reply, b64_audio, word_times

VERSION = "2026-08-14"
_RATES = frozenset({8_000, 16_000, 22_050, 24_000, 44_100, 48_000})


class CartesiaLiveTTS(LiveTTS):
    """A `StreamingTTS` on Cartesia Sonic (`sonic-3`) over one shared WebSocket.

    `speed` (0.6 to 1.5), `volume` (0.5 to 2.0) and `emotion` (`neutral`, `calm`,
    `angry`, `content`, `sad`) go in `generation_config`. `max_buffer_delay_ms` is how
    long Cartesia may wait for more text before it speaks (lower is sooner).
    """

    provider = "cartesia"
    env = "CARTESIA_API_KEY"
    shared = True

    def __init__(
        self,
        voice_id: str,
        *,
        model: str = "sonic-3",
        sample_rate: int = 24_000,
        language: str | None = "en",
        speed: float | None = None,
        volume: float | None = None,
        emotion: str | None = None,
        max_buffer_delay_ms: int | None = None,
        timestamps: bool = True,
        version: str = VERSION,
        api_key: str | None = None,
        url: str = "wss://api.cartesia.ai/tts/websocket",
        connect: Connect | None = None,
    ) -> None:
        if sample_rate not in _RATES:
            raise ValueError(f"sample_rate must be one of {sorted(_RATES)}")
        super().__init__(model, sample_rate=sample_rate, api_key=api_key, url=url, connect=connect)
        self.voice_id = voice_id
        self.language = language
        self.generation_config = {
            name: value
            for name, value in (("speed", speed), ("volume", volume), ("emotion", emotion))
            if value is not None
        }
        self.max_buffer_delay_ms = max_buffer_delay_ms
        self.timestamps = timestamps
        self.version = version

    def _url(self) -> str:
        return f"{self.url}?{urlencode({'cartesia_version': self.version})}"

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._key(), "Cartesia-Version": self.version}

    def _request(self, context: str, text: str, more: bool, flush: bool = False) -> str:
        body: dict[str, Any] = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"id": self.voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "context_id": context,
            "continue": more,
            "add_timestamps": self.timestamps,
        }
        if self.language:
            body["language"] = self.language
        if self.generation_config:
            body["generation_config"] = self.generation_config
        if self.max_buffer_delay_ms is not None:
            body["max_buffer_delay_ms"] = self.max_buffer_delay_ms
        if flush:
            body["flush"] = True
        return dumps(body)

    def _text(self, context: str, text: str, flush: bool) -> list[Frame]:
        return [self._request(context, text, True, flush)]

    def _end(self, context: str) -> list[Frame]:
        return [self._request(context, "", False)]

    def _cancel(self, context: str) -> list[Frame]:
        return [dumps({"context_id": context, "cancel": True})]

    def _decode(self, raw: Frame) -> tuple[str | None, Reply]:
        message = parse(raw)
        context = message.get("context_id")
        context = context if isinstance(context, str) else None
        kind = message.get("type")
        if kind == "chunk":
            return context, b64_audio(message.get("data"))
        if kind == "timestamps":
            return context, word_times(message.get("word_timestamps"))
        if kind == "done":
            return context, DONE
        if kind == "error":
            detail = message.get("message") or message.get("title") or message.get("error")
            return context, failure(self.provider, self.model, detail, message.get("status_code"))
        return context, None


class CartesiaInkSTT(LiveSTT):
    """A `StreamingSTT` on Cartesia Ink (`ink-2`, or `ink-whisper` with `language`).

    At the end of the turn we send `finalize` and stop at its `flush_done`.
    """

    provider = "cartesia"
    env = "CARTESIA_API_KEY"

    def __init__(
        self,
        model: str = "ink-2",
        *,
        language: str | None = None,
        keyterms: tuple[str, ...] = (),
        version: str = VERSION,
        api_key: str | None = None,
        url: str = "wss://api.cartesia.ai/stt/websocket",
        finish_timeout_s: float = 5.0,
        connect: Connect | None = None,
    ) -> None:
        super().__init__(
            model, api_key=api_key, url=url, connect=connect, finish_timeout_s=finish_timeout_s
        )
        self.language = language
        self.keyterms = keyterms
        self.version = version

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._key(), "Cartesia-Version": self.version}

    def open(self, sample_rate: int) -> _InkStream:
        params: list[tuple[str, str | int]] = [
            ("model", self.model),
            ("encoding", "pcm_s16le"),
            ("sample_rate", sample_rate),
            ("cartesia_version", self.version),
        ]
        if self.language:
            params.append(("language", self.language))
        params += [("keyterm", term) for term in self.keyterms]
        return _InkStream(self, f"{self.url}?{urlencode(params)}")


class _InkStream(LiveSTTStream):
    def __init__(self, stt: LiveSTT, url: str) -> None:
        self._parts: list[str] = []
        super().__init__(stt, url)

    def _closing(self) -> list[Frame]:
        return ["finalize", "close"]

    def _handle(self, raw: Frame) -> bool:
        message = parse(raw) if isinstance(raw, str) else {}
        kind = message.get("type")
        if kind == "transcript" and message.get("is_final"):
            self._parts.append(str(message.get("text") or ""))  # deltas: join as sent
        elif kind == "error":
            raise self._fail(
                message.get("message") or message.get("title"), message.get("status_code")
            )
        return kind in ("flush_done", "done")

    def _result(self) -> str:
        return "".join(self._parts).strip()


__all__ = ["CartesiaInkSTT", "CartesiaLiveTTS"]
