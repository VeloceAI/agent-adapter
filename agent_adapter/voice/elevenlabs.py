"""ElevenLabs over WebSocket: `ElevenLabsLiveTTS` (`stream-input`) and
`ElevenLabsScribeSTT` (Scribe realtime).

`ElevenLabsLiveTTS` opens one socket per reply and sends text as the model writes it;
`chunk_length_schedule` sets how many characters ElevenLabs waits for before each piece
of audio, and `flush` speaks what is buffered now. `warm()` opens the next socket ahead.
The key comes from `api_key=` or `ELEVENLABS_API_KEY` and travels only in the
`xi-api-key` header. ElevenLabs over plain HTTP is `agent_adapter.voice.cloud.ElevenLabsTTS`.
"""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import quote, urlencode

from agent_adapter.voice._ws import Connect, Frame, LiveSTT, LiveSTTStream, dumps, failure, parse
from agent_adapter.voice._ws_tts import DONE, LiveTTS, Reply, b64_audio, words_from_chars

_RATES = frozenset({8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000})
_SCRIBE_RATES = frozenset({8_000, 16_000, 22_050, 24_000, 44_100, 48_000})
_SCRIBE_ERRORS = frozenset(
    {
        "error",
        "auth_error",
        "quota_exceeded",
        "commit_throttled",
        "unaccepted_terms",
        "rate_limited",
        "queue_overflow",
        "resource_exhausted",
        "session_time_limit_exceeded",
        "input_error",
        "invalid_request",
        "chunk_size_exceeded",
        "transcriber_error",
    }
)
_STATUS = {"auth_error": 401, "quota_exceeded": 429, "rate_limited": 429}


class ElevenLabsLiveTTS(LiveTTS):
    """A `StreamingTTS` on ElevenLabs' `stream-input` WebSocket (`eleven_flash_v2_5`).

    `chunk_length_schedule` (characters, 50 to 500 each) trades a sooner first sound for
    smoother speech; `voice_settings` is ElevenLabs' object (`stability`, `speed`, ...).
    """

    provider = "elevenlabs"
    env = "ELEVENLABS_API_KEY"
    chunk_relative_words = True

    def __init__(
        self,
        voice_id: str,
        *,
        model: str = "eleven_flash_v2_5",
        sample_rate: int = 24_000,
        language: str | None = None,
        chunk_length_schedule: tuple[int, ...] | None = (50, 120, 160, 290),
        voice_settings: dict[str, Any] | None = None,
        auto_mode: bool = False,
        inactivity_timeout: int = 60,
        api_key: str | None = None,
        url: str = "wss://api.elevenlabs.io/v1/text-to-speech",
        connect: Connect | None = None,
    ) -> None:
        if sample_rate not in _RATES:
            raise ValueError(f"sample_rate must be one of {sorted(_RATES)}")
        super().__init__(model, sample_rate=sample_rate, api_key=api_key, url=url, connect=connect)
        self.voice_id = voice_id
        self.language = language
        self.chunk_length_schedule = chunk_length_schedule
        self.voice_settings = voice_settings
        self.auto_mode = auto_mode
        self.inactivity_timeout = inactivity_timeout

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._key()}

    def _url(self) -> str:
        params: dict[str, str | int] = {
            "model_id": self.model,
            "output_format": f"pcm_{self.sample_rate}",
            "inactivity_timeout": self.inactivity_timeout,
            "sync_alignment": "true",
        }
        if self.language:
            params["language_code"] = self.language
        if self.auto_mode:
            params["auto_mode"] = "true"
        voice = quote(self.voice_id, safe="")
        return f"{self.url}/{voice}/stream-input?{urlencode(params)}"

    def _first(self, context: str) -> list[Frame]:
        first: dict[str, Any] = {"text": " "}
        if self.voice_settings:
            first["voice_settings"] = self.voice_settings
        if self.chunk_length_schedule:
            first["generation_config"] = {"chunk_length_schedule": list(self.chunk_length_schedule)}
        return [dumps(first)]

    def _text(self, context: str, text: str, flush: bool) -> list[Frame]:
        if text and not text[-1].isspace():
            text += " "  # ElevenLabs wants each piece to end with a space
        message: dict[str, Any] = {"text": text}
        if flush:
            message["flush"] = True
        return [dumps(message)]

    def _end(self, context: str) -> list[Frame]:
        return [dumps({"text": ""})]

    def _decode(self, raw: Frame) -> tuple[str | None, Reply]:
        message = parse(raw)
        if message.get("isFinal"):
            return None, DONE
        if "audio" in message or "alignment" in message:
            replies: list[Any] = []
            alignment = message.get("alignment")
            if isinstance(alignment, dict):
                chars = alignment.get("chars") or []
                starts = [float(ms) / 1000 for ms in alignment.get("charStartTimesMs") or []]
                lengths = [float(ms) / 1000 for ms in alignment.get("charDurationsMs") or []]
                ends = [s + d for s, d in zip(starts, lengths, strict=False)]
                replies.append(words_from_chars(chars, starts, ends))
            if (audio := b64_audio(message.get("audio"))) is not None:
                replies.append(audio)
            return None, tuple(replies) or None
        if message.get("error") or message.get("message"):
            detail = message.get("message") or message.get("error")
            return None, failure(self.provider, self.model, detail, message.get("code"))
        return None, None


class ElevenLabsScribeSTT(LiveSTT):
    """A `StreamingSTT` on ElevenLabs Scribe realtime (`scribe_v2_realtime`).

    With `commit_strategy="vad"` (or `"turn_prediction"` on `scribe_v3_live`) Scribe
    commits a segment when the speaker pauses, and the stream passes that on as an `end`
    `TurnSignal`; with `"manual"` only the end of our turn commits.
    """

    provider = "elevenlabs"
    env = "ELEVENLABS_API_KEY"

    def __init__(
        self,
        model: str = "scribe_v2_realtime",
        *,
        language: str | None = None,
        commit_strategy: str = "manual",
        vad_silence_threshold_secs: float | None = None,
        keyterms: tuple[str, ...] = (),
        api_key: str | None = None,
        url: str = "wss://api.elevenlabs.io/v1/speech-to-text/realtime",
        finish_timeout_s: float = 5.0,
        connect: Connect | None = None,
    ) -> None:
        super().__init__(
            model, api_key=api_key, url=url, connect=connect, finish_timeout_s=finish_timeout_s
        )
        self.language = language
        self.commit_strategy = commit_strategy
        self.vad_silence_threshold_secs = vad_silence_threshold_secs
        self.keyterms = keyterms

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._key()}

    def open(self, sample_rate: int) -> _ScribeStream:
        if sample_rate not in _SCRIBE_RATES:
            raise ValueError(f"sample_rate must be one of {sorted(_SCRIBE_RATES)}")
        params: list[tuple[str, str | int | float]] = [
            ("model_id", self.model),
            ("audio_format", f"pcm_{sample_rate}"),
            ("commit_strategy", self.commit_strategy),
        ]
        if self.language:
            params.append(("language_code", self.language))
        if self.vad_silence_threshold_secs is not None:
            params.append(("vad_silence_threshold_secs", self.vad_silence_threshold_secs))
        params += [("keyterms", term) for term in self.keyterms]
        return _ScribeStream(self, f"{self.url}?{urlencode(params)}", sample_rate)


class _ScribeStream(LiveSTTStream):
    def __init__(self, stt: LiveSTT, url: str, sample_rate: int) -> None:
        self._rate = sample_rate
        self._committed: list[str] = []
        self._fresh = False  # audio sent since the last commit
        self._finishing = False
        super().__init__(stt, url)

    def _chunk(self, pcm: bytes, commit: bool) -> str:
        audio = base64.b64encode(pcm).decode("ascii")
        message = {"message_type": "input_audio_chunk", "audio_base_64": audio}
        return dumps({**message, "commit": commit, "sample_rate": self._rate})

    def _frame(self, pcm: bytes) -> Frame:
        self._fresh = True
        return self._chunk(pcm, False)

    def _closing(self) -> list[Frame]:
        self._finishing = True
        return [self._chunk(b"", True)] if self._fresh else []

    def _settled(self) -> bool:
        return not self._fresh

    def _handle(self, raw: Frame) -> bool:
        message = parse(raw) if isinstance(raw, str) else {}
        kind = message.get("message_type")
        if kind == "committed_transcript":
            if text := str(message.get("text") or "").strip():
                self._committed.append(text)
            self._fresh = False
            self._emit("end", self._result())
            return self._finishing
        if kind == "insufficient_audio_activity":
            return self._finishing  # nothing left to commit
        if kind in _SCRIBE_ERRORS:
            raise self._fail(message.get("error") or kind, _STATUS.get(str(kind)))
        return False

    def _result(self) -> str:
        return " ".join(self._committed)


__all__ = ["ElevenLabsLiveTTS", "ElevenLabsScribeSTT"]
