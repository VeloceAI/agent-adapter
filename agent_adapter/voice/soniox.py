"""`SonioxSTT`: Soniox real-time speech-to-text, one socket per turn.

With endpoint detection on, Soniox sends an `<end>` token when the speaker is done; the
stream passes it on as an `end` `TurnSignal`. At the end of the turn we send `finalize`
and stop at its `<fin>` token. The key comes from `api_key=` or `SONIOX_API_KEY` and
travels only in the `Authorization` header (never in the config message).
"""

from __future__ import annotations

from typing import Any

from agent_adapter.voice._ws import Connect, Frame, LiveSTT, LiveSTTStream, dumps, parse


class SonioxSTT(LiveSTT):
    """A `StreamingSTT` on Soniox (`stt-rt-v6`).

    `language_hints` (such as `("en",)`) and `context` (Soniox's context object: terms,
    text) go in the config message; `max_endpoint_delay_ms` (500 to 3000) bounds how long
    Soniox waits before it calls the end of speech.
    """

    provider = "soniox"
    env = "SONIOX_API_KEY"

    def __init__(
        self,
        model: str = "stt-rt-v6",
        *,
        language_hints: tuple[str, ...] = (),
        endpoint_detection: bool = True,
        max_endpoint_delay_ms: int | None = None,
        context: dict[str, Any] | None = None,
        api_key: str | None = None,
        url: str = "wss://stt-rt.soniox.com/transcribe-websocket",
        finish_timeout_s: float = 5.0,
        connect: Connect | None = None,
    ) -> None:
        super().__init__(
            model, api_key=api_key, url=url, connect=connect, finish_timeout_s=finish_timeout_s
        )
        self.language_hints = language_hints
        self.endpoint_detection = endpoint_detection
        self.max_endpoint_delay_ms = max_endpoint_delay_ms
        self.context = context

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key()}"}

    def open(self, sample_rate: int) -> _SonioxStream:
        config: dict[str, Any] = {
            "model": self.model,
            "audio_format": "pcm_s16le",
            "sample_rate": sample_rate,
            "num_channels": 1,
            "enable_endpoint_detection": self.endpoint_detection,
        }
        if self.language_hints:
            config["language_hints"] = list(self.language_hints)
        if self.max_endpoint_delay_ms is not None:
            config["max_endpoint_delay_ms"] = self.max_endpoint_delay_ms
        if self.context:
            config["context"] = self.context
        return _SonioxStream(self, self.url, dumps(config))


class _SonioxStream(LiveSTTStream):
    def __init__(self, stt: LiveSTT, url: str, config: str) -> None:
        self._config = config
        self._final = ""
        self._partial = ""
        super().__init__(stt, url)

    def _opening(self) -> list[Frame]:
        return [self._config]

    def _closing(self) -> list[Frame]:
        return [dumps({"type": "finalize"}), ""]  # an empty text frame ends the audio

    def _handle(self, raw: Frame) -> bool:
        message = parse(raw) if isinstance(raw, str) else {}
        if "error_code" in message:
            raise self._fail(message.get("error_message"), message.get("error_code"))
        partial, done = "", False
        for token in message.get("tokens") or ():
            if not isinstance(token, dict):
                continue
            text = str(token.get("text", ""))
            if text == "<end>":  # every word before an endpoint is final
                self._emit("end", self._final.strip())
            elif text == "<fin>":
                done = True
            elif token.get("is_final"):
                self._final += text
            else:
                partial += text
        self._partial = partial
        return done or bool(message.get("finished"))

    def _result(self) -> str:
        return (self._final + self._partial).strip()

    @property
    def partial(self) -> str:
        """The words so far, final and not: an early reply may start on them."""
        return self._result()


__all__ = ["SonioxSTT"]
