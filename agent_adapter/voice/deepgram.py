"""Deepgram speech-to-text that hears each turn as it is spoken: `DeepgramSTT` (Nova) and
`DeepgramFluxSTT` (Flux, which also says when the turn ends, early).

`DeepgramSTT.open()` streams the turn over the live WebSocket; at its end we send
`Finalize` and `CloseStream`, so only the last moment is left to transcribe.
`transcribe()` sends a finished turn in one HTTP request. The key comes from `api_key=`
or `DEEPGRAM_API_KEY` and travels only in the `Authorization` header.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from agent_adapter.voice._http import HttpClient, post, wav
from agent_adapter.voice._ws import (
    Connect,
    Frame,
    LiveSocket,
    LiveSTT,
    LiveSTTStream,
    dumps,
    parse,
)


class _Deepgram(LiveSTT):
    provider = "deepgram"
    env = "DEEPGRAM_API_KEY"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._key()}"}


class DeepgramSTT(_Deepgram):
    """A `StreamingSTT` on Deepgram (`model` such as `nova-3`).

    `finish_timeout_s` bounds the wait for the final words after the turn ends; past it
    the session falls back to `transcribe()` on the whole turn.
    """

    def __init__(
        self,
        model: str = "nova-3",
        *,
        language: str = "en",
        smart_format: bool = True,
        api_key: str | None = None,
        host: str = "api.deepgram.com",
        finish_timeout_s: float = 5.0,
        client: Any = None,
        connect: Connect | None = None,
        timeout: float = 30,
    ) -> None:
        url = f"wss://{host}/v1/listen"
        super().__init__(
            model, api_key=api_key, url=url, connect=connect, finish_timeout_s=finish_timeout_s
        )
        self.language = language
        self.smart_format = smart_format
        self.host = host
        self._http = HttpClient(client, timeout)

    def _query(self, **extra: str | int) -> str:
        params: dict[str, str | int] = {"model": self.model, "language": self.language}
        if self.smart_format:
            params["smart_format"] = "true"
        return urlencode({**params, **extra})

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        response = await post(
            self._http,
            self.provider,
            self.model,
            f"https://{self.host}/v1/listen?{self._query()}",
            headers={**self._headers(), "Content-Type": "audio/wav"},
            content=wav(pcm, sample_rate),
        )
        channels = (response.json().get("results") or {}).get("channels") or [{}]
        return _transcript(channels[0])

    def open(self, sample_rate: int) -> _NovaStream:
        query = self._query(
            encoding="linear16", sample_rate=sample_rate, channels=1, interim_results="true"
        )
        return _NovaStream(self, f"{self.url}?{query}")

    async def aclose(self) -> None:
        """Close the HTTP client this object made (one you passed in stays open)."""
        await self._http.aclose()


class _NovaStream(LiveSTTStream):
    def __init__(self, stt: LiveSTT, url: str) -> None:
        self._finals: list[str] = []
        self._interim = ""
        super().__init__(stt, url)

    @property
    def partial(self) -> str:
        """The words so far, final and interim: an early reply may start on them."""
        return " ".join([*self._finals, self._interim]).strip()

    def _closing(self) -> list[Frame]:
        return [dumps({"type": "Finalize"}), dumps({"type": "CloseStream"})]

    def _handle(self, raw: Frame) -> bool:
        message = parse(raw) if isinstance(raw, str) else {}
        kind = message.get("type")
        if kind == "Results":
            text = _transcript(message.get("channel"))
            if message.get("is_final"):
                self._finals += [text] if text else []
                self._interim = ""
            else:
                self._interim = text
            return bool(message.get("from_finalize"))
        if kind == "Error":
            raise self._fail(message.get("description") or message.get("message"))
        return kind == "Metadata"

    def _result(self) -> str:
        return " ".join(self._finals)


class DeepgramFluxSTT(_Deepgram):
    """A `StreamingSTT` on Deepgram Flux (`flux-general-en`) whose streams send `TurnSignal`s.

    Flux decides the end of the turn itself. `eot_threshold` is how sure it must be to
    send `EndOfTurn`; with `eager_eot_threshold` set (lower, such as 0.5) it first sends
    `EagerEndOfTurn` (our `eager_end`), so a reply can start early, and `TurnResumed`
    (`resumed`) if the caller goes on. Once `EndOfTurn` is in, `finish()` returns at once;
    before that it sends `ForceEndTurn`, so Flux closes the turn without waiting for silence.
    """

    def __init__(
        self,
        model: str = "flux-general-en",
        *,
        eot_threshold: float | None = 0.7,
        eager_eot_threshold: float | None = 0.5,
        eot_timeout_ms: int | None = None,
        keyterms: tuple[str, ...] = (),
        api_key: str | None = None,
        url: str = "wss://api.deepgram.com/v2/listen",
        finish_timeout_s: float = 5.0,
        connect: Connect | None = None,
    ) -> None:
        super().__init__(
            model, api_key=api_key, url=url, connect=connect, finish_timeout_s=finish_timeout_s
        )
        self.eot_threshold = eot_threshold
        self.eager_eot_threshold = eager_eot_threshold
        self.eot_timeout_ms = eot_timeout_ms
        self.keyterms = keyterms

    def open(self, sample_rate: int) -> _FluxStream:
        params: list[tuple[str, str | int | float]] = [
            ("model", self.model),
            ("encoding", "linear16"),
            ("sample_rate", sample_rate),
        ]
        for name in ("eot_threshold", "eager_eot_threshold", "eot_timeout_ms"):
            if (value := getattr(self, name)) is not None:
                params.append((name, value))
        params += [("keyterm", term) for term in self.keyterms]
        return _FluxStream(self, f"{self.url}?{urlencode(params)}")


class _FluxStream(LiveSTTStream):
    def __init__(self, stt: LiveSTT, url: str) -> None:
        self._turns: list[str] = []
        self._current = ""
        self._ended = self._finishing = False
        super().__init__(stt, url)

    def _closing(self) -> list[Frame]:
        self._finishing = True  # ForceEndTurn makes Flux send EndOfTurn now
        return [dumps({"type": "ForceEndTurn"}), dumps({"type": "CloseStream"})]

    def _settled(self) -> bool:
        return self._ended

    def _handle(self, raw: Frame) -> bool:
        message = parse(raw) if isinstance(raw, str) else {}
        kind = message.get("type")
        if kind in ("Error", "FatalError"):
            raise self._fail(message.get("description") or message.get("code"))
        if kind != "TurnInfo":
            return False
        event = message.get("event")
        text = str(message.get("transcript") or "").strip()
        confidence = message.get("end_of_turn_confidence")
        confidence = float(confidence) if isinstance(confidence, int | float) else None
        if event == "EndOfTurn":
            if text:
                self._turns.append(text)
            self._current, self._ended = "", True
            self._emit("end", self._result(), confidence)
            return self._finishing
        self._ended = False
        if text or event == "StartOfTurn":
            self._current = text
        if event == "EagerEndOfTurn":
            self._emit("eager_end", self._result(), confidence)
        elif event == "TurnResumed":
            self._emit("resumed", self._result(), confidence)
        return False

    def _result(self) -> str:
        return " ".join(t for t in (*self._turns, self._current) if t)


def _transcript(channel: Any) -> str:
    alternatives = channel.get("alternatives") if isinstance(channel, dict) else None
    if not isinstance(alternatives, list) or not alternatives:
        return ""
    first = alternatives[0]
    return str(first.get("transcript", "")).strip() if isinstance(first, dict) else ""


__all__ = ["Connect", "DeepgramFluxSTT", "DeepgramSTT", "LiveSocket"]
