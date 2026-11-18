"""`AssemblyAISTT`: AssemblyAI Universal-Streaming (v3) speech-to-text, one socket per turn.

The model decides the end of the turn itself and says so with `end_of_turn`; the stream
passes that on as an `end` `TurnSignal`. The key comes from `api_key=` or
`ASSEMBLYAI_API_KEY` and travels only in the `Authorization` header.
"""

from __future__ import annotations

from urllib.parse import urlencode

from agent_adapter.voice._ws import Connect, Frame, LiveSTT, LiveSTTStream, dumps, parse


class AssemblyAISTT(LiveSTT):
    """A `StreamingSTT` on AssemblyAI (`speech_model`, such as `universal-3-6-pro` or
    `assemblyai-stt-3-7-realtime-agent`).

    `mode` is the API's preset (`min_latency`, `balanced`, `max_accuracy`);
    `min_turn_silence` and `max_turn_silence` (ms) tune the end of the turn. AssemblyAI
    takes 20 to 1000 ms chunks no faster than real time, so the stream regroups audio
    into 50 to 1000 ms chunks and `transcribe()` sends at the pace it plays.
    """

    provider = "assemblyai"
    env = "ASSEMBLYAI_API_KEY"
    realtime_only = True

    def __init__(
        self,
        model: str = "universal-3-6-pro",
        *,
        mode: str | None = None,
        min_turn_silence: int | None = None,
        max_turn_silence: int | None = None,
        keyterms: tuple[str, ...] = (),
        api_key: str | None = None,
        url: str = "wss://streaming.assemblyai.com/v3/ws",
        finish_timeout_s: float = 5.0,
        connect: Connect | None = None,
    ) -> None:
        super().__init__(
            model, api_key=api_key, url=url, connect=connect, finish_timeout_s=finish_timeout_s
        )
        self.mode = mode
        self.min_turn_silence = min_turn_silence
        self.max_turn_silence = max_turn_silence
        self.keyterms = keyterms

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self._key()}  # the raw key: AssemblyAI takes no "Bearer"

    def open(self, sample_rate: int) -> _AssemblyStream:
        params: dict[str, str | int] = {
            "sample_rate": sample_rate,
            "encoding": "pcm_s16le",
            "speech_model": self.model,
        }
        for name in ("mode", "min_turn_silence", "max_turn_silence"):
            if (value := getattr(self, name)) is not None:
                params[name] = value
        if self.keyterms:
            params["keyterms_prompt"] = dumps({"k": list(self.keyterms)})[5:-1]
        return _AssemblyStream(self, f"{self.url}?{urlencode(params)}", sample_rate)


class _AssemblyStream(LiveSTTStream):
    def __init__(self, stt: LiveSTT, url: str, sample_rate: int) -> None:
        self._turns: dict[int, str] = {}
        self._ended = self._finishing = False
        self._pending = b""
        self._least = sample_rate // 20 * 2  # 50 ms
        self._most = sample_rate * 2  # 1000 ms
        super().__init__(stt, url)

    def feed(self, pcm: bytes) -> None:
        self._pending += pcm
        while len(self._pending) >= self._least:
            piece, self._pending = self._pending[: self._most], self._pending[self._most :]
            super().feed(piece)

    async def finish(self) -> str:
        if self._pending:  # pad the tail to the 20 ms the API takes at least
            tail = self._pending.ljust(max(len(self._pending), self._least * 2 // 5), b"\0")
            self._pending = b""
            super().feed(tail)
        return await super().finish()

    def _closing(self) -> list[Frame]:
        self._finishing = True  # the turn ForceEndpoint closes is the last we need
        return [dumps({"type": "ForceEndpoint"}), dumps({"type": "Terminate"})]

    def _settled(self) -> bool:
        return self._ended

    def _handle(self, raw: Frame) -> bool:
        message = parse(raw) if isinstance(raw, str) else {}
        kind = message.get("type")
        if kind == "Termination":
            return True
        if kind == "Error":
            raise self._fail(message.get("error"), message.get("error_code"))
        if kind != "Turn":
            return False
        order = message.get("turn_order")
        self._turns[order if isinstance(order, int) else 0] = str(
            message.get("transcript") or ""
        ).strip()
        self._ended = bool(message.get("end_of_turn"))
        if self._ended:
            confidence = message.get("end_of_turn_confidence")
            confidence = float(confidence) if isinstance(confidence, int | float) else None
            self._emit("end", self._result(), confidence)
        return self._ended and self._finishing

    def _result(self) -> str:
        return " ".join(text for _, text in sorted(self._turns.items()) if text)


__all__ = ["AssemblyAISTT"]
