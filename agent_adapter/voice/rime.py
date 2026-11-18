"""`RimeTTS`: Rime text-to-speech over its WebSocket v1 protocol (Coda or Mist v3).

One socket stays open; each reply is a `contextId` that `start`s, takes `text` as the
model writes it and `end`s, with audio and word spans coming back on the same context.
Rime greets each socket with `ready` before it takes requests. The key comes from
`api_key=` or `RIME_API_KEY` and travels only in the `Authorization` header.
"""

from __future__ import annotations

from typing import Any

from agent_adapter.voice._ws import Connect, Frame, LiveSocket, dumps, failure, parse
from agent_adapter.voice._ws_tts import DONE, LiveTTS, Reply, b64_audio
from agent_adapter.voice.ports import WordTime

_HOSTS = {"coda": "wss://coda.api.rime.ai/ws", "mistv3": "wss://mist.api.rime.ai/ws"}
_STATUS = {"unauthenticated": 401, "permission_denied": 403, "not_found": 404}
_STATUS |= {"resource_exhausted": 429, "invalid_input": 400, "unavailable": 503}


class RimeTTS(LiveTTS):
    """A `StreamingTTS` on Rime (`model` `coda` or `mistv3`), one shared socket.

    `speaker` is a Rime voice of that model; `time_scale` above 1 speaks slower.
    Text is grouped into sentences as it arrives (`SPLIT_STRATEGY_ACCUMULATE_SENTENCE`).
    """

    provider = "rime"
    env = "RIME_API_KEY"
    shared = True

    def __init__(
        self,
        speaker: str,
        *,
        model: str = "coda",
        sample_rate: int = 24_000,
        language: str | None = "en",
        time_scale: float | None = None,
        timestamps: bool = True,
        api_key: str | None = None,
        url: str | None = None,
        connect: Connect | None = None,
    ) -> None:
        url = url or _HOSTS.get(model)
        if url is None:
            raise ValueError(f"pass url= for a Rime model other than {sorted(_HOSTS)}")
        super().__init__(model, sample_rate=sample_rate, api_key=api_key, url=url, connect=connect)
        self.speaker = speaker
        self.language = language
        self.time_scale = time_scale
        self.timestamps = timestamps

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key()}"}

    async def _ready(self, socket: LiveSocket) -> None:
        async for raw in socket:
            ready = parse(raw).get("ready")
            if isinstance(ready, dict):
                if ready.get("protocol") != 1:
                    raise failure(self.provider, self.model, "unsupported protocol version")
                return
            if (error := self._error(parse(raw))) is not None:
                raise error
        raise failure(self.provider, self.model, "closed before it was ready")

    def _first(self, context: str) -> list[Frame]:
        audio: dict[str, Any] = {"audioFormat": "audio/pcm", "samplingRate": self.sample_rate}
        if self.time_scale is not None:
            audio["timeScaleFactor"] = self.time_scale
        start: dict[str, Any] = {
            "speaker": self.speaker,
            "audioParameters": audio,
            "splitStrategy": "SPLIT_STRATEGY_ACCUMULATE_SENTENCE",
            "timestamps": {"enable": self.timestamps},
        }
        if self.language:
            start["language"] = self.language
        return [dumps({"contextId": context, "start": start})]

    def _text(self, context: str, text: str, flush: bool) -> list[Frame]:
        return [dumps({"contextId": context, "text": text})] if text else []  # no flush in v1

    def _end(self, context: str) -> list[Frame]:
        return [dumps({"contextId": context, "end": {}})]

    def _cancel(self, context: str) -> list[Frame]:
        return [dumps({"contextId": context, "cancel": {}})]

    def _decode(self, raw: Frame) -> tuple[str | None, Reply]:
        message = parse(raw)
        context = message.get("contextId")
        context = context if isinstance(context, str) else None
        if "audio" in message:
            return context, b64_audio(message["audio"])
        if isinstance(message.get("done"), dict):
            words = _spans(message["done"].get("timestamps"))
            return context, (words, DONE) if words else DONE
        if "cancelled" in message:
            return context, DONE
        return context, self._error(message)

    def _error(self, message: dict[str, Any]) -> Exception | None:
        error = message.get("error")
        if not isinstance(error, dict):
            return None
        kind = str(error.get("kind") or "")
        detail = f"{kind}: {error.get('message') or ''}".strip(": ")
        return failure(self.provider, self.model, detail, _STATUS.get(kind))


def _spans(timestamps: Any) -> list[WordTime]:
    spans = timestamps.get("spans") if isinstance(timestamps, dict) else None
    words: list[WordTime] = []
    for span in spans if isinstance(spans, list) else ():
        try:
            start, end = (float(str(span[k]).rstrip("s")) for k in ("start", "end"))
        except (KeyError, TypeError, ValueError):
            continue
        words.append(WordTime(str(span.get("text", "")), start, end))
    return words


__all__ = ["RimeTTS"]
