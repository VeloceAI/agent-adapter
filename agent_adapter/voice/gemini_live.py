"""Gemini Live (BidiGenerateContent) wire format for `RealtimeAgent`.

Connect with `websockets.connect("wss://generativelanguage.googleapis.com/ws/google.ai.
generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=...")` and pass the
connection in. Gemini cancels pending calls when the caller cuts in (`toolCallCancellation`);
read-only ones stop, and one that changes things finishes and is reported as done.

Gemini sends the caller's words in pieces; they count as one answer once the model starts
its reply or the turn ends. Gemini has no "say exactly" message, so an approval prompt is a
text turn asking the model to read it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_adapter.core.tool import ToolSpec
from agent_adapter.voice.realtime import (
    CallerSaid,
    CallsCancelled,
    RealtimeEvent,
    SpeechStarted,
    ToolCalled,
)


class GeminiLive:
    """`RealtimeCodec` for the Gemini Live API; `setup` adds fields to the setup message."""

    def __init__(self, model: str, *, setup: Mapping[str, Any] | None = None) -> None:
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.extra = dict(setup or {})
        self._heard: list[str] = []

    def setup(self, instructions: str, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        declarations = [
            {"name": s.name, "description": s.description, "parameters": s.json_schema()}
            for s in tools
        ]
        setup: dict[str, Any] = {
            "model": self.model,
            "generationConfig": {"responseModalities": ["AUDIO"]},
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
        if instructions:
            setup["systemInstruction"] = {"parts": [{"text": instructions}]}
        if declarations:
            setup["tools"] = [{"functionDeclarations": declarations}]
        setup.update(self.extra)
        return [{"setup": setup}]

    def decode(self, message: Mapping[str, Any]) -> list[RealtimeEvent]:
        events: list[RealtimeEvent] = []
        calls = _get(message, "toolCall").get("functionCalls")
        for call in calls if isinstance(calls, list) else []:
            if isinstance(call, Mapping):
                args = call.get("args", {})
                arguments = args if isinstance(args, Mapping) else None
                call_id, name = str(call.get("id", "")), str(call.get("name", ""))
                events.append(ToolCalled(call_id, name, arguments))
        content = _get(message, "serverContent")
        for holder in (message, content):  # the docs place cancellations in both
            ids = _get(holder, "toolCallCancellation").get("ids")
            if isinstance(ids, list):
                events.append(CallsCancelled(tuple(str(i) for i in ids)))
        if content.get("interrupted") is True:
            events.append(SpeechStarted())
        heard = _get(content, "inputTranscription").get("text")
        if isinstance(heard, str):
            self._heard.append(heard)
        replying = "modelTurn" in content or "outputTranscription" in content
        if self._heard and (replying or content.get("turnComplete") is True):
            text, self._heard = "".join(self._heard).strip(), []
            if text:
                events.append(CallerSaid(text))
        return events

    def tool_result(self, call: ToolCalled, output: str, *, respond: bool) -> list[dict[str, Any]]:
        reply = {"id": call.call_id, "name": call.name, "response": {"result": output}}
        return [{"toolResponse": {"functionResponses": [reply]}}]

    def say(self, text: str) -> list[dict[str, Any]]:
        ask = f"(From the app, not the caller: read this to the caller word for word: {text})"
        return [_turn(ask, complete=True)]

    def note(self, text: str) -> list[dict[str, Any]]:
        return [_turn(f"(From the app, not the caller: {text})", complete=False)]


def _turn(text: str, *, complete: bool) -> dict[str, Any]:
    turn = {"role": "user", "parts": [{"text": text}]}
    return {"clientContent": {"turns": [turn], "turnComplete": complete}}


def _get(message: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = message.get(key)
    return value if isinstance(value, Mapping) else {}


__all__ = ["GeminiLive"]
