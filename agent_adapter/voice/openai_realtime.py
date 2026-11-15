"""OpenAI Realtime (GA) wire format for `RealtimeAgent`.

Connect with `websockets.connect("wss://api.openai.com/v1/realtime?model=gpt-realtime",
additional_headers={"Authorization": f"Bearer {key}"})` and pass the connection in.
The server's own VAD stays on: it cancels the model's speech when the caller cuts in, and
this side keeps the tools safe.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.tool import ToolSpec
from agent_adapter.voice.realtime import (
    CallerSaid,
    RealtimeEvent,
    SpeechStarted,
    ToolCalled,
)

logger = logging.getLogger(__name__)


class OpenAIRealtime:
    """`RealtimeCodec` for the OpenAI Realtime API; `session` adds fields to `session.update`."""

    def __init__(
        self,
        *,
        transcribe_model: str = "gpt-4o-transcribe",
        session: Mapping[str, Any] | None = None,
    ) -> None:
        self.transcribe_model = transcribe_model
        self.session = dict(session or {})

    def setup(self, instructions: str, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        session: dict[str, Any] = {
            "type": "realtime",
            "instructions": instructions,
            "tools": [
                {
                    "type": "function",
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.json_schema(),
                }
                for spec in tools
            ],
            "tool_choice": "auto",
            "audio": {"input": {"transcription": {"model": self.transcribe_model}}},
        }
        session.update(self.session)
        return [{"type": "session.update", "session": session}]

    def decode(self, message: Mapping[str, Any]) -> list[RealtimeEvent]:
        kind = message.get("type")
        if kind == "input_audio_buffer.speech_started":
            return [SpeechStarted()]
        if kind == "conversation.item.input_audio_transcription.completed":
            text = message.get("transcript")
            return [CallerSaid(text)] if isinstance(text, str) and text.strip() else []
        if kind == "response.function_call_arguments.done":
            return [_tool_called(message)]
        if kind == "error":
            logger.warning("openai realtime error: %s", message.get("error"))
        return []

    def tool_result(self, call: ToolCalled, output: str, *, respond: bool) -> list[dict[str, Any]]:
        item = {"type": "function_call_output", "call_id": call.call_id, "output": output}
        messages: list[dict[str, Any]] = [{"type": "conversation.item.create", "item": item}]
        if respond:
            messages.append({"type": "response.create"})
        return messages

    def say(self, text: str) -> list[dict[str, Any]]:
        instructions = f"Say exactly the following, then stop and wait: {text}"
        return [{"type": "response.create", "response": {"instructions": instructions}}]

    def note(self, text: str) -> list[dict[str, Any]]:
        content = [{"type": "input_text", "text": text}]
        item = {"type": "message", "role": "system", "content": content}
        return [{"type": "conversation.item.create", "item": item}]


def _tool_called(message: Mapping[str, Any]) -> ToolCalled:
    raw = message.get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) and raw else {}
    except ValueError:
        arguments = None
    if arguments is not None and not isinstance(arguments, dict):
        arguments = None
    return ToolCalled(str(message.get("call_id", "")), str(message.get("name", "")), arguments)


__all__ = ["OpenAIRealtime"]
