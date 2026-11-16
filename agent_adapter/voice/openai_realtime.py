"""OpenAI Realtime (GA) wire format for `RealtimeAgent`.

Connect with `websockets.connect("wss://api.openai.com/v1/realtime?model=gpt-realtime",
additional_headers={"Authorization": f"Bearer {key}"})` and pass the connection in.
The server's own VAD stays on: it cancels the model's speech when the caller cuts in, and
this side keeps the tools safe. Each line the app asks for carries `metadata.aa_say`, so its
`response.done` tells the agent when an approval prompt has played.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.tool import ToolSpec
from agent_adapter.voice.realtime import (
    CallerSaid,
    PromptDone,
    RealtimeEvent,
    SpeechStarted,
    ToolCalled,
)

logger = logging.getLogger(__name__)
_MAX_OPEN = 64


class OpenAIRealtime:
    """`RealtimeCodec` for the OpenAI Realtime API; `session` adds fields to `session.update`."""

    prompt_done_events = True

    def __init__(
        self,
        *,
        transcribe_model: str = "gpt-4o-transcribe",
        session: Mapping[str, Any] | None = None,
    ) -> None:
        self.transcribe_model = transcribe_model
        self.session = dict(session or {})
        self._ids = itertools.count(1)
        self._open: dict[str, None] = {}  # tags of lines asked for and not yet done

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
            return [SpeechStarted(_item(message))]
        if kind == "conversation.item.input_audio_transcription.completed":
            text = message.get("transcript")
            ok = isinstance(text, str) and text.strip()
            return [CallerSaid(text, _item(message))] if ok and isinstance(text, str) else []
        if kind == "response.done":
            return self._done(message)
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
        tag = f"aa-{next(self._ids)}"
        if len(self._open) >= _MAX_OPEN:
            del self._open[next(iter(self._open))]
        self._open[tag] = None
        response = {"instructions": instructions, "metadata": {"aa_say": tag}}
        return [{"type": "response.create", "response": response}]

    def _done(self, message: Mapping[str, Any]) -> list[RealtimeEvent]:
        response = message.get("response")
        metadata = response.get("metadata") if isinstance(response, Mapping) else None
        tag = metadata.get("aa_say") if isinstance(metadata, Mapping) else None
        if isinstance(tag, str) and tag in self._open:
            del self._open[tag]
            return [PromptDone()]
        return []

    def note(self, text: str) -> list[dict[str, Any]]:
        content = [{"type": "input_text", "text": text}]
        item = {"type": "message", "role": "system", "content": content}
        return [{"type": "conversation.item.create", "item": item}]


def _item(message: Mapping[str, Any]) -> str:
    item = message.get("item_id")
    return item if isinstance(item, str) else ""


def _tool_called(message: Mapping[str, Any]) -> ToolCalled:
    raw = message.get("arguments")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) and raw else {}
    except (ValueError, RecursionError):
        arguments = None
    if arguments is not None and not isinstance(arguments, dict):
        arguments = None
    return ToolCalled(str(message.get("call_id", "")), str(message.get("name", "")), arguments)


__all__ = ["OpenAIRealtime"]
