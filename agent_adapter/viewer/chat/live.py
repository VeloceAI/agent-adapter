"""A running turn's event list: written on the loop thread, read by HTTP stream threads."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from agent_adapter.core._partial_json import PartialJSON
from agent_adapter.core.messages import ToolCallDelta
from agent_adapter.core.types import to_jsonable
from agent_adapter.hooks.builtin.logging import redact_secrets

MAX_EVENTS = 20_000  # per turn; text chunks past it are dropped, the end event never is
MAX_ANSWER = 1_000_000  # characters kept of one answer
MAX_PREVIEW = 2_048  # characters kept of tool arguments or output
MAX_DRAFTS = 64  # tool calls streaming at once in one turn; more are not shown
MAX_NAME = 200  # characters kept of a streaming tool call's name
TERMINAL = frozenset({"done", "error"})


def clean(text: str) -> str:
    """`text` with lone surrogates replaced, so it can always be saved and sent."""
    return text.encode("utf-8", "replace").decode("utf-8")


def preview(value: Any) -> str:
    """`value` as JSON text with secret-named keys masked, cut to `MAX_PREVIEW` characters.

    A string that holds a JSON object or array is masked the same way.
    """
    if isinstance(value, str):
        value = _parsed(value)
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(redact_secrets(to_jsonable(value)), ensure_ascii=False, default=str)
    text = clean(text)
    if len(text) <= MAX_PREVIEW:
        return text
    return f"{text[:MAX_PREVIEW]}... [+{len(text) - MAX_PREVIEW} chars]"


def _parsed(text: str) -> Any:
    """The JSON object or array a string holds, else the string itself."""
    if not text.lstrip().startswith(("{", "[")):
        return text
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return text


class _Draft:
    """A tool call whose arguments are still streaming from the model."""

    def __init__(self, delta: ToolCallDelta) -> None:
        self.call_id = delta.id[:MAX_NAME]
        self.name = delta.name[:MAX_NAME]
        self.arguments = PartialJSON()
        self.shown = ""
        self.sent = False


def _masked(value: Any) -> str:
    """Partial arguments as JSON text with secret-named keys masked, cut to `MAX_PREVIEW`."""
    text = clean(json.dumps(redact_secrets(to_jsonable(value)), ensure_ascii=False, default=str))
    return text if len(text) <= MAX_PREVIEW else f"{text[:MAX_PREVIEW]}..."


class LiveTurn:
    """Events of one turn in order; `seq` is the index. Ends with one `done` or `error`."""

    def __init__(self, turn_id: str, session_id: str, run_id: str) -> None:
        self.turn_id = turn_id
        self.session_id = session_id
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self.ended = False
        self.dropped = 0
        self.task: asyncio.Task[None] | None = None
        self.stop_reason = ""  # set by stop() before it cancels the task
        self.answer: list[str] = []
        self.answer_chars = 0
        self.tools: dict[str, dict[str, Any]] = {}
        self.drafts: dict[int, _Draft] = {}  # by stream index; never saved with the turn
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._cond = threading.Condition(threading.RLock())

    def push(self, type: str, **data: Any) -> None:
        """Append one event and wake the readers; nothing is added after the end event."""
        with self._cond:
            if self.ended:
                return
            if len(self.events) >= MAX_EVENTS and type not in TERMINAL:
                self.dropped += 1
                return
            self.events.append({"seq": len(self.events), "type": type, **data})
            self.ended = type in TERMINAL
            self._cond.notify_all()

    def text(self, chunk: str) -> None:
        """Add answer text, up to `MAX_ANSWER` characters."""
        with self._cond:
            room = MAX_ANSWER - self.answer_chars
            if room <= 0 or not chunk:
                return
            kept = clean(chunk[:room])
            self.answer.append(kept)
            self.answer_chars += len(kept)
            self.push("text", text=kept)

    def draft(self, delta: ToolCallDelta) -> None:
        """Show a streaming tool call's arguments as they arrive: parsed, masked, cut.

        A delta with another call id starts a new call at its index; unparsable text shows
        nothing. Past `MAX_DRAFTS` open drafts, new indexes are not shown.
        """
        with self._cond:
            found = self.drafts.get(delta.index)
            other = (
                found is not None
                and bool(delta.id and found.call_id)
                and (delta.id[:MAX_NAME] != found.call_id)
            )
            if found is None or other:
                if found is not None:
                    self.push("tool_draft", index=delta.index, call_id="", closed=True)
                elif len(self.drafts) >= MAX_DRAFTS:
                    return
                found = self.drafts[delta.index] = _Draft(delta)
            found.call_id = found.call_id or delta.id[:MAX_NAME]
            found.name = found.name or delta.name[:MAX_NAME]
            changed, value = found.arguments.feed(delta.arguments_delta)
            shown = _masked(value) if changed else found.shown
            if found.sent and shown == found.shown:
                return
            found.shown, found.sent = shown, True
            self.push(
                "tool_draft",
                index=delta.index,
                call_id=found.call_id,
                name=clean(found.name),
                arguments=shown,
            )

    def close_drafts(self) -> None:
        """Drop every open draft (a new model call started: their calls never ran)."""
        with self._cond:
            for index in sorted(self.drafts):
                self.push("tool_draft", index=index, call_id="", closed=True)
            self.drafts.clear()

    def _close_draft(self, call_id: str, name: str) -> None:
        """The draft this call came from (same id, else the first unnamed-id draft of `name`)."""
        match = next((i for i, d in self.drafts.items() if d.call_id == call_id), None)
        if match is None:
            match = next(
                (i for i, d in sorted(self.drafts.items()) if not d.call_id and d.name == name),
                None,
            )
        if match is not None:
            del self.drafts[match]
            self.push("tool_draft", index=match, call_id=call_id, closed=True)

    def tool(self, call_id: str, **fields: Any) -> None:
        """Merge fields into a tool call's summary and send the whole summary."""
        with self._cond:
            if call_id not in self.tools and self.drafts:
                self._close_draft(call_id, str(fields.get("name") or ""))
            summary = self.tools.setdefault(
                call_id, {"call_id": call_id, "name": "", "status": "running", "ms": None}
            )
            summary.update({key: value for key, value in fields.items() if value is not None})
            self.push("tool", **summary)

    def usage(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        """Add one model call's usage; the event carries the turn's totals."""
        with self._cond:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cost_usd += cost_usd
            self.push(
                "usage",
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cost_usd=round(self.cost_usd, 8),
            )

    def read(self, since: int, timeout: float) -> tuple[list[dict[str, Any]], bool]:
        """Events from index `since` (waiting up to `timeout` s for one), and whether it ended."""
        with self._cond:
            if len(self.events) <= since and not self.ended:
                self._cond.wait(timeout)
            return list(self.events[since:]), self.ended

    def tool_list(self) -> list[dict[str, Any]]:
        """Copies of the tool call summaries, in call order."""
        with self._cond:
            return [dict(summary) for summary in self.tools.values()]

    def wake(self) -> None:
        """Wake every waiting reader (the server is closing)."""
        with self._cond:
            self._cond.notify_all()
