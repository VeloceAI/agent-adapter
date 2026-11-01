"""A running turn's event list: written on the loop thread, read by HTTP stream threads."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from agent_adapter.core.types import to_jsonable
from agent_adapter.hooks.builtin.logging import redact_secrets

MAX_EVENTS = 20_000  # per turn; text chunks past it are dropped, the end event never is
MAX_ANSWER = 1_000_000  # characters kept of one answer
MAX_PREVIEW = 2_048  # characters kept of tool arguments or output
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

    def tool(self, call_id: str, **fields: Any) -> None:
        """Merge fields into a tool call's summary and send the whole summary."""
        with self._cond:
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
