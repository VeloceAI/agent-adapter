"""Speculative replies: start answering on the early transcript, keep it only if it was final.

A `Speculation` runs the agent's tool loop before the turn is over. Its text and progress
lines are held back until the agent adopts it (the final transcript matched) and dropped if
the caller goes on. Tools wait for adoption at the executor, in `SpeculationGate`, which runs
before guards and approval: only tools named in `SpeculativeTools` that are declared
`effects="read_only"` and have no `confirm` may run early, within their own rate limits.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections import deque
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import Any, Protocol

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import Message
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolResult
from agent_adapter.hooks.base import Hook, LLMInvocation, ShortCircuit, ToolInvocation
from agent_adapter.tracing.events import emit, make_event
from agent_adapter.voice.meter import record_units

SPECULATIVE_STAGE = "voice_speculation"  # the ledger stage of a speculative reply's model calls
TOOL_ITEM = "voxcat:speculative_tool"  # the ledger line of each speculative tool call

_CURRENT: ContextVar[Speculation | None] = ContextVar("voice_speculation", default=None)


class ReplyOut(Protocol):
    """Where a turn's words go: the speech queue, or a tap in front of it."""

    def feed(self, chunk: str) -> None: ...

    def put(self, text: str, *, record: bool = True) -> None: ...

    def flush(self) -> None: ...


def same_words(text: str) -> str:
    """`text` reduced for an exact match: lower case, no punctuation, single spaces."""
    return " ".join(re.findall(r"[\w']+", text.lower()))


def can_speculate(spec: ToolSpec) -> bool:
    """A tool declared read-only, with no confirm rule: it may run before the turn is final."""
    return spec.effects == "read_only" and spec.confirm is False


class SpeculativeTools:
    """Tools a speculative reply may run before the caller has finished: opt-in by name.

    Each must be declared `effects="read_only"` with no `confirm`; the executor checks this
    on every call, not only here. At most `max_per_turn` run early per speculation and
    `max_per_minute` per call; past either, a call waits for the turn to be final. Each early
    call is billed as one unit of `budget_item` and traced as `custom.voice.speculative_tool`.
    Give them `cache_ttl`, so a dropped speculation's call is reused by the next one.
    """

    def __init__(
        self,
        names: Iterable[str],
        *,
        max_per_turn: int = 2,
        max_per_minute: int = 30,
        budget_item: str = TOOL_ITEM,
    ) -> None:
        self.names = frozenset(names)
        if not self.names:
            raise ToolUsageError("SpeculativeTools needs at least one tool name")
        if max_per_turn < 1 or max_per_minute < 1:
            raise ToolUsageError("max_per_turn and max_per_minute must be >= 1")
        self.max_per_turn = max_per_turn
        self.max_per_minute = max_per_minute
        self.budget_item = budget_item

    def check(self, registry: Registry) -> None:
        """Raise `ToolUsageError` if a named tool is missing or may change something."""
        for name in sorted(self.names):
            if name not in registry:
                raise ToolUsageError(f"speculative tool {name!r} is not in the registry")
            if not can_speculate(registry.get(name)):
                raise ToolUsageError(
                    f"speculative tool {name!r} must be effects='read_only' with no confirm"
                )

    def allows(self, spec: ToolSpec) -> bool:
        return spec.name in self.names and can_speculate(spec)


class Speculation:
    """One reply started on the early transcript; its output waits until it is adopted."""

    def __init__(self, text: str, history: list[Message]) -> None:
        self.text = text
        self.key = same_words(text)
        self.history = history
        self.started = time.monotonic()
        self.task: asyncio.Task[None] | None = None
        self.tool_calls = 0
        self.running_tools = 0
        self.dropped = False
        self.adopted = False
        self.failed = False
        self._settled = asyncio.Event()  # adopted or dropped
        self._held: list[tuple[str, str]] = []
        self._out: ReplyOut | None = None

    @property
    def ready(self) -> bool:
        """Some reply text is already waiting."""
        return any(kind == "feed" and text.strip() for kind, text in self._held)

    def matches(self, text: str, history: list[Message]) -> bool:
        """The final transcript is this one, and nothing was said since it started."""
        live = self.task is not None and not self.failed
        same = len(history) == len(self.history) and all(
            a is b for a, b in zip(history, self.history, strict=True)
        )
        return live and not self.dropped and same and same_words(text) == self.key

    def adopt(self, out: ReplyOut) -> None:
        """Keep it: replay the held output into `out` and pass the rest straight through."""
        held, self._held, self._out = self._held, [], out
        for kind, text in held:
            self._send(kind, text)
        self.adopted = True
        self._settled.set()

    def drop(self) -> None:
        """The caller went on: cancel it, once any early tool call has landed in the cache."""
        self.dropped = True
        self._settled.set()
        if self.task is not None and self.running_tools == 0:
            self.task.cancel()

    async def kept(self) -> None:
        """Return once adopted; a dropped speculation is cancelled here."""
        await self._settled.wait()
        if not self.adopted:
            raise asyncio.CancelledError

    def feed(self, chunk: str) -> None:
        self._send("feed", chunk)

    def put(self, text: str, *, record: bool = True) -> None:
        self._send("put", text)

    def flush(self) -> None:
        if self._out is not None:
            self._out.flush()

    def _send(self, kind: str, text: str) -> None:
        if self._out is None:
            self._held.append((kind, text))
        elif kind == "feed":
            self._out.feed(text)
        else:
            self._out.put(text, record=False)

    def bind(self) -> Any:
        """Mark the running task as this speculation (for the gate); returns a reset token."""
        return _CURRENT.set(self)


def current() -> Speculation | None:
    return _CURRENT.get()


class SpeculationGate(Hook):
    """Holds every tool call of an unadopted speculation, except allowed read-only ones.

    It runs first among the executor's hooks, so nothing (guards, budget, approval) sees a
    held call until the turn is final; a dropped speculation is cancelled while it waits.
    """

    name = "voice_speculation"

    def __init__(
        self, tools: SpeculativeTools | None, *, clock: Callable[[], float] = time.monotonic
    ):
        self.tools = tools
        self._clock = clock
        self._recent: deque[float] = deque()

    async def before_tool(self, inv: ToolInvocation) -> ShortCircuit | None:
        spec = current()
        if spec is None or spec.adopted:
            return None
        if self.tools is not None and self.tools.allows(inv.spec) and self._take(spec):
            spec.tool_calls += 1
            spec.running_tools += 1
            record_units(inv.ctx.ledger, inv.ctx, self.tools.budget_item, 1)
            event = make_event("custom.voice.speculative_tool", inv.ctx, tool=inv.spec.name)
            emit(inv.ctx.sink, event)
            return None
        await spec.kept()  # cancelled with the speculation if the caller goes on
        return None

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> ToolResult | None:
        self._landed()
        return None

    async def on_tool_error(self, inv: ToolInvocation, error: BaseException) -> ShortCircuit | None:
        self._landed()
        return None

    async def before_llm(self, inv: LLMInvocation) -> ShortCircuit | None:
        spec = current()
        if spec is not None and spec.dropped and not spec.adopted:
            raise asyncio.CancelledError  # its early tool calls are in the cache: stop here
        return None

    def _landed(self) -> None:
        spec = current()
        if spec is not None and not spec.adopted and spec.running_tools > 0:
            spec.running_tools -= 1

    def _take(self, spec: Speculation) -> bool:
        assert self.tools is not None
        now = self._clock()
        while self._recent and now - self._recent[0] > 60:
            self._recent.popleft()
        if spec.tool_calls >= self.tools.max_per_turn:
            return False
        if len(self._recent) >= self.tools.max_per_minute:
            return False
        self._recent.append(now)
        return True


async def cancel_quietly(task: asyncio.Task[Any] | None) -> None:
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


__all__ = [
    "SPECULATIVE_STAGE",
    "TOOL_ITEM",
    "ReplyOut",
    "Speculation",
    "SpeculationGate",
    "SpeculativeTools",
    "can_speculate",
    "current",
    "same_words",
]
