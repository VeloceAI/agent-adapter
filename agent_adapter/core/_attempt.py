"""Which executor attempt is running (a ContextVar the executor sets and `ToolSpec.call` clears)."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Attempt:
    """Which attempt of which call is running: a tool can tell whether a retry may follow."""

    call_id: str
    number: int  # 1-based
    total: int  # retries + 1

    @property
    def last(self) -> bool:
        return self.number >= self.total


ATTEMPT: ContextVar[Attempt | None] = ContextVar("agent_adapter_attempt", default=None)
