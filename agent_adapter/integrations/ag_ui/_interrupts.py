"""Open AG-UI interrupts: which paused call each one stands for, and who may answer it."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations._expiring import ExpiringTable
from agent_adapter.integrations.ag_ui._support import METADATA_KEY, core

logger = logging.getLogger(__name__)

INTERRUPT_TTL_S = 900.0
MAX_INTERRUPTS = 1024
MAX_PER_PRINCIPAL = 64
REASON = "tool_call"  # the AG-UI convention for a tool approval
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"approved": {"type": "boolean"}},
    "required": ["approved"],
}


@dataclass(frozen=True)
class PendingCall:
    """One paused tool call, bound to its thread, the principal who started it and our run."""

    interrupt_id: str
    thread_id: str
    principal: str
    run_id: str  # the executor run the call belongs to (not the AG-UI run id)
    call_id: str
    tool: str
    arguments: Mapping[str, Any] | None = None  # None: unknown, so a resume fails closed
    approval_id: str = ""
    expires_at: float = 0.0  # epoch seconds, for `Interrupt.expires_at`

    def interrupt(self) -> Any:
        """The AG-UI `Interrupt` a client answers with a resume entry."""
        expires = datetime.fromtimestamp(self.expires_at, UTC).isoformat().replace("+00:00", "Z")
        return core().Interrupt(
            id=self.interrupt_id,
            reason=REASON,
            message=f"Approve the call to {self.tool}?",
            tool_call_id=self.call_id,
            response_schema=RESPONSE_SCHEMA,
            expires_at=expires,
            metadata={METADATA_KEY: {"approval_id": self.approval_id, "tool": self.tool}},
        )


class InterruptTable:
    """The bridge's open interrupts, each answerable once, within its TTL, by its owner."""

    def __init__(
        self,
        *,
        ttl: float = INTERRUPT_TTL_S,
        max_interrupts: int = MAX_INTERRUPTS,
        max_per_principal: int = MAX_PER_PRINCIPAL,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(max_per_principal, bool) or not isinstance(max_per_principal, int):
            raise ConfigError("max_per_principal must be a whole number")
        if max_per_principal < 1:
            raise ConfigError("max_per_principal must be at least 1")
        self._ttl = ttl
        self._wall = wall
        self._per_principal = max_per_principal
        self._lock = threading.Lock()  # count-then-drop must not interleave
        self._open: ExpiringTable[str, PendingCall] = ExpiringTable(
            ttl=ttl, max_entries=max_interrupts, clock=clock, label="AG-UI interrupt(s)"
        )

    def __len__(self) -> int:
        return len(self._open)

    def expiry(self) -> float:
        """The wall-clock time an interrupt opened now stops being answerable."""
        return self._wall() + self._ttl

    def open(self, pending: PendingCall) -> None:
        """Record `pending`; over the owner's cap or a full table, the oldest go (fail closed)."""

        def owned(other: PendingCall) -> bool:
            return other.principal == pending.principal

        with self._lock:
            dropped: list[PendingCall] = []
            while self._open.count(owned) >= self._per_principal:
                oldest = self._open.pop_first(owned)
                if oldest is None:
                    break
                dropped.append(oldest)
            dropped.extend(self._open.add(pending.interrupt_id, pending))
        for dropped_call in dropped:
            logger.warning(
                "ag_ui: interrupt cap reached; dropped interrupt for %s (call %s)",
                dropped_call.tool,
                dropped_call.call_id,
            )

    def take(self, interrupt_id: str, *, thread_id: str, principal: str) -> PendingCall | None:
        """Remove and return the open interrupt with this id, thread and principal.

        Anything else (unknown, expired, answered before, another thread or principal)
        gives None and leaves the table unchanged.
        """

        def matches(pending: PendingCall) -> bool:
            return (
                pending.interrupt_id == interrupt_id
                and pending.thread_id == thread_id
                and pending.principal == principal
            )

        return self._open.pop_first(matches)
