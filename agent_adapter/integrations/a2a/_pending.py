"""Calls parked in `input-required`, keyed by task owner and task id, kept server-side."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from agent_adapter.integrations.a2a._owned import OwnedTable, OwnerFull

logger = logging.getLogger(__name__)

PENDING_TTL_S = 24 * 3600.0
MAX_PENDING = 10_000
MAX_PENDING_PER_OWNER = 100


@dataclass(frozen=True)
class PendingCall:
    """A call waiting for approval; the follow-up message never changes what runs."""

    tool: str
    arguments: dict[str, Any]
    run_id: str
    approval_id: str
    mode: Literal["port", "client"]  # who decides: the executor's port, or the A2A client


class PendingCalls:
    """Bounded, expiring table of parked calls; each entry is taken once."""

    def __init__(
        self,
        *,
        ttl: float = PENDING_TTL_S,
        max_entries: int = MAX_PENDING,
        max_per_owner: int = MAX_PENDING_PER_OWNER,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._table: OwnedTable[PendingCall] = OwnedTable(
            ttl=ttl,
            max_entries=max_entries,
            max_per_owner=max_per_owner,
            evict_own=False,  # one owner never pushes out another owner's approvals
            clock=clock,
        )

    def __len__(self) -> int:
        return len(self._table)

    def park(self, owner: str, task_id: str, call: PendingCall) -> bool:
        """Remember `call` for the task's next message; False when the owner is at its cap."""
        try:
            _, dropped = self._table.put(owner, task_id, call)
        except OwnerFull:
            logger.warning(
                "A2A caller at its parked-call limit: %s (approval %s) not parked",
                call.tool,
                call.approval_id,
            )
            return False
        for _, _, old in dropped:
            logger.warning(
                "pending A2A calls full: dropped %s (approval %s)", old.tool, old.approval_id
            )
        return True

    def take(self, owner: str, task_id: str) -> PendingCall | None:
        """Remove and return the call parked for this owner's task."""
        return self._table.pop(owner, task_id)

    def drop(self, owner: str, task_id: str) -> None:
        """Forget the task's call (cancelled)."""
        self._table.pop(owner, task_id)
