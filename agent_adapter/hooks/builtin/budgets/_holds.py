"""The open reservations of one BudgetHook, by model call, until the call ends."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

from agent_adapter.hooks.base import LLMInvocation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hold:
    """One reservation waiting for its call's `on_llm_end`."""

    inv: LLMInvocation  # kept so its id() is not reused while the hold is open
    key: str
    reservation_id: str
    expire_at: float | None


class Holds:
    """Holds keyed by call; bounded, so calls that never end cannot grow it forever."""

    def __init__(self, *, max_open: int) -> None:
        self._max_open = max_open
        self._open: OrderedDict[int, Hold] = OrderedDict()
        self._lock = threading.Lock()

    def keep(
        self, inv: LLMInvocation, key: str, reservation_id: str, expire_at: float | None
    ) -> None:
        """Remember the hold of `inv`; past `max_open` the oldest is forgotten (WARNING)."""
        hold = Hold(inv, key, reservation_id, expire_at)  # allocated outside the lock
        forgotten = 0
        with self._lock:
            self._open[id(inv)] = hold
            while len(self._open) > self._max_open:
                self._open.popitem(last=False)
                forgotten += 1
        if forgotten:
            logger.warning("budget forgot an unsettled hold (more than %d open)", self._max_open)

    def take(self, inv: LLMInvocation) -> Hold | None:
        """The hold of `inv`, removed; None when it has none (or it was forgotten)."""
        with self._lock:
            hold = self._open.get(id(inv))
            if hold is None or hold.inv is not inv:
                return None
            del self._open[id(inv)]
            return hold

    def __len__(self) -> int:
        with self._lock:
            return len(self._open)
