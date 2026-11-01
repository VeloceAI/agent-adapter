"""Run ownership: the first principal to invoke into a run key owns it; ids are scoped by owner."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

from agent_adapter.runtime.ids import scoped_id
from agent_adapter.server.auth import Principal

logger = logging.getLogger(__name__)

MAX_RUNS = 100_000
PER_PRINCIPAL = 10_000


def owner_key(principal: Principal) -> str | None:
    """The principal as scoped ids see it: anonymous callers (id "") are all `None`."""
    return principal.id or None


def run_id_for(principal_id: str | None, run_key: str) -> str:
    """The executor's run id for a client run key (another owner never shares it)."""
    return scoped_id("run_", principal_id, run_key)


def thread_id_for(principal: Principal, thread_key: str) -> str:
    """The executor's thread id; "" stays "" (no thread)."""
    return scoped_id("thr_", owner_key(principal), thread_key) if thread_key else ""


def call_id_for(principal: Principal, idempotency_key: str) -> str:
    """The call id of an `Idempotency-Key`: no client can choose another's."""
    return scoped_id("call_", owner_key(principal), idempotency_key)


class RunOwners:
    """Bounded table run key -> owner id, least recently used first.

    Each principal keeps at most `per_principal` runs (its own oldest go first), so one
    caller cannot push the others' runs out. A forgotten run stays safe: its executor run
    id is scoped by owner, so a later claimant gets a different run.
    """

    def __init__(self, *, max_runs: int = MAX_RUNS, per_principal: int = PER_PRINCIPAL) -> None:
        self._owners: OrderedDict[str, str] = OrderedDict()
        self._by_owner: dict[str, OrderedDict[str, None]] = {}
        self._max_runs = max_runs
        self._per_principal = per_principal
        self._lock = threading.Lock()

    def claim(self, run_key: str, owner: str) -> bool:
        """True when `owner` owns `run_key` (claiming it if nobody does); False otherwise."""
        with self._lock:
            current = self._owners.get(run_key)
            if current is not None and current != owner:
                return False
            self._owners[run_key] = owner
            self._owners.move_to_end(run_key)
            runs = self._by_owner.setdefault(owner, OrderedDict())
            runs[run_key] = None
            runs.move_to_end(run_key)
            if len(runs) > self._per_principal:
                self._forget(next(iter(runs)))
                logger.warning("a principal's oldest run left the owner table (quota)")
            while len(self._owners) > self._max_runs:
                self._forget(next(iter(self._owners)))
            return True

    def owner(self, run_key: str) -> str | None:
        """The owner of `run_key`, or None when nobody has invoked into it (or it was forgotten)."""
        with self._lock:
            return self._owners.get(run_key)

    def _forget(self, run_key: str) -> None:
        owner = self._owners.pop(run_key)
        runs = self._by_owner[owner]
        del runs[run_key]
        if not runs:
            del self._by_owner[owner]
