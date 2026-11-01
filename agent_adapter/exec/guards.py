"""Run guards and per-run concurrency: limits, call identities, the run lock, kind caps."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import Counter, OrderedDict, deque
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from agent_adapter.cache.keys import digest
from agent_adapter.core.errors import ConfigError, RunLimitExceeded
from agent_adapter.exec._canon import NotSealable
from agent_adapter.exec._canon import digest as canonical_digest
from agent_adapter.runtime.loop_local import LoopLocal

logger = logging.getLogger(__name__)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@dataclass(frozen=True)
class RunLimits:
    """Per-run limits; `per_kind` caps concurrent calls of one tool kind within a run."""

    max_calls: int | None = None  # distinct call ids admitted per run
    max_concurrency: int = 8  # default batch concurrency
    per_kind: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_calls is not None and not _positive_int(self.max_calls):
            raise ConfigError(f"RunLimits.max_calls must be an int >= 1, got {self.max_calls!r}")
        if not _positive_int(self.max_concurrency):
            raise ConfigError(
                f"RunLimits.max_concurrency must be an int >= 1, got {self.max_concurrency!r}"
            )
        caps = dict(self.per_kind)
        for kind, cap in caps.items():
            if not (isinstance(kind, str) and kind) or not _positive_int(cap):
                raise ConfigError(
                    f"RunLimits.per_kind needs {{kind: int >= 1}}, got {kind!r}: {cap!r}"
                )
        object.__setattr__(self, "per_kind", MappingProxyType(caps))


class _RunLock:
    """Readers-writer lock for one run on one loop; FIFO, so writers are not starved."""

    def __init__(self) -> None:
        self.readers = 0
        self.writer = False
        self._waiters: deque[tuple[bool, asyncio.Future[None]]] = deque()

    def _free_for(self, exclusive: bool) -> bool:
        return not self.writer and (not exclusive or self.readers == 0)

    def _take(self, exclusive: bool) -> None:
        if exclusive:
            self.writer = True
        else:
            self.readers += 1

    async def acquire(self, exclusive: bool) -> None:
        """Wait for the lock: exclusive needs it idle, shared needs no writer."""
        if not self._waiters and self._free_for(exclusive):
            self._take(exclusive)
            return
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (exclusive, waiter)
        self._waiters.append(entry)
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                self.release(exclusive)  # granted just before the cancel landed
            else:
                self._waiters.remove(entry)
                self._wake()
            raise

    def release(self, exclusive: bool) -> None:
        """Give the lock back and admit the waiters now allowed in."""
        if exclusive:
            self.writer = False
        else:
            self.readers -= 1
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self._free_for(self._waiters[0][0]):
            exclusive, waiter = self._waiters.popleft()
            self._take(exclusive)
            waiter.set_result(None)


class _LoopRun:
    """A run's asyncio primitives on one event loop."""

    def __init__(self) -> None:
        self.lock = _RunLock()
        self.active: Counter[str] = Counter()  # call ids running in the run's slots on this loop
        self.held: dict[str, frozenset[str]] = {}  # call id -> capped kinds held by its chain
        self._kinds: dict[tuple[str, int], asyncio.Semaphore] = {}

    def semaphore(self, kind: str, cap: int) -> asyncio.Semaphore:
        key = (kind, cap)
        if key not in self._kinds:
            self._kinds[key] = asyncio.Semaphore(cap)
        return self._kinds[key]


CallIdentity = tuple[str, str, str]  # (call id, tool, digest of the raw arguments)


def _call_identity(call_id: str, tool: str, arguments: Mapping[str, Any]) -> CallIdentity:
    """What makes a repeated call a re-entry (resume, re-run, replay) rather than a new call."""
    try:
        args = digest(dict(arguments))
    except TypeError:
        try:  # the tagged form covers what JSON cannot write (huge ints, bytes, tuples)
            args = canonical_digest(dict(arguments))
        except NotSealable:
            args = digest(repr(sorted(arguments.items(), key=lambda item: str(item[0]))))
    return (call_id, tool, args)


class _RunState:
    """Guard counters of one run (guarded by the tracker's threading.Lock) and its loop state."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.calls: set[CallIdentity] = set()  # committed calls
        self.once: set[str] = set()  # once_per_run tools already committed
        self.approvals: dict[str, str] = {}  # call id -> approval id of a pending approval
        self.active = 0  # calls in flight, on any loop
        self.loops: LoopLocal[_LoopRun] = LoopLocal(_LoopRun)


class _RunTracker:
    """Run states by run id, bounded: the oldest idle runs are forgotten first."""

    def __init__(self, max_runs: int) -> None:
        if not _positive_int(max_runs):
            raise ConfigError(f"max_tracked_runs must be an int >= 1, got {max_runs!r}")
        self.max_runs = max_runs
        self._runs: OrderedDict[str, _RunState] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)

    def admit(
        self, run_id: str, identity: CallIdentity, *, once_per_run: bool, max_calls: int | None
    ) -> _RunState:
        """Reject a call the guards would refuse; nothing is counted yet. Pair with `leave`.

        Raises:
            RunLimitExceeded: a second call of a `once_per_run` tool, or over `max_calls`.
        """
        with self._lock:
            state = self._state(run_id)
            self._check(state, identity, once_per_run, max_calls)
            state.active += 1
            return state

    def commit(
        self, state: _RunState, identity: CallIdentity, *, once_per_run: bool, max_calls: int | None
    ) -> None:
        """Count the call now that it will execute; a re-entry of a counted call is free.

        Raises:
            RunLimitExceeded: a concurrent call took the last slot since `admit`.
        """
        with self._lock:
            self._check(state, identity, once_per_run, max_calls)
            state.calls.add(identity)
            if once_per_run:
                state.once.add(identity[1])

    def _check(
        self, state: _RunState, identity: CallIdentity, once_per_run: bool, max_calls: int | None
    ) -> None:
        if identity in state.calls:
            return
        tool = identity[1]
        if once_per_run and tool in state.once:
            raise RunLimitExceeded(tool, "once_per_run", run_id=state.run_id)
        if max_calls is not None and len(state.calls) >= max_calls:
            raise RunLimitExceeded(tool, "max_calls", run_id=state.run_id)

    def leave(self, state: _RunState) -> None:
        """The call admitted into `state` has finished."""
        with self._lock:
            state.active -= 1

    def pending_approval(self, state: _RunState, call_id: str) -> str:
        """The approval id this call waits on, or ""."""
        with self._lock:
            return state.approvals.get(call_id, "")

    def set_pending_approval(self, state: _RunState, call_id: str, approval_id: str) -> None:
        """Remember (or, with "", forget) the approval a call waits on."""
        with self._lock:
            if approval_id:
                state.approvals[call_id] = approval_id
            else:
                state.approvals.pop(call_id, None)

    def end(self, run_id: str) -> None:
        """Forget a run's guards and locks."""
        with self._lock:
            self._runs.pop(run_id, None)

    def _state(self, run_id: str) -> _RunState:
        state = self._runs.get(run_id)
        if state is not None:
            self._runs.move_to_end(run_id)
            return state
        if len(self._runs) >= self.max_runs:
            self._evict()
        state = self._runs[run_id] = _RunState(run_id)
        return state

    def _evict(self) -> None:
        for run_id, state in self._runs.items():
            if state.active == 0:
                del self._runs[run_id]
                # Its guards restart if the run comes back; call end_run to avoid this.
                logger.info("run %s evicted from the executor's run table", run_id)
                return
        logger.warning("run table over %d entries: every tracked run is active", self.max_runs)


@asynccontextmanager
async def _run_slot(
    state: _RunState, call_id: str, parent_id: str, *, exclusive: bool, kind: str, cap: int | None
) -> AsyncIterator[None]:
    """Hold the run lock and the kind's semaphore while a call executes.

    A call nested in a running call of the same run skips the run lock (the parent may hold
    it) and the semaphores of kinds its ancestors hold; other capped kinds still wait.
    """
    loop_run = state.loops.get()
    nested = bool(parent_id) and loop_run.active[parent_id] > 0
    inherited = loop_run.held.get(parent_id, frozenset()) if nested else frozenset()
    held = inherited | {kind} if cap is not None else inherited
    limit: Any = nullcontext()
    if cap is not None and kind not in inherited:
        limit = loop_run.semaphore(kind, cap)
    if not nested:
        await loop_run.lock.acquire(exclusive)
    try:
        async with limit:
            with _active(loop_run, call_id, held):
                yield
    finally:
        if not nested:
            loop_run.lock.release(exclusive)


@contextmanager
def _active(loop_run: _LoopRun, call_id: str, held: frozenset[str]) -> Iterator[None]:
    loop_run.active[call_id] += 1
    loop_run.held[call_id] = held
    try:
        yield
    finally:
        loop_run.active[call_id] -= 1
        if loop_run.active[call_id] <= 0:
            del loop_run.active[call_id]
            loop_run.held.pop(call_id, None)
