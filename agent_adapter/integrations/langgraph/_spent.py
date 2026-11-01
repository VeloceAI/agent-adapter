"""One-use resumes: an approval answered in a LangGraph task is claimed once per task run.

LangGraph replays resume values on every re-run of a task, so each task run (a
`task_scope`, or `one_use` for your own node) claims the answers it uses. A pause,
cancellation or bad resume value releases its claims; any other error spends the claims
whose approved call started.
"""

from __future__ import annotations

import logging
from collections import Counter, OrderedDict, deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Unpack

from agent_adapter.core._bounded import Bounded, Bounds
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import ApprovalAlreadyDecided, ToolUsageError
from agent_adapter.core.types import ApprovalRequest

logger = logging.getLogger(__name__)

ClaimKey = tuple[str, ...]
NOT_SCOPED = "LangGraphApproval in your own node: wrap it with one_use(node)"


@dataclass(frozen=True)
class Asked:
    """An answer `LangGraphApproval` read in a task, in resume order (a one_use node's)."""

    request: ApprovalRequest
    key: bytes  # the port's MAC key
    finished: bool = False  # its call finished: the re-run replays it without asking


@dataclass(eq=False)
class Owner:
    """One execution of one task; the claims it holds and which of them started."""

    run_id: str
    namespace: str
    keys: dict[ClaimKey, bool] = field(default_factory=dict)  # key -> its call started
    seen: Counter[ClaimKey] = field(default_factory=Counter)  # occurrences in this execution
    consumed: set[ClaimKey] = field(default_factory=set)  # store records this run consumed
    replays: deque[Asked] = field(default_factory=deque)  # earlier executions' answers, in order
    asked: list[Asked] = field(default_factory=list)  # answers this execution read, in order


class ResumeClaims(Bounded):
    """Process-wide claims: a lock, at most `max_entries` spent keys for `ttl` seconds."""

    def __init__(self, **bounds: Unpack[Bounds]) -> None:
        super().__init__(**bounds)
        self._spent: OrderedDict[ClaimKey, float] = OrderedDict()
        self._unused: OrderedDict[ClaimKey, float] = OrderedDict()  # consumed, call never ran
        self._held: dict[ClaimKey, Owner] = {}

    def take(self, key: ClaimKey, owner: Owner, *, started: bool = True) -> None:
        """Claim `key` for `owner`.

        Raises:
            ApprovalAlreadyDecided: spent, or held by another live execution.
        """
        with self._lock:
            self._expire()
            holder = self._held.get(key)
            if key in self._spent or (holder is not None and holder is not owner):
                logger.warning("resume for run %s was already used (%s)", owner.run_id, key[-2])
                raise ApprovalAlreadyDecided(key[-2], status="resumed")
            self._held[key] = owner
            owner.keys[key] = owner.keys.get(key, False) or started

    def release(self, owner: Owner) -> None:
        """Let the next execution take all of `owner`'s claims (it paused)."""
        self._finish(owner, unstarted=False, started=False)

    def spend(self, owner: Owner) -> None:
        """`owner` finished: its claims are spent for good."""
        self._finish(owner, unstarted=True, started=True)

    def fail(self, owner: Owner) -> None:
        """`owner` failed: spend the claims whose call started, release the others."""
        self._finish(owner, unstarted=False, started=True)

    def _finish(self, owner: Owner, *, unstarted: bool, started: bool) -> None:
        with self._lock:
            now = self._clock()
            for key, ran in owner.keys.items():
                if self._held.get(key) is owner:
                    del self._held[key]
                if started if ran else unstarted:
                    self._spent[key] = now
                    self._spent.move_to_end(key)
                    self._unused.pop(key[:-1], None)
                elif key in owner.consumed and not ran:  # consumed, its call never started
                    self._unused[key[:-1]] = now
                    self._unused.move_to_end(key[:-1])
            owner.keys.clear()
            owner.consumed.clear()
            for kept in (self._spent, self._unused):
                while len(kept) > self.max_entries:
                    kept.popitem(last=False)

    def unused(self, *base: str) -> bool:
        """Whether a failed execution consumed this store record before its call started."""
        with self._lock:
            self._expire()
            return tuple(base) in self._unused

    def clear(self) -> None:
        """Forget everything (tests; a new process starts empty)."""
        with self._lock:
            self._spent.clear()
            self._unused.clear()
            self._held.clear()

    def _expire(self) -> None:
        horizon = self._horizon()
        for kept in (self._spent, self._unused):
            while kept and next(iter(kept.values())) < horizon:
                kept.popitem(last=False)


CLAIMS = ResumeClaims()
_OWNER: ContextVar[Owner | None] = ContextVar("aa_langgraph_owner", default=None)


@contextmanager
def task_scope(run_id: str, namespace: str) -> Iterator[Owner]:
    """One execution of a task: its claims are released on a pause, spent otherwise."""
    owner = Owner(run_id, namespace)
    token = _OWNER.set(owner)
    try:
        yield owner
    except BaseException as exc:
        if rerun_follows(exc):
            CLAIMS.release(owner)
        else:
            CLAIMS.fail(owner)
        raise
    else:
        CLAIMS.spend(owner)
    finally:
        _OWNER.reset(token)


def rerun_follows(exc: BaseException) -> bool:
    """Whether LangGraph re-runs the task after `exc` (a pause, cancel or invalid resume)."""
    from langgraph.errors import is_invalid_resume

    return is_control_flow(exc) or is_invalid_resume(exc)


def scoped_owner(run_id: str, namespace: str) -> Owner:
    """The running task's owner (the scope of this task; the request names its own run).

    Raises:
        ToolUsageError: no task scope for this task (an unwrapped user node).
    """
    owner = _OWNER.get()
    if owner is None or not namespace or owner.namespace != namespace:
        logger.warning("approval for run %s asked outside a task scope", run_id)
        raise ToolUsageError(NOT_SCOPED)
    return owner


def claim(run_id: str, namespace: str, *parts: str, started: bool = True) -> ClaimKey:
    """Claim one use of an answer in the running task; returns the claim's key.

    Args:
        started: The approved call starts now; else `mark_claims_started` marks it later.

    Raises:
        ApprovalAlreadyDecided: the answer was used by another or a finished execution.
        ToolUsageError: outside a task scope.
    """
    owner = scoped_owner(run_id, namespace)
    base = (run_id, namespace, *parts)
    owner.seen[base] += 1
    key = (*base, str(owner.seen[base]))
    CLAIMS.take(key, owner, started=started)
    return key


def note_consumed(run_id: str, namespace: str, *parts: str) -> None:
    """The running task consumed the store record its latest claim on `parts` stands for."""
    owner = scoped_owner(run_id, namespace)
    base = (run_id, namespace, *parts)
    owner.consumed.add((*base, str(owner.seen[base])))


def mark_claims_started(keys: Iterable[ClaimKey]) -> None:
    """Mark the running task's claims `keys` started (the executor used their answers)."""
    owner = _OWNER.get()
    if owner is None:
        return
    for key in keys:
        if key in owner.keys:
            owner.keys[key] = True


def mark_started(*parts: str) -> None:
    """Mark the running task's claims on `parts` started (e.g. the gated node now runs)."""
    owner = _OWNER.get()
    if owner is None:
        return
    for key in owner.keys:
        if key[2:-1] == parts:
            owner.keys[key] = True
