"""Crash injection: numbered step boundaries, a store that freezes at the crash, an outside bank.

A *life* is one worker process: a start, a resume or a recovery. Every durable write is two
boundaries (before and after) and every tool or model body is one (after its effect). A plan
`{life: k}` kills life `life` at its boundary `k`: that boundary raises `Crash` and the store
refuses every later read and write of that life, as if the process were gone.
"""

from __future__ import annotations

import asyncio
import builtins
from collections import Counter, deque
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_adapter.approval import MemoryApprovalStore
from agent_adapter.errors import (
    CheckpointConflict,
    CheckpointNotFound,
    CheckpointStoreCorrupt,
    RecoveryNeedsDecision,
)
from agent_adapter.exec.checkpoints import (
    CallRecord,
    Checkpoint,
    MemoryCheckpointStore,
    StoredCheckpoint,
    idempotency_key,
)

KEY = b"resume-contract-key-of-32-bytes!"
MAX_LIVES = 12


class Crash(BaseException):
    """The worker died here: nothing below runs, nothing is cleaned up."""


def is_crash(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return exc.subgroup(lambda e: isinstance(e, Crash)) is not None
    return isinstance(exc, Crash)


class World:
    """One life: its numbered boundaries, its planned crash, and the nodes it entered."""

    def __init__(self, crash_at: int | None = None) -> None:
        self.crash_at = crash_at
        self.trace: list[str] = []
        self.entered: list[tuple[str, Any]] = []
        self.finished: set[str] = set()  # effects whose tool body returned in this life
        self.dead = False

    def boundary(self, label: str) -> None:
        if self.dead:
            raise Crash("after death")
        self.trace.append(label)
        if len(self.trace) - 1 == self.crash_at:
            self.dead = True
            raise Crash(label)


@dataclass
class Bank:
    """The outside system the effects land in; it survives every crash and dedupes by key."""

    attempts: Counter[str] = field(default_factory=Counter)
    applied: Counter[str] = field(default_factory=Counter)
    keys: set[tuple[str, str]] = field(default_factory=set)

    def apply(self, effect: str, key: str | None) -> None:
        self.attempts[effect] += 1
        if key is not None:
            if (effect, key) in self.keys:
                return
            self.keys.add((effect, key))
        self.applied[effect] += 1

    def copy(self) -> Bank:
        return Bank(Counter(self.attempts), Counter(self.applied), set(self.keys))


class Active:
    """What the module-level tools and nodes of a subject act on in the current life."""

    world: World = World()
    bank: Bank = Bank()


ACTIVE = Active()
EXECUTOR: ContextVar[Any] = ContextVar("resume_contract_executor")  # per task and thread


def effect(name: str) -> None:
    """A tool body's side effect: it lands at the bank, then the worker may die."""
    if ACTIVE.world.dead:  # a killed process runs nothing more, in any thread
        raise Crash("after death")
    ACTIVE.bank.apply(name, idempotency_key())
    ACTIVE.world.boundary(f"inside {name}")
    ACTIVE.world.finished.add(name)


def enter(node: str, state: Any) -> None:
    ACTIVE.world.entered.append((node, state))


class SnapStore(MemoryCheckpointStore):
    """The durable state: a memory store that can be copied byte for byte."""

    def snapshot(self) -> SnapStore:
        other = SnapStore()
        with self._lock:
            other._runs = dict(self._runs)
            other._calls = {k: dict(v) for k, v in self._calls.items()}
        return other

    def drop(self, run_id: str, seq: int) -> None:
        """Lose one call record of the run's current segment (a store that drops writes)."""
        segment = self._runs[run_id].checkpoint.segment
        log = self._calls[(run_id, segment)]
        for key, record in list(log.items()):
            if record.seq == seq:
                del log[key]


class SnapApprovals(MemoryApprovalStore):
    """The approval store is durable state too: copied with the checkpoints."""

    def snapshot(self) -> SnapApprovals:
        other = SnapApprovals()
        with self._lock:
            other._records = dict(self._records)
            other._by_call = {k: list(v) for k, v in self._by_call.items()}
            other._consumed = deque(self._consumed)
            other._decided = dict(self._decided)
        return other


class FaultStore:
    """A view of `inner` for one life: each write is two boundaries; dead means unreachable."""

    def __init__(self, inner: SnapStore, world: World, *, yielding: bool = False) -> None:
        self.inner, self.world, self.yielding = inner, world, yielding

    async def _reach(self) -> None:
        if self.world.dead:
            raise Crash("store unreachable after death")
        if self.yielding:
            await asyncio.sleep(0)

    async def _write(self, label: str, write: Any) -> None:
        await self._reach()
        self.world.boundary(f"before {label}")
        await write()
        self.world.boundary(f"after {label}")

    async def save(self, run_id: str, step: int, checkpoint: Checkpoint) -> None:
        label = f"save {step} {checkpoint.status}"
        await self._write(label, lambda: self.inner.save(run_id, step, checkpoint))

    async def record_call(self, run_id: str, segment: str, call: CallRecord) -> None:
        label = f"log {call.tool} {call.state}"
        await self._write(label, lambda: self.inner.record_call(run_id, segment, call))

    async def delete(self, run_id: str) -> bool:
        await self._reach()
        return await self.inner.delete(run_id)

    async def latest(self, run_id: str) -> StoredCheckpoint | None:
        await self._reach()
        return await self.inner.latest(run_id)

    async def list_unfinished(
        self, *, name: str | None = None, limit: int = 100
    ) -> builtins.list[StoredCheckpoint]:
        await self._reach()
        return await self.inner.list_unfinished(name=name, limit=limit)

    async def calls(self, run_id: str, segment: str) -> builtins.list[CallRecord]:
        await self._reach()
        return await self.inner.calls(run_id, segment)


class OurStores:
    """The store hooks of an agent-adapter subject: a `SnapStore` seen through `FaultStore`."""

    run_id: str

    def durable(self) -> SnapStore:
        return SnapStore()

    def view(self, inner: SnapStore, world: World, *, yielding: bool = False) -> FaultStore:
        return FaultStore(inner, world, yielding=yielding)

    def dropped(self, inner: SnapStore) -> list[SnapStore]:
        found = inner._runs.get(self.run_id)
        if found is None:
            return []
        log = inner._calls.get((self.run_id, found.checkpoint.segment), {})
        copies = []
        for seq in sorted(r.seq for r in log.values())[:-1]:
            copy = inner.snapshot()
            copy.drop(self.run_id, seq)
            copies.append(copy)
        return copies

    def refused(self, exc: BaseException) -> bool:
        return isinstance(exc, CheckpointStoreCorrupt)


@dataclass
class Outcome:
    status: str  # "done" or "paused"
    final: Any = None  # comparable end state
    token: Any = None  # what a resume needs when paused


class Subject(Protocol):
    """A durable runtime under test: a fixed workload it can start, resume and recover."""

    name: str
    run_id: str
    effects: tuple[str, ...]

    def fresh(self) -> Subject:
        """A new worker on a byte copy of this subject's other durable state (approvals)."""
        ...

    def durable(self) -> Any:
        """A new, empty checkpoint store with `snapshot()`."""
        ...

    def view(self, inner: Any, world: World, *, yielding: bool = False) -> Any:
        """The store as one life sees it: writes are boundaries, death makes it unreachable."""
        ...

    def dropped(self, inner: Any) -> list[Any]:
        """Copies of `inner`, each with one durable record lost (not the newest)."""
        ...

    def refused(self, exc: BaseException) -> bool:
        """Whether recovering a copy with a lost record failed as it should."""
        ...

    async def start(self, store: FaultStore) -> Outcome: ...

    async def resume(self, store: FaultStore, token: Any) -> Outcome: ...

    async def recover(self, store: FaultStore, *, rerun: bool) -> Outcome: ...

    async def ended(self, store: FaultStore) -> Outcome: ...

    async def position(self, store: FaultStore) -> Any: ...

    def continued(self, position: Any, world: World) -> str | None: ...


@dataclass
class Case:
    """One crash schedule driven to the end, and what it showed."""

    plan: dict[int, int]
    bank: Bank
    worlds: list[World] = field(default_factory=list)
    in_doubt: list[tuple[str, ...]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    rd: list[str] = field(default_factory=list)
    final: Any = None

    def reran_finished(self, effect: str) -> bool:
        """The effect's body returned in a life that then died, and a later life ran it again."""
        lives = [w for w in self.worlds if f"inside {effect}" in w.trace]
        return any(w.dead and effect in w.finished for w in lives[:-1])

    def where(self) -> str:
        return ", ".join(
            f"life {life} at {self.worlds[life].trace[k]!r}"
            for life, k in sorted(self.plan.items())
        )


def _use(world: World, bank: Bank) -> tuple[World, Bank]:
    old = (ACTIVE.world, ACTIVE.bank)
    ACTIVE.world, ACTIVE.bank = world, bank
    return old


async def drive(make: Callable[[], Subject], plan: dict[int, int], *, rd: bool = False) -> Case:
    """Run the workload on a new subject, crash as planned, recover (or resume) until it ends."""
    subject, bank = make(), Bank()
    inner = subject.durable()
    case = Case(dict(plan), bank)
    action, token, rerun = "start", None, False
    for life in range(MAX_LIVES):
        world = World(plan.get(life))
        case.worlds.append(world)
        if action == "recover" and rd:
            case.rd += await _recovery_is_a_function_of_the_store(subject, inner, bank, rerun)
        position = None if action == "start" else await _position(subject, inner)
        _use(world, bank)
        store = subject.view(inner, world)
        try:
            if action == "start":
                out = await subject.start(store)
            elif action == "resume":
                out = await subject.resume(store, token)
            else:
                out = await subject.recover(store, rerun=rerun)
        except RecoveryNeedsDecision as exc:
            case.in_doubt.append(tuple(sorted(tool for tool, _ in exc.calls)))
            action, rerun = "recover", True  # the operator checked the bank: rerun, keyed
            continue
        except CheckpointNotFound:
            action, rerun = "start", False  # died before step 0 was stored: nothing began
            continue
        except CheckpointConflict:
            out = await subject.ended(subject.view(inner, World()))  # died after the last save
        except BaseException as exc:
            if not is_crash(exc):
                raise
            action, rerun = "recover", False
            continue
        finally:
            if position is not None and (msg := subject.continued(position, world)):
                case.problems.append(f"life {life}: {msg}")
        if out.status == "paused":
            action, token, rerun = "resume", out.token, False
            continue
        case.final = out.final
        return case
    raise AssertionError(f"{subject.name}: no end after {MAX_LIVES} lives ({case.where()})")


async def _position(subject: Subject, inner: Any) -> Any:
    try:
        return await subject.position(subject.view(inner, World()))
    except CheckpointNotFound:
        return None


async def _signature(subject: Subject, inner: Any, bank: Bank, rerun: bool, yld: bool) -> Any:
    world = World()
    old = _use(world, bank.copy())
    try:
        store = subject.view(inner.snapshot(), world, yielding=yld)
        out = await subject.fresh().recover(store, rerun=rerun)
        return ("ran", out.status, out.final, sorted(Counter(world.trace).items()))
    except RecoveryNeedsDecision as exc:
        return ("decide", sorted(exc.calls))
    except (CheckpointConflict, CheckpointNotFound) as exc:
        return (type(exc).__name__,)
    finally:
        _use(*old)


async def _recovery_is_a_function_of_the_store(
    subject: Subject, inner: Any, bank: Bank, rerun: bool
) -> list[str]:
    """Two fresh workers recover byte copies of the store, one with other scheduling."""
    first = await _signature(subject, inner, bank, rerun, False)
    second = await _signature(subject, inner, bank, rerun, True)
    return [] if first == second else [f"{first!r} != {second!r}"]


async def schedules(make: Callable[[], Subject], depth: int) -> list[dict[int, int]]:
    """Every crash schedule of up to `depth` crashes, each in a later life than the last."""
    found: list[dict[int, int]] = []
    level: list[dict[int, int]] = [{}]
    for _ in range(depth):
        nxt: list[dict[int, int]] = []
        for plan in level:
            case = await drive(make, plan)
            for life in range(max(plan, default=-1) + 1, len(case.worlds)):
                nxt += [{**plan, life: k} for k in range(len(case.worlds[life].trace))]
        found += nxt
        level = nxt
    return found
