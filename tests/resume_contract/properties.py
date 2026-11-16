"""The crash-matrix properties: prefix continuation, effect exactly-once, recovery determinism."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tests.resume_contract.harness import (
    ACTIVE,
    Bank,
    Case,
    Subject,
    World,
    drive,
    is_crash,
    schedules,
)

PROPERTIES = (
    "prefix continuation",
    "effect exactly-once",
    "fork determinism",
    "checkpoint validity",
    "consume-once",
    "recovery determinism",
)
EXTRA = "lost record refused (extra)"


@dataclass
class Verdict:
    prop: str
    subject: str
    cases: int
    failures: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def held(self) -> bool:
        return not self.failures


@dataclass
class Matrix:
    """Every crash schedule of a subject, driven to the end, next to the run with no crash."""

    reference: Case
    cases: list[Case]
    singles: int  # schedules with one crash; the rest have two


async def crash_matrix(make: Callable[[], Subject], depth: int = 2) -> Matrix:
    reference = await drive(make, {})
    plans = await schedules(make, depth)
    cases = [await drive(make, plan) for plan in plans]
    return Matrix(reference, cases, sum(1 for p in plans if len(p) == 1))


def prefix_continuation(name: str, m: Matrix) -> Verdict:
    """Each life goes on from the durable frontier and the run ends where the clean run ends."""
    failures = [f"{c.where()}: {p}" for c in m.cases for p in c.problems]
    failures += [
        f"{c.where()}: ended in {c.final}" for c in m.cases if c.final != m.reference.final
    ]
    note = "every recovery and resume starts on the durable frontier with its state"
    return Verdict(PROPERTIES[0], name, len(m.cases), failures, note)


def exactly_once(name: str, effects: tuple[str, ...], m: Matrix) -> Verdict:
    """Each effect lands once; a body runs again only for a call reported in doubt."""
    failures: list[str] = []
    reruns = 0
    for c in m.cases:
        for e in effects:
            if c.bank.applied[e] != 1:
                failures.append(f"{c.where()}: {e} applied {c.bank.applied[e]} times")
            allowed = 1 + sum(e in doubt for doubt in c.in_doubt)
            if c.bank.attempts[e] > allowed:
                failures.append(f"{c.where()}: {e} ran {c.bank.attempts[e]} times unasked")
            reruns += c.bank.attempts[e] - 1
    asked = sum(1 for c in m.cases if c.in_doubt)
    twice = sum(1 for c in m.cases if any(c.bank.applied[e] > 1 for e in effects))
    finished = sum(1 for c in m.cases if any(c.reran_finished(e) for e in effects))
    if failures:
        note = (
            f"{twice} schedules applied an effect twice; in {finished} a tool body that had "
            "already returned ran again"
        )
    else:
        note = (
            f"{asked} schedules left a call in doubt (RecoveryNeedsDecision); the {reruns} "
            "reruns the operator then allowed were absorbed by idempotency_key()"
        )
    return Verdict(PROPERTIES[1], name, len(m.cases), failures, note)


async def recovery_determinism(make: Callable[[], Subject], m: Matrix) -> Verdict:
    """Which calls replay and which run again depends on the durable store alone."""
    singles = [c.plan for c in m.cases if len(c.plan) == 1]
    failures: list[str] = []
    for plan in singles:
        case = await drive(make, plan, rd=True)
        failures += [f"{case.where()}: {r}" for r in case.rd]
    note = "each recovery redone on two byte copies of the store, with other scheduling"
    return Verdict(PROPERTIES[5], make().name, len(singles), failures, note)


async def lost_record(make: Callable[[], Subject], m: Matrix) -> Verdict:
    """Extra, not in the paper: a store that lost a record must be refused, not run on."""
    tried, failures = await _dropped_records_are_refused(make, [c.plan for c in m.cases])
    note = "one record dropped from the latest step's log (never the newest)"
    return Verdict(EXTRA, make().name, tried, failures, note)


async def _dropped_records_are_refused(
    make: Callable[[], Subject], plans: list[dict[int, int]]
) -> tuple[int, list[str]]:
    """A store that lost a durable record (not the newest one) must fail loudly, never go on."""
    tried, failures = 0, []
    for plan in (p for p in plans if list(p) == [0]):
        subject = make()
        inner = subject.durable()
        world = World(plan[0])
        ACTIVE.world, ACTIVE.bank = world, Bank()
        try:
            await subject.start(subject.view(inner, world))
        except BaseException as exc:
            if not is_crash(exc):
                raise
        for copy in subject.dropped(inner):
            tried += 1
            ACTIVE.world = World()
            try:
                await subject.fresh().recover(subject.view(copy, ACTIVE.world), rerun=True)
                failures.append(f"plan {plan}: a record was lost and the run went on")
            except BaseException as exc:
                if is_crash(exc) or not subject.refused(exc):
                    failures.append(f"plan {plan}: a lost record raised {type(exc).__name__}")
    return tried, failures
