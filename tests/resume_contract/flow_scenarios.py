"""`FlowRunner` scenarios: checkpoint validity, consume-once and fork determinism."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent_adapter.approval import DeferredApproval
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.errors import AdapterError, FlowError
from agent_adapter.exec import Executor
from agent_adapter.exec.checkpoints import SQLiteCheckpointStore
from agent_adapter.flow import Flow, FlowRunner
from tests.resume_contract.flow_subject import (
    APPROVE,
    FLOW,
    RUN,
    FlowSubject,
    Trip,
    registry,
)
from tests.resume_contract.harness import (
    ACTIVE,
    EXECUTOR,
    KEY,
    Bank,
    FaultStore,
    SnapStore,
    World,
)
from tests.resume_contract.properties import PROPERTIES, Verdict

DENY = ApprovalDecision(approved=False, reason="no", decided_by="ana")
EDITED = ApprovalDecision(
    approved=True, edited_args={"order": "o-1", "amount": 9}, decided_by="ana"
)
REFUSED = ("CheckpointConflict", "ApprovalAlreadyDecided")


def _fresh_world() -> Bank:
    ACTIVE.world, ACTIVE.bank = World(), Bank()
    return ACTIVE.bank


async def _paused(subject: FlowSubject, inner: SnapStore) -> Any:
    out = await subject.start(FaultStore(inner, ACTIVE.world))
    assert out.status == "paused", out
    return out.token


def _ending(result: Any) -> Any:
    return (result.status, result.state.model_dump())


async def _branch(subject: FlowSubject, inner: SnapStore, cp: Any, decision: Any) -> Any:
    bank = _fresh_world()
    try:
        ending = _ending(
            await subject.runner(FaultStore(inner, ACTIVE.world)).resume(FLOW, cp, decision)
        )
    except AdapterError as exc:
        ending = ("error", exc.kind)
    return ending, bank.applied["refund"]


async def fork_determinism() -> Verdict:
    """Resumes of one paused checkpoint, each on its own copy of the durable state.

    There is no fork intent in one store (a second resume is refused: consume-once), so a fork
    is a byte copy of the checkpoint and approval stores. Each branch must end exactly as a run
    that was only ever resumed with that branch's decision.
    """
    branches = {"approve": APPROVE, "approve, amount edited": EDITED, "deny": DENY}
    failures: list[str] = []
    _fresh_world()
    base, inner = FlowSubject(), SnapStore()
    cp = await _paused(base, inner)
    for order in (list(branches), list(reversed(branches))):
        for name in order:
            got = await _branch(base.fresh(), inner.snapshot(), cp, branches[name])
            _fresh_world()
            alone, alone_inner = FlowSubject(), SnapStore()
            want = await _branch(
                alone, alone_inner, await _paused(alone, alone_inner), branches[name]
            )
            if got != want:
                failures.append(f"{name}: forked {got}, alone {want}")
    endings = {
        name: (await _branch(base.fresh(), inner.snapshot(), cp, d))[0]
        for name, d in branches.items()
    }
    if len({repr(e) for e in endings.values()}) != len(branches):
        failures.append(f"branches did not follow their decisions: {endings}")
    note = "fork = byte copy of the stores; no in-store fork intent (FI not offered)"
    return Verdict(PROPERTIES[2], "FlowRunner", 2 * len(branches) + len(branches), failures, note)


async def wrong_type(state: Trip) -> dict[str, Any]:
    return {"order": 7}


async def in_place(state: Trip) -> dict[str, Any]:
    state.log.append(7)  # type: ignore[arg-type]  # bypasses validation: no update returned
    return {}


async def first(state: Trip) -> dict[str, Any]:
    return {"log": ["first"]}


def _bad_flow(name: str, node: Any) -> Flow[Trip]:
    return Flow(name, state=Trip).node("ok", first).node("bad", node).edge("ok", "bad").end("bad")


async def _stored_state_is_valid(runner: FlowRunner, flow: Flow[Trip], run_id: str) -> str | None:
    try:
        _, checkpoint = await runner.inspect(flow, run_id)
        Trip.model_validate(checkpoint.state.model_dump(warnings=False))
    except (FlowError, ValidationError) as exc:
        return f"the store holds an invalid checkpoint ({type(exc).__name__})"
    return None


async def checkpoint_validity() -> Verdict:
    """A write of schema-invalid state is refused with an error and never stored."""
    failures: list[str] = []
    cases = 0
    for name, node in (("wrong type", wrong_type), ("mutated in place", in_place)):
        cases += 1
        flow, store = _bad_flow(f"cv_{name.replace(' ', '_')}", node), SnapStore()
        runner = FlowRunner(Executor(registry), checkpoints=store, checkpoint_key=KEY)
        try:
            await runner.run(flow, {}, runner.executor.context(run_id=f"cv-{cases}"))
            failures.append(f"{name}: the run accepted it")
        except FlowError:
            pass
        if problem := await _stored_state_is_valid(runner, flow, f"cv-{cases}"):
            failures.append(f"{name}: {problem}")
    cases += 1
    _fresh_world()
    subject, inner = FlowSubject(), SnapStore()
    cp = await _paused(subject, inner)
    runner = subject.runner(inner)
    forged = dataclasses.replace(cp, state=Trip.model_construct(order=7, log=[]))
    try:
        runner.load_checkpoint(FLOW, runner.dump_checkpoint(forged))
        failures.append("a signed checkpoint with an invalid state loaded")
    except FlowError:
        pass
    note = "a wrong-type update, an in-place mutation and a signed invalid state are all refused"
    return Verdict(PROPERTIES[3], "FlowRunner", cases, failures, note)


async def _concurrent(decisions: tuple[Any, Any]) -> list[str]:
    bank = _fresh_world()
    subject, inner = FlowSubject(), SnapStore()
    cp = await _paused(subject, inner)
    runs = [
        FlowSubject(subject.approvals).runner(FaultStore(inner, ACTIVE.world, yielding=True))
        for _ in decisions
    ]
    got = await asyncio.gather(
        *(r.resume(FLOW, cp, d) for r, d in zip(runs, decisions, strict=True)),
        return_exceptions=True,
    )
    return _one_winner(got, bank, f"decisions {[d.approved for d in decisions]}")


def _one_winner(got: list[Any], bank: Bank, what: str) -> list[str]:
    won = [g for g in got if not isinstance(g, BaseException)]
    lost = [type(g).__name__ for g in got if isinstance(g, BaseException)]
    failures = []
    if len(won) != 1 or any(name not in REFUSED for name in lost):
        failures.append(f"{what}: {len(won)} resumes won, the rest raised {lost}")
    if bank.attempts["refund"] > 1:
        failures.append(f"{what}: the refund ran {bank.attempts['refund']} times")
    return failures


def _threaded(tmp: Path) -> list[str]:
    """Two threads, each with its own loop, store and approval port, on one SQLite file."""
    bank = _fresh_world()
    db, approvals = str(tmp / "runs.db"), str(tmp / "approvals.db")

    def worker() -> FlowRunner:
        executor = Executor(registry, approval=DeferredApproval.sqlite(approvals))
        EXECUTOR.set(executor)
        return FlowRunner(executor, checkpoints=SQLiteCheckpointStore(db), checkpoint_key=KEY)

    async def pause() -> Any:
        r = worker()
        return (await r.run(FLOW, {}, r.executor.context(run_id=RUN))).checkpoint

    cp = asyncio.run(pause())
    barrier, got = threading.Barrier(2), []

    def resume() -> None:
        async def go() -> Any:
            r = worker()
            barrier.wait()
            return await r.resume(FLOW, cp, APPROVE)

        try:
            got.append(asyncio.run(go()))
        except BaseException as exc:  # the loser's error is the result
            got.append(exc)

    threads = [threading.Thread(target=resume) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    return _one_winner(got, bank, "two threads on SQLite")


async def _redelivered() -> list[str]:
    bank = _fresh_world()
    subject, inner = FlowSubject(), SnapStore()
    cp = await _paused(subject, inner)
    await subject.runner(FaultStore(inner, ACTIVE.world)).resume(FLOW, cp, APPROVE)
    failures = []
    for label, again in (
        ("the same resume again", lambda: subject.runner(inner).resume(FLOW, cp, APPROVE)),
        ("another answer after it", lambda: subject.runner(inner).resume(FLOW, cp, DENY)),
        ("a recover of the ended run", lambda: subject.runner(inner).recover(FLOW, RUN)),
    ):
        try:
            await again()
            failures.append(f"{label}: accepted")
        except AdapterError as exc:
            if type(exc).__name__ not in REFUSED:
                failures.append(f"{label}: raised {exc.kind}")
    if dict(bank.attempts) != dict.fromkeys(FlowSubject.effects, 1):
        failures.append(f"redelivery changed the effects: {dict(bank.attempts)}")
    return failures


async def consume_once(tmp: Path) -> Verdict:
    """One paused checkpoint is consumed by one resume; every later or rival resume does nothing."""
    failures = await _concurrent((APPROVE, APPROVE))
    failures += await _concurrent((APPROVE, DENY))
    failures += await asyncio.to_thread(_threaded, tmp)
    failures += await _redelivered()
    note = "rival resumes (same and different answers, asyncio and two threads on SQLite): one wins"
    return Verdict(PROPERTIES[4], "FlowRunner", 6, failures, note)
