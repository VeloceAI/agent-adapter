"""`run_tool_loop` scenarios: checkpoint validity, consume-once and fork determinism."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent_adapter.core.messages import Message
from agent_adapter.errors import AdapterError, CheckpointStoreCorrupt
from agent_adapter.exec._durable_seal import open_checkpoint, seal_checkpoint
from agent_adapter.exec.checkpoints import StoredCheckpoint
from tests.resume_contract.flow_scenarios import DENY, EDITED, REFUSED, _fresh_world, _one_winner
from tests.resume_contract.harness import (
    ACTIVE,
    KEY,
    FaultStore,
    SnapStore,
    World,
    is_crash,
)
from tests.resume_contract.loop_subject import APPROVE, RUN, LoopSubject
from tests.resume_contract.properties import PROPERTIES, Verdict


async def _paused(subject: LoopSubject, inner: SnapStore) -> Any:
    out = await subject.start(FaultStore(inner, ACTIVE.world))
    assert out.status == "paused", out
    return out.token


async def _decide_and_resume(subject: LoopSubject, inner: SnapStore, token: Any, d: Any) -> Any:
    bank = _fresh_world()
    for record in await subject.approvals.list(status="pending"):
        await subject.approvals.decide(record.approval_id, d)
    try:
        ending = await subject.resume(FaultStore(inner, ACTIVE.world), token)
        result: Any = ending.final
    except AdapterError as exc:
        result = ("error", exc.kind)
    return result, bank.applied["refund"]


async def fork_determinism() -> Verdict:
    """Each branch resumes a byte copy of the paused loop with its own decision."""
    branches = {"approve": APPROVE, "approve, amount edited": EDITED, "deny": DENY}
    failures: list[str] = []
    _fresh_world()
    base, inner = LoopSubject(), SnapStore()
    token = await _paused(base, inner)
    endings = {}
    for name, decision in branches.items():
        got = await _decide_and_resume(base.fresh(), inner.snapshot(), token, decision)
        _fresh_world()
        alone, alone_inner = LoopSubject(), SnapStore()
        alone_token = await _paused(alone, alone_inner)
        want = await _decide_and_resume(alone, alone_inner, alone_token, decision)
        endings[name] = repr(got)
        if got != want:
            failures.append(f"{name}: forked {got}, alone {want}")
    if len(set(endings.values())) != len(branches):
        failures.append("branches did not follow their decisions")
    note = "fork = byte copy of the stores; no in-store fork intent (FI not offered)"
    return Verdict(PROPERTIES[2], "run_tool_loop", 2 * len(branches), failures, note)


async def _crashed_mid_run() -> SnapStore:
    """A loop paused, was approved, and its resume died right after its first save."""
    _fresh_world()
    subject, inner = LoopSubject(), SnapStore()
    token = await _paused(subject, inner)
    for record in await subject.approvals.list(status="pending"):
        await subject.approvals.decide(record.approval_id, APPROVE)
    ACTIVE.world = World(crash_at=1)
    try:
        await subject.resume(FaultStore(inner, ACTIVE.world), token)
    except BaseException as exc:
        if not is_crash(exc):
            raise
    return inner


async def _tampered(change: Any, *, resign: bool) -> bool:
    """True when recovery refused a stored loop checkpoint changed by `change`."""
    inner = await _crashed_mid_run()
    found = await inner.latest(RUN)
    assert found is not None
    if resign:
        envelope = open_checkpoint(KEY, found)
        body = json.loads(json.dumps(envelope.body))
        change(body)
        saved = seal_checkpoint(
            KEY, RUN, found.step, found.checkpoint, body, envelope.identity,
            carried=envelope.carried, spend=envelope.spend,
        )  # fmt: skip
    else:
        saved = found.checkpoint.model_copy(update={"data": change(found.checkpoint.data)})
    inner._runs[RUN] = StoredCheckpoint(run_id=RUN, step=found.step, checkpoint=saved)
    ACTIVE.world = World()
    try:
        await LoopSubject().recover(FaultStore(inner, ACTIVE.world), rerun=True)
    except CheckpointStoreCorrupt:
        return True
    return False


def _bad_role(body: dict[str, Any]) -> None:
    body["messages"][0]["role"] = "robot"


def _bad_steps(body: dict[str, Any]) -> None:
    body["steps"] = "many"


class _Checking(SnapStore):
    """Validates every loop checkpoint as it is written: its history must parse."""

    written = 0

    async def save(self, run_id: str, step: int, checkpoint: Any) -> None:
        await super().save(run_id, step, checkpoint)
        found = await self.latest(run_id)
        assert found is not None
        body = open_checkpoint(KEY, found).body
        [Message.model_validate(m) for m in body["messages"]]
        self.written += 1


async def checkpoint_validity() -> Verdict:
    """Only valid histories are written; an invalid stored one is refused on recovery."""
    failures: list[str] = []
    _fresh_world()
    subject, store = LoopSubject(), _Checking()
    token = await _paused(subject, store)
    await _decide_and_resume(subject, store, token, APPROVE)
    for label, change, resign in (
        ("a signed history with an invalid role", _bad_role, True),
        ("a signed step count that is not a number", _bad_steps, True),
        ("an unsigned change", lambda data: data.replace("o-1", "o-2"), False),
    ):
        if not await _tampered(change, resign=resign):
            failures.append(f"{label}: recovery went on")
    note = f"all {store.written} checkpoints written parse; 3 invalid stored ones are refused"
    return Verdict(PROPERTIES[3], "run_tool_loop", store.written + 3, failures, note)


async def consume_once() -> Verdict:
    """One paused loop is resumed once; a rival or a redelivered resume does nothing."""
    failures: list[str] = []
    bank = _fresh_world()
    subject, inner = LoopSubject(), SnapStore()
    token = await _paused(subject, inner)
    for record in await subject.approvals.list(status="pending"):
        await subject.approvals.decide(record.approval_id, APPROVE)
    rivals = [LoopSubject(subject.approvals) for _ in range(2)]
    got = await asyncio.gather(
        *(r.resume(FaultStore(inner, ACTIVE.world, yielding=True), token) for r in rivals),
        return_exceptions=True,
    )
    failures += _one_winner(list(got), bank, "two rival resumes")
    for label, again in (
        ("the same resume again", lambda: subject.resume(FaultStore(inner, World()), token)),
        (
            "a recover of the ended run",
            lambda: subject.recover(FaultStore(inner, World()), rerun=False),
        ),
    ):
        try:
            await again()
            failures.append(f"{label}: accepted")
        except AdapterError as exc:
            if type(exc).__name__ not in REFUSED:
                failures.append(f"{label}: raised {exc.kind}")
    try:
        [record] = await subject.approvals.list()
        await subject.approvals.decide(record.approval_id, DENY)
        failures.append("the approval took a second answer")
    except AdapterError as exc:
        if type(exc).__name__ not in REFUSED:
            failures.append(f"a second answer raised {exc.kind}")
    if dict(bank.attempts) != dict.fromkeys(LoopSubject.effects, 1):
        failures.append(f"the effects changed: {dict(bank.attempts)}")
    note = "two rival resumes: one wins; a redelivered resume and a second answer are refused"
    return Verdict(PROPERTIES[4], "run_tool_loop", 4, failures, note)
