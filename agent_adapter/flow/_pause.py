"""A run's position, and the sealed checkpoint and result it pauses (or is denied) with."""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel

from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec._canon import NotSealable
from agent_adapter.exec._journal import JournalData, unsealable_calls
from agent_adapter.exec._resume import Pending
from agent_adapter.flow._nodes import NodeOutcome
from agent_adapter.flow._result import FlowCheckpoint, FlowResult, ResumeEntry, frozen
from agent_adapter.flow._seal import Sealer, not_sealable, seal
from agent_adapter.flow._withdraw import SUPERSEDED, withdraw

logger = logging.getLogger(__name__)


@dataclass
class Cursor:
    """The run's position: state, frontier, counters, granted gates, finished paused nodes."""

    state: BaseModel
    frontier: tuple[str, ...]
    visits: dict[str, int] = field(default_factory=dict)
    loops: dict[str, int] = field(default_factory=dict)
    steps: int = 0
    granted: dict[str, ApprovalDecision] = field(default_factory=dict)
    writes: dict[str, NodeOutcome] = field(default_factory=dict)
    answers: dict[str, tuple[ResumeEntry, ...]] = field(default_factory=dict)  # resumed approvals
    items: dict[str, Mapping[int, tuple[str, Any]]] = field(default_factory=dict)  # Map items
    journal: dict[str, JournalData] = field(default_factory=dict)  # replays
    children: dict[str, FlowCheckpoint] = field(default_factory=dict)  # paused sub-flow runs


@dataclass(frozen=True)
class RunInfo:
    """What a pause needs to know about its run."""

    flow: str
    owned: bool  # the runner made the run and ends it
    digest: str  # the graph's shape, sealed into its checkpoints
    ctx: ToolContext  # the run's context (events)
    gate_store: ApprovalStore | None  # the store behind the gate port, if any


def cursor_from(
    checkpoint: FlowCheckpoint,
    *,
    granted: Mapping[str, ApprovalDecision] | None = None,
    answers: Mapping[str, tuple[ResumeEntry, ...]] | None = None,
    children: Mapping[str, FlowCheckpoint] | None = None,
) -> Cursor:
    """The position a checkpoint resumes from, with the resume's decisions put in."""
    return Cursor(
        state=checkpoint.state,
        frontier=checkpoint.frontier,
        visits=dict(checkpoint.visits),
        loops=dict(checkpoint.loops),
        steps=checkpoint.steps,
        granted=dict(checkpoint.decisions if granted is None else granted),
        writes={n: NodeOutcome(n, updates=u) for n, u in checkpoint._writes.items()},
        answers=dict(checkpoint._answers if answers is None else answers),
        items=dict(checkpoint._items),
        journal=dict(checkpoint._journal),
        children=dict(checkpoint._children if children is None else children),
    )


def build_checkpoint(
    run: RunInfo,
    cursor: Cursor,
    pending: ApprovalRequest | None,
    decisions: Mapping[str, ApprovalDecision],
    outcomes: Sequence[NodeOutcome],
) -> tuple[FlowCheckpoint, list[Pending], list[NodeOutcome]]:
    """The unsealed checkpoint, every pending record it made, and the waiting outcomes."""
    waiting = sorted((o for o in outcomes if o.pending is not None), key=lambda o: o.node)
    if pending is None:
        pending = cast(ApprovalRequest, waiting[0].pending)
        pendings = [made for o in waiting for made in o.pendings]
    else:
        pendings = [(pending, run.gate_store)]
    answers = {o.node: cursor.answers[o.node] for o in waiting if o.node in cursor.answers}
    checkpoint = FlowCheckpoint(
        flow=run.flow,
        state=cursor.state,
        frontier=cursor.frontier,
        visits=frozen(cursor.visits),
        loops=frozen(cursor.loops),
        steps=cursor.steps,
        pending=pending,
        decisions=frozen(decisions),
        _writes=frozen({o.node: o.updates for o in outcomes if o.pending is None}),
        _owned=run.owned,
        _answers=frozen(answers),
        _items=frozen({o.node: frozen(dict(o.items)) for o in waiting if o.items}),
        _journal=frozen({o.node: o.journal for o in waiting if o.journal}),
        _children=frozen({o.node: o.child for o in waiting if o.child is not None}),
        _graph=run.digest,
    )
    return checkpoint, pendings, waiting


async def paused(
    sealer: Sealer,
    run: RunInfo,
    cursor: Cursor,
    pending: ApprovalRequest | None,
    decisions: Mapping[str, ApprovalDecision],
    outcomes: Sequence[NodeOutcome],
) -> FlowResult[Any]:
    """A paused result; `pending` is a gate's request, else the first waiting node's.

    Raises:
        FlowError: `state-not-sealable`: some part cannot be sealed; every pending record the
            pause made is withdrawn first (denied and consumed), so nothing runs it later.
    """
    checkpoint, pendings, waiting = build_checkpoint(run, cursor, pending, decisions, outcomes)
    try:
        _check_journals(checkpoint)
        checkpoint = dataclasses.replace(checkpoint, _mac=seal(sealer, checkpoint))
    except (NotSealable, ValueError) as exc:
        await withdraw(run.flow, run.ctx, pendings)
        problem = exc if isinstance(exc, NotSealable) else NotSealable("$", "value", str(exc))
        raise not_sealable(run.flow, problem) from exc
    surfaced = cast(ApprovalRequest, checkpoint.pending)
    stale = _superseded(waiting, surfaced)
    if stale:
        await withdraw(run.flow, run.ctx, stale, SUPERSEDED)
    return FlowResult(
        state=cursor.state,
        status="paused",
        steps=cursor.steps,
        visits=checkpoint.visits,
        checkpoint=checkpoint,
        pending=surfaced,
    )


async def child_paused(
    run: RunInfo,
    cursor: Cursor,
    pending: ApprovalRequest | None,
    decisions: Mapping[str, ApprovalDecision],
    outcomes: Sequence[NodeOutcome],
) -> tuple[FlowCheckpoint, list[Pending]]:
    """A sub-flow's unsealed checkpoint and the records it keeps; stale ones are withdrawn."""
    checkpoint, pendings, waiting = build_checkpoint(run, cursor, pending, decisions, outcomes)
    stale = _superseded(waiting, cast(ApprovalRequest, checkpoint.pending))
    if stale:
        await withdraw(run.flow, run.ctx, stale, SUPERSEDED)
    gone = {request.approval_id for request, _ in stale}
    return checkpoint, [made for made in pendings if made[0].approval_id not in gone]


def _superseded(waiting: Sequence[NodeOutcome], pending: ApprovalRequest) -> list[Pending]:
    """Records a resume never reuses: not the checkpoint's, with a call id the flow did not assign.

    A `Call` / `Map(tool=)` call id (`{visit_id}:{i}`, or under a sub-flow `{visit_id}/...:{i}`)
    repeats on resume, so the port finds its record again; any other id (a node function's own
    calls) asks afresh under a new one.
    """
    stale = []
    for outcome in waiting:
        assigned = re.compile(re.escape(outcome.visit_id) + r"(?:/.+)?:\d+")
        for request, store in outcome.pendings:
            if request.approval_id == pending.approval_id:
                continue
            if outcome.visit_id and assigned.fullmatch(request.call_id):
                continue
            stale.append((request, store))
    return stale


def _check_journals(checkpoint: FlowCheckpoint, where: str = "journal") -> None:
    """Refuse a pause whose nodes ran a call that a resume could neither replay nor re-run."""
    for node, data in checkpoint._journal.items():
        problems = unsealable_calls(data)
        if problems:
            raise NotSealable(f"{where}.{node}", "call", problems[0])
    for node, child in checkpoint._children.items():
        _check_journals(child, f"{where}.{node}")


def denied(cursor: Cursor, node: str) -> FlowResult[Any]:
    return FlowResult(
        state=cursor.state,
        status="denied",
        steps=cursor.steps,
        visits=frozen(cursor.visits),
        denied_at=node,
    )
