"""Crash-safe flow runs: a sealed checkpoint after every superstep, and recovery from the last."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    CheckpointConflict,
    CheckpointNotFound,
    FlowError,
    log_failure,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec._canon import NotSealable
from agent_adapter.exec._durable import CallLog, check_in_doubt, idempotent_in, new_segment
from agent_adapter.exec._durable_seal import (
    Envelope,
    corrupt,
    merge_calls,
    open_checkpoint,
    seal_checkpoint,
    verified_calls,
)
from agent_adapter.exec.checkpoints import (
    CallRecord,
    Checkpoint,
    CheckpointStatus,
    CheckpointStore,
    StoredCheckpoint,
)
from agent_adapter.flow._pause import Cursor, _check_journals
from agent_adapter.flow._result import FlowCheckpoint, FlowResult, frozen
from agent_adapter.flow._seal import Sealer, dump_json, load_json, not_sealable, seal
from agent_adapter.flow.actions import Call, Map
from agent_adapter.flow.graph import FlowGraph
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DurableRun:
    """One durable run's place in the store: the step it last saved and its call log."""

    store: CheckpointStore
    flow: str
    sealer: Sealer
    digest: str
    owned: bool
    ctx: ToolContext
    step: int  # the last stored step; -1 before the first save
    log: CallLog
    identity: Mapping[str, str]  # user, thread and principal, sealed into every checkpoint

    @property
    def run_id(self) -> str:
        return self.ctx.run_id

    async def save(
        self,
        cursor: Cursor,
        status: CheckpointStatus,
        checkpoint: FlowCheckpoint | None = None,
        *,
        carried: tuple[CallRecord, ...] = (),
        spend: Mapping[str, Any] | None = None,
    ) -> None:
        """Store the run's next checkpoint (signed) and start a new call-log segment.

        `carried` are the calls of a step that a recovery runs again (they replay from the
        new segment); `spend` is an approval decision stored before it is spent.

        Raises:
            CheckpointConflict: another worker saved this run's next step first.
            FlowError: `state-not-sealable`: the position has no canonical form.
        """
        if checkpoint is None:
            checkpoint = self.sealed(cursor)
        record = Checkpoint(
            kind="flow", name=self.flow, status=status, segment=new_segment(), data=""
        )
        record = seal_checkpoint(
            self.sealer.key,
            self.run_id,
            self.step + 1,
            record,
            dump_json(self.sealer, checkpoint),
            self.identity,
            carried=carried,
            spend=spend,
        )
        try:
            await self.store.save(self.run_id, self.step + 1, record)
        except BaseException:
            self.log.broken = True
            raise
        self.step += 1
        missed = self.log.move_to(record.segment, carried)
        extra: dict[str, Any] = {"unreplayed": missed} if missed else {}
        event = make_event(
            EventType.CHECKPOINT_SAVED,
            self.ctx,
            kind="flow",
            step=self.step,
            status=status,
            flow=self.flow,
            superstep=cursor.steps,
            **extra,
        )
        emit(self.ctx.sink, event)
        logger.debug("flow %s run %s: checkpoint %d saved", self.flow, self.run_id, self.step)

    async def commit(self, cursor: Cursor) -> None:
        """After a superstep: the next one starts from this checkpoint (the last: `finish`)."""
        if cursor.frontier:
            await self.save(cursor, "running")

    async def finish(self, cursor: Cursor, result: FlowResult[Any]) -> None:
        """The run ended (done or denied) or paused: no worker continues it on its own."""
        status: CheckpointStatus = result.status
        await self.save(cursor, status, result.checkpoint)

    async def failed(self, cursor: Cursor) -> None:
        """A node or route failed: mark the run failed, so a worker does not retry it.

        A run whose store write failed is left as it is: a worker can recover it later.
        """
        if self.log.broken:
            return
        try:
            await self.save(cursor, "failed")
        except Exception as exc:  # the store may be what failed; the run's error wins
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "flow %s: failed run not marked in the store",
                self.flow,
            )

    def sealed(self, cursor: Cursor) -> FlowCheckpoint:
        """The cursor as a sealed checkpoint with nothing pending.

        Raises:
            FlowError: `state-not-sealable`.
        """
        checkpoint = FlowCheckpoint(
            flow=self.flow,
            state=cursor.state,
            frontier=cursor.frontier,
            visits=frozen(cursor.visits),
            loops=frozen(cursor.loops),
            steps=cursor.steps,
            pending=None,
            decisions=frozen(cursor.granted),
            _writes=frozen({n: o.updates for n, o in cursor.writes.items()}),
            _owned=self.owned,
            _answers=frozen(cursor.answers),
            _items=frozen({n: frozen(dict(v)) for n, v in cursor.items.items()}),
            _journal=frozen(cursor.journal),
            _children=frozen(cursor.children),
            _graph=self.digest,
        )
        try:
            _check_journals(checkpoint)
            return dataclasses.replace(checkpoint, _mac=seal(self.sealer, checkpoint))
        except (NotSealable, ValueError) as exc:
            problem = exc if isinstance(exc, NotSealable) else NotSealable("$", "value", str(exc))
            raise not_sealable(self.flow, problem) from exc


@dataclasses.dataclass(frozen=True)
class Recovery:
    """What `FlowRunner.recover` found: the checkpoint, its envelope, the logged calls."""

    stored: StoredCheckpoint
    envelope: Envelope
    checkpoint: FlowCheckpoint
    calls: tuple[CallRecord, ...]  # the cut-short step's calls, every one signature checked


async def load_recovery(
    store: CheckpointStore, sealer: Sealer, graph: FlowGraph, run_id: str
) -> Recovery:
    """The run's latest checkpoint, checked: its kind, flow, MAC and graph.

    Raises:
        CheckpointNotFound: nothing is stored for `run_id`.
        CheckpointConflict: the checkpoint belongs to another flow or to a tool loop.
        CheckpointStoreCorrupt: a row or the signed document does not check out.
        FlowError: the flow's graph changed since the checkpoint was saved.
    """
    stored = await store.latest(run_id)
    if stored is None:
        raise CheckpointNotFound(run_id)
    saved = stored.checkpoint
    if saved.kind != "flow" or saved.name != graph.name:
        raise CheckpointConflict(
            run_id, f"the checkpoint is for {saved.kind} {saved.name!r}, not flow {graph.name!r}"
        )
    envelope = open_checkpoint(sealer.key, stored)  # a bad signature is corruption
    if not isinstance(envelope.body, str):
        raise corrupt(run_id, "the signed checkpoint holds no flow checkpoint")
    checkpoint = load_json(sealer, graph, envelope.body)  # graph changed: FlowError
    calls: list[CallRecord] = []
    if saved.status == "running":
        logged = await store.calls(run_id, saved.segment)
        current = verified_calls(sealer.key, run_id, saved.segment, logged)
        calls = merge_calls(envelope.carried, current)
    return Recovery(stored, envelope, checkpoint, tuple(calls))


def check_calls(
    run_id: str, graph: FlowGraph, registry: Registry | None, recovery: Recovery, *, rerun: bool
) -> int:
    """The number of in-doubt calls that will run again (each is allowed, or this raises).

    Raises:
        RecoveryNeedsDecision: a call may have run and its tool is not `idempotent=True`.
    """
    check = idempotent_in(registry, _graph_tools(graph))
    return len(check_in_doubt(run_id, recovery.calls, check, rerun=rerun))


def recovered_event(run: DurableRun, recovery: Recovery, rerun: int) -> None:
    """`run.recovered`: the step it continues from and how many calls replay."""
    replayed = sum(1 for call in recovery.calls if call.state == "done")
    event = make_event(
        EventType.RUN_RECOVERED,
        run.ctx,
        kind="flow",
        step=recovery.stored.step,
        replayed=replayed,
        rerun=rerun,
        flow=run.flow,
    )
    emit(run.ctx.sink, event)
    logger.info(
        "flow %s run %s recovered from step %d (%d calls replay, %d run again)",
        run.flow,
        run.run_id,
        recovery.stored.step,
        replayed,
        rerun,
    )


def ended(run_id: str, status: str) -> CheckpointConflict:
    """The error for recovering a run that already ended."""
    return CheckpointConflict(run_id, f"the run already ended ({status}); nothing to recover")


def paused_result(checkpoint: FlowCheckpoint) -> FlowResult[Any]:
    """A run that was paused when it stopped: resume it with `FlowRunner.resume`."""
    return FlowResult(
        state=checkpoint.state,
        status="paused",
        steps=checkpoint.steps,
        visits=checkpoint.visits,
        checkpoint=checkpoint,
        pending=checkpoint.pending,
    )


async def check_latest(
    store: CheckpointStore, key: bytes, checkpoint: FlowCheckpoint, run_id: str
) -> tuple[int, Mapping[str, str]]:
    """The stored step and identity of a paused checkpoint being resumed (the run's latest).

    Raises:
        CheckpointConflict: the store holds another (newer) position for the run.
        CheckpointStoreCorrupt: the stored checkpoint does not check out.
    """
    stored = await store.latest(run_id)
    if stored is None or stored.checkpoint.status != "paused":
        raise CheckpointConflict(run_id, "the store does not hold this run as paused")
    envelope = open_checkpoint(key, stored)
    body = envelope.body if isinstance(envelope.body, str) else ""
    if _mac_of(body) != checkpoint._mac:
        raise CheckpointConflict(run_id, "the checkpoint is not the run's latest one")
    return stored.step, envelope.identity


def _mac_of(text: str) -> str:
    try:
        document = json.loads(text)
        return str(document.get("mac", "")) if isinstance(document, dict) else ""
    except ValueError:
        return ""


def _graph_tools(graph: FlowGraph) -> Mapping[str, bool]:
    """`idempotent` of every `ToolSpec` a `Call` or `Map` names directly, sub-flows included."""
    found: dict[str, bool] = {}
    for spec in graph.nodes.values():
        action = spec.action
        tool = action.tool if isinstance(action, Call | Map) else None
        if isinstance(tool, ToolSpec):
            found[tool.name] = tool.idempotent is True
        if spec.child is not None:
            found.update(_graph_tools(spec.child))
    return found


def not_durable(flow: str) -> FlowError:
    """The error for `recover` on a runner without a checkpoint store."""
    return FlowError(f"flow {flow!r}: this FlowRunner has no checkpoints= store", flow=flow)
