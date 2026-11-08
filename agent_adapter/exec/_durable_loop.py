"""Crash-safe tool loops: a signed checkpoint of the history after each step, and recovery."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import CheckpointConflict, CheckpointNotFound, ToolUsageError
from agent_adapter.core.messages import Message
from agent_adapter.core.registry import Registry
from agent_adapter.exec._durable import CallLog, check_in_doubt, idempotent_in, new_segment
from agent_adapter.exec._durable_seal import (
    check_identity,
    corrupt,
    identity_of,
    merge_calls,
    open_checkpoint,
    seal_checkpoint,
    verified_calls,
)
from agent_adapter.exec._journal import _make_journal
from agent_adapter.exec._resume import _visit_scope, current_journal
from agent_adapter.exec.checkpoints import (
    CallRecord,
    Checkpoint,
    CheckpointStatus,
    CheckpointStore,
    StoredCheckpoint,
)
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)


@dataclass
class LoopCheckpoints:
    """A durable tool loop's place in the store: its last step, key and call log."""

    store: CheckpointStore
    key: bytes
    stage: str
    ctx: ToolContext
    step: int  # the last stored step; -1 before the first save
    log: CallLog
    unsaved: bool = True  # the log's segment is not stored yet: save before any call
    carried: tuple[CallRecord, ...] = ()  # a recovered turn's calls, for the claim save

    @property
    def run_id(self) -> str:
        return self.ctx.run_id

    async def save(self, history: Sequence[Message], steps: int, status: CheckpointStatus) -> None:
        """Store the loop's next checkpoint (signed) and start a new call-log segment.

        Raises:
            CheckpointConflict: another worker saved this run's next step first.
        """
        payload = {
            "stage": self.stage,
            "steps": steps,
            "messages": [m.model_dump(mode="json") for m in history],
        }
        record = Checkpoint(
            kind="loop", name=self.stage, status=status, segment=new_segment(), data=""
        )
        carried, self.carried = self.carried, ()
        record = seal_checkpoint(
            self.key,
            self.run_id,
            self.step + 1,
            record,
            payload,
            identity_of(self.ctx),
            carried=carried,
        )
        try:
            await self.store.save(self.run_id, self.step + 1, record)
        except BaseException:
            self.log.broken = True
            raise
        self.step += 1
        self.unsaved = False
        missed = self.log.move_to(record.segment, carried)
        extra: dict[str, Any] = {"unreplayed": missed} if missed else {}
        event = make_event(
            EventType.CHECKPOINT_SAVED,
            self.ctx,
            kind="loop",
            step=self.step,
            status=status,
            stage=self.stage,
            **extra,
        )
        emit(self.ctx.sink, event)

    @contextmanager
    def calls(self, index: int) -> Iterator[None]:
        """Log (and replay) the tool calls of the assistant turn at `index` of the history."""
        root = f"{self.stage}:turn:{index}"
        journal = _make_journal(self.log.seed(root), observer=self.log, path=(root,))
        with _visit_scope(None, journal):
            yield


@dataclass(frozen=True)
class Saved:
    """A loop checkpoint read back: where it stopped, its identity and the logged calls."""

    stored: StoredCheckpoint
    messages: list[Message]
    steps: int
    calls: tuple[CallRecord, ...]
    identity: Mapping[str, str]


async def begin(
    store: CheckpointStore, key: bytes, stage: str, ctx: ToolContext
) -> LoopCheckpoints:
    """The checkpoints of a new loop run, or of a paused one being resumed.

    Raises:
        ToolUsageError: inside a flow node (the flow's own checkpoints cover the loop), or a
            paused run of another user, thread or principal.
        CheckpointConflict: the run is running elsewhere (use `recover_tool_loop`) or ended.
        CheckpointStoreCorrupt: the paused checkpoint does not check out.
    """
    if current_journal() is not None:
        raise ToolUsageError(
            "run_tool_loop(checkpoints=) inside a flow node: the FlowRunner's checkpoints "
            "already cover its tool calls"
        )
    found = await store.latest(ctx.run_id)
    step = -1
    if found is not None:
        status = found.checkpoint.status
        if found.checkpoint.kind != "loop" or status != "paused":
            raise CheckpointConflict(
                ctx.run_id, f"the store holds this run as {status}; use recover_tool_loop"
            )
        check_identity(ctx.run_id, open_checkpoint(key, found).identity, ctx)
        step = found.step
    log = CallLog(store, ctx.run_id, new_segment(), key)
    return LoopCheckpoints(store, key, stage, ctx, step, log)


async def load(store: CheckpointStore, key: bytes, run_id: str, stage: str) -> Saved:
    """A crashed loop's latest checkpoint and the calls of its cut-short turn, all signed.

    Raises:
        CheckpointNotFound: nothing is stored for `run_id`.
        CheckpointConflict: the checkpoint belongs to a flow or another stage, the run ended,
            or it is paused (resume it through `run_tool_loop`).
        CheckpointStoreCorrupt: the row, its envelope or a logged call does not check out.
    """
    stored = await store.latest(run_id)
    if stored is None:
        raise CheckpointNotFound(run_id)
    saved = stored.checkpoint
    if saved.kind != "loop" or saved.name != stage:
        raise CheckpointConflict(
            run_id, f"the checkpoint is for {saved.kind} {saved.name!r}, not tool loop {stage!r}"
        )
    if saved.status == "paused":
        raise CheckpointConflict(run_id, "the loop is paused: resume it with run_tool_loop")
    if saved.status != "running":
        raise CheckpointConflict(run_id, f"the run already ended ({saved.status})")
    envelope = open_checkpoint(key, stored)
    payload = envelope.body
    try:
        if not isinstance(payload, dict) or payload.get("stage") != stage:
            raise ValueError("the document is for another stage")
        messages = [Message.model_validate(m) for m in payload["messages"]]
        steps = payload["steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps is not an int >= 0")
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise corrupt(run_id, "the tool loop checkpoint is unreadable") from exc
    logged = await store.calls(run_id, saved.segment)
    calls = merge_calls(envelope.carried, verified_calls(key, run_id, saved.segment, logged))
    return Saved(stored, messages, steps, tuple(calls), envelope.identity)


async def claim(
    saved: Saved,
    store: CheckpointStore,
    key: bytes,
    ctx: ToolContext,
    registry: Registry,
    *,
    rerun: bool,
) -> LoopCheckpoints:
    """Take over a crashed loop: check in-doubt calls, then win the next step (a new segment).

    Raises:
        RecoveryNeedsDecision: a call may have run and its tool is not `idempotent=True`.
        CheckpointConflict: another worker took the run first.
    """
    doubtful = check_in_doubt(ctx.run_id, saved.calls, idempotent_in(registry), rerun=rerun)
    record = saved.stored.checkpoint
    log = CallLog(store, ctx.run_id, new_segment(), key)
    loop = LoopCheckpoints(
        store, key, record.name, ctx, saved.stored.step, log, carried=saved.calls
    )
    await loop.save(saved.messages, saved.steps, "running")  # a cut-off worker is fenced now
    replayed = sum(1 for call in saved.calls if call.state == "done")
    event = make_event(
        EventType.RUN_RECOVERED,
        ctx,
        kind="loop",
        step=saved.stored.step,
        replayed=replayed,
        rerun=len(doubtful),
        stage=record.name,
    )
    emit(ctx.sink, event)
    logger.info(
        "tool loop run %s recovered from step %d (%d calls replay)",
        ctx.run_id,
        saved.stored.step,
        replayed,
    )
    return loop
