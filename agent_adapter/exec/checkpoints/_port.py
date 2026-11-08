"""The checkpoint store port and its records: one run's latest checkpoint and its call log."""

from __future__ import annotations

import builtins
import json
import logging
from contextvars import ContextVar
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_adapter.core.errors import (
    CheckpointConflict,
    CheckpointStoreCorrupt,
    ConfigError,
    ToolUsageError,
    log_failure,
)

logger = logging.getLogger(__name__)

CheckpointKind = Literal["flow", "loop"]
CheckpointStatus = Literal["running", "paused", "done", "denied", "failed"]
CallState = Literal["started", "done", "pending", "unrecordable"]
FINISHED: frozenset[str] = frozenset({"done", "denied", "failed"})
_MAX_ID = 512  # characters of a run id, segment or call key

_FROZEN = ConfigDict(frozen=True, extra="forbid")
M = TypeVar("M", bound=BaseModel)


class Checkpoint(BaseModel):
    """One saved position of a run: what it is, its status, and the sealed document."""

    model_config = _FROZEN

    kind: CheckpointKind
    name: str  # the flow name, or the tool loop's stage
    status: CheckpointStatus
    segment: str  # the call log the run's next calls are written to
    data: str  # the sealed JSON document (MAC-checked by the runner, never by the store)
    saved_at: float = 0.0


class StoredCheckpoint(BaseModel):
    """A run's latest checkpoint and its step; the next save must use `step + 1`."""

    model_config = _FROZEN

    run_id: str
    step: int = Field(ge=0)
    checkpoint: Checkpoint


class CallRecord(BaseModel):
    """One tool call of a durable run: started, done (result kept), pending or unrecordable."""

    model_config = _FROZEN

    key: str  # the call's identity inside the run (visit, path, tool, argument digest, count)
    tool: str
    state: CallState
    result: dict[str, Any] | None = None  # the journaled result when `state == "done"`
    why: str = ""
    identity: list[Any] = Field(default_factory=list)  # what `key` digests: replay position
    seq: int = -1  # its place in the segment's log (0, 1, ...): a gap means a lost record
    mac: str = ""  # HMAC by the runner's checkpoint key over everything above


IDEMPOTENCY_KEY: ContextVar[str | None] = ContextVar("aa_idempotency_key", default=None)


def idempotency_key() -> str | None:
    """A key unique to the running tool call of a durable run, the same after a recovery.

    Pass it to an external API that accepts idempotency keys (payments, mail), so a call that
    runs again after a crash, or on a worker that was cut off, is not applied twice.
    """
    return IDEMPOTENCY_KEY.get()


@runtime_checkable
class CheckpointStore(Protocol):
    """Where durable runs keep their latest checkpoint and the tool calls of the current step.

    `save` is a compare-and-set on the step: step 0 only for a new run, else exactly one past
    the stored step; anything else raises `CheckpointConflict`, so two workers never both
    continue a run. A save that moves to a new segment drops the older segments' call logs.
    `record_call` refuses a segment that is not the run's current one (`CheckpointConflict`).
    """

    async def save(self, run_id: str, step: int, checkpoint: Checkpoint) -> None: ...

    async def latest(self, run_id: str) -> StoredCheckpoint | None: ...

    async def list_unfinished(
        self, *, name: str | None = None, limit: int = 100
    ) -> builtins.list[StoredCheckpoint]: ...

    async def delete(self, run_id: str) -> bool: ...

    async def record_call(self, run_id: str, segment: str, call: CallRecord) -> None: ...

    async def calls(self, run_id: str, segment: str) -> builtins.list[CallRecord]: ...


def check_id(value: object, what: str) -> str:
    """A run id, segment or call key: a non-empty str of bounded length.

    Raises:
        ToolUsageError: not a non-empty str, or longer than 512 characters.
    """
    if not isinstance(value, str) or not value or len(value) > _MAX_ID:
        raise ToolUsageError(f"{what} must be a non-empty str of at most {_MAX_ID} characters")
    return value


def check_save(run_id: object, step: object, checkpoint: object) -> None:
    """Validate `save` arguments before the backend is touched.

    Raises:
        ToolUsageError: a bad run id, step or checkpoint.
    """
    check_id(run_id, "run_id")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ToolUsageError(f"step must be an int >= 0, got {step!r}")
    if not isinstance(checkpoint, Checkpoint):
        raise ToolUsageError(f"checkpoint must be a Checkpoint, got {type(checkpoint).__name__}")
    check_id(checkpoint.segment, "segment")


def check_limit(limit: object) -> int:
    """A list limit.

    Raises:
        ConfigError: not an int >= 0.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ConfigError(f"limit must be an int >= 0, got {limit!r}")
    return limit


def conflict(run_id: str, step: int, found: int | None) -> CheckpointConflict:
    """The error for a save whose step is not the next one."""
    have = "no checkpoint" if found is None else f"step {found}"
    logger.warning("run %s: checkpoint step %d refused (store has %s)", run_id, step, have)
    return CheckpointConflict(
        run_id, f"checkpoint step {step} refused: the store has {have} (another worker?)"
    )


def stale_segment(run_id: str) -> CheckpointConflict:
    """The error for a call logged to a segment the run has moved past."""
    logger.warning("run %s: call log write to a stale segment refused", run_id)
    return CheckpointConflict(run_id, "the run moved on; this worker's call log is stale")


def encode_checkpoint(checkpoint: Checkpoint) -> str:
    """The stored JSON text of a checkpoint."""
    return checkpoint.model_dump_json()


def encode_call(call: CallRecord) -> str:
    """The stored JSON text of a call record.

    Raises:
        ToolUsageError: the result holds a value JSON cannot carry.
    """
    try:
        return json.dumps(call.model_dump(mode="json"), allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        log_failure(logger, logging.WARNING, exc, "call record of tool %s not stored", call.tool)
        raise ToolUsageError(f"call record of tool {call.tool!r} is not JSON") from exc


def decode_checkpoint(raw: object, where: str) -> Checkpoint:
    """A stored checkpoint.

    Raises:
        CheckpointStoreCorrupt: not text, not JSON, or not a `Checkpoint`.
    """
    return _decode(Checkpoint, raw, where)


def decode_call(raw: object, where: str) -> CallRecord:
    """A stored call record.

    Raises:
        CheckpointStoreCorrupt: not text, not JSON, or not a `CallRecord`.
    """
    return _decode(CallRecord, raw, where)


def _decode(model: type[M], raw: object, where: str) -> M:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            raise TypeError(f"stored row is {type(raw).__name__}, not text")
        return model.model_validate_json(raw)
    except (ValueError, TypeError, ValidationError, RecursionError) as exc:
        log_failure(logger, logging.ERROR, exc, "checkpoint store row %s is unreadable", where)
        raise CheckpointStoreCorrupt(f"checkpoint store row {where} is unreadable") from exc


def stored(run_id: object, step: object, raw: object, status: object = None) -> StoredCheckpoint:
    """A stored row as a `StoredCheckpoint`; its columns must agree with its record.

    Raises:
        CheckpointStoreCorrupt: unreadable, or a column that disagrees with the record.
    """
    run = _text(run_id)
    where = repr(run[:64]) if run is not None else f"<{type(run_id).__name__}>"
    checkpoint = decode_checkpoint(raw, where)
    number = _number(step)
    if run is None or number is None or (status is not None and _text(status) != checkpoint.status):
        logger.error("checkpoint store row %s disagrees with its record: failing closed", where)
        raise CheckpointStoreCorrupt(f"checkpoint store row {where} disagrees with its record")
    return StoredCheckpoint(run_id=run, step=number, checkpoint=checkpoint)


def _text(value: object) -> str | None:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "surrogatepass")
        except UnicodeDecodeError:
            return None
    return value if isinstance(value, str) else None


def _number(value: object) -> int | None:
    """A stored step (an int, or its digits as text or bytes); None for anything else."""
    if isinstance(value, bytes | str):
        text = _text(value)
        return int(text) if text is not None and text.isdigit() else None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
