"""MemoryCheckpointStore: checkpoints in this process, for tests and single-process apps."""

from __future__ import annotations

import builtins
import logging
import threading
import time

from agent_adapter.exec.checkpoints._port import (
    CallRecord,
    Checkpoint,
    StoredCheckpoint,
    check_id,
    check_limit,
    check_save,
    conflict,
    stale_segment,
)

logger = logging.getLogger(__name__)


class MemoryCheckpointStore:
    """A `CheckpointStore` in a dict; it survives a failed run, not a process exit."""

    def __init__(self) -> None:
        self._runs: dict[str, StoredCheckpoint] = {}
        self._calls: dict[tuple[str, str], dict[str, CallRecord]] = {}
        self._lock = threading.Lock()

    async def save(self, run_id: str, step: int, checkpoint: Checkpoint) -> None:
        """Store the run's next checkpoint (compare-and-set on `step`).

        Raises:
            CheckpointConflict: `step` is not 0 for a new run or one past the stored step.
            ToolUsageError: a bad run id, step or checkpoint.
        """
        check_save(run_id, step, checkpoint)
        if not checkpoint.saved_at:
            checkpoint = checkpoint.model_copy(update={"saved_at": time.time()})
        with self._lock:
            found = self._runs.get(run_id)
            have = None if found is None else found.step
            if step != (0 if have is None else have + 1):
                raise conflict(run_id, step, have)
            self._runs[run_id] = StoredCheckpoint(run_id=run_id, step=step, checkpoint=checkpoint)
            for run, segment in [k for k in self._calls if k[0] == run_id]:
                if segment != checkpoint.segment:
                    del self._calls[(run, segment)]

    async def latest(self, run_id: str) -> StoredCheckpoint | None:
        """The run's latest checkpoint, or None."""
        check_id(run_id, "run_id")
        with self._lock:
            return self._runs.get(run_id)

    async def list_unfinished(
        self, *, name: str | None = None, limit: int = 100
    ) -> builtins.list[StoredCheckpoint]:
        """Runs whose latest checkpoint is `running`, oldest first, optionally of one name.

        Raises:
            ConfigError: a negative `limit`.
        """
        check_limit(limit)
        with self._lock:
            found = [
                s
                for s in self._runs.values()
                if s.checkpoint.status == "running" and name in (None, s.checkpoint.name)
            ]
        found.sort(key=lambda s: (s.checkpoint.saved_at, s.run_id))
        return found[:limit]

    async def delete(self, run_id: str) -> bool:
        """Forget the run and its call log; False when nothing was stored."""
        check_id(run_id, "run_id")
        with self._lock:
            for k in [k for k in self._calls if k[0] == run_id]:
                del self._calls[k]
            return self._runs.pop(run_id, None) is not None

    async def record_call(self, run_id: str, segment: str, call: CallRecord) -> None:
        """Keep a call record in the run's current segment (latest record per key wins).

        Raises:
            CheckpointConflict: `segment` is not the run's current one.
        """
        check_id(run_id, "run_id")
        check_id(segment, "segment")
        check_id(call.key, "call key")
        with self._lock:
            found = self._runs.get(run_id)
            if found is None or found.checkpoint.segment != segment:
                raise stale_segment(run_id)
            self._calls.setdefault((run_id, segment), {})[call.key] = call

    async def calls(self, run_id: str, segment: str) -> builtins.list[CallRecord]:
        """The call records of one segment, in the order first written."""
        check_id(run_id, "run_id")
        check_id(segment, "segment")
        with self._lock:
            return list(self._calls.get((run_id, segment), {}).values())
