"""The audit sink: keeps security-relevant events, signed, and `audit(executor)` to attach it."""

from __future__ import annotations

import atexit
import logging
import os
import threading
import weakref
from pathlib import Path

from agent_adapter.audit._actions import AuditEntry, Mapper
from agent_adapter.audit._writer import DEFAULT_QUEUE, Writer
from agent_adapter.audit.errors import AuditUnavailable
from agent_adapter.audit.store import DEFAULT_MAX_BYTES, AuditStore, FileAuditStore
from agent_adapter.core.errors import ConfigError, ToolUsageError, log_failure
from agent_adapter.core.types import Event
from agent_adapter.exec.defaults import add_default_sink, default_executor, remove_default_sink
from agent_adapter.exec.executor import Executor
from agent_adapter.tracing.sinks import members

logger = logging.getLogger(__name__)

DEFAULT_FOLDER = Path(".agent_adapter") / "audit"
REQUIRE_TIMEOUT_S = 10.0  # strict mode: how long a call waits for its records to be stored
_WARN_EVERY = 100  # after the first failure, log once per this many
_LIVE: weakref.WeakSet[AuditSink] = weakref.WeakSet()


class AuditSink:
    """Queues security-relevant events for `store` on a writer thread; ignores all others.

    Raises:
        ConfigError: `store` is not an AuditStore, or a bad `max_queue`.
    """

    def __init__(
        self, store: AuditStore, *, fail_closed: bool = False, max_queue: int = DEFAULT_QUEUE
    ) -> None:
        if not isinstance(store, AuditStore):
            raise ConfigError(f"AuditSink needs an AuditStore, got {type(store).__name__}")
        if isinstance(max_queue, bool) or not isinstance(max_queue, int) or max_queue < 1:
            raise ConfigError(f"max_queue must be an int >= 1, got {max_queue!r}")
        self.store = store
        self.fail_closed = bool(fail_closed)
        self._mapper = Mapper(store.digest)
        self._writer = Writer(self._store_one, max_queue=max_queue)
        self.written = 0
        self.failures = 0
        self.stopped = False
        self._lock = threading.Lock()
        self._executors: list[weakref.ref[Executor]] = []
        self._follows_default = False
        _LIVE.add(self)

    @property
    def healthy(self) -> bool:
        """True while no record was lost."""
        return self.failures == 0

    def settled(self) -> bool:
        """True when `require` would return at once: lenient, stopped, or nothing queued."""
        return not self.fail_closed or self.stopped or self._writer.pending == 0

    def require(self) -> None:
        """Strict mode: wait until queued records are stored; raise if any was lost."""
        if not self.fail_closed or self.stopped:
            return
        if not self._writer.drain(REQUIRE_TIMEOUT_S):
            raise AuditUnavailable(self.failures, waiting=self._writer.pending)
        if self.failures:
            raise AuditUnavailable(self.failures)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every queued record is stored; False on timeout."""
        return self._writer.drain(timeout)

    def emit(self, event: Event) -> None:
        if self.stopped:
            return
        try:
            entries = self._mapper.entries(event)
        except Exception as exc:
            self._lost(exc, event.type)
            return
        for entry in entries:
            if not self._writer.put(entry):
                self._lost(OverflowError("audit queue full"), event.type)

    def stop(self, timeout: float = 10.0) -> None:
        """Detach, store the queue, then seal; a failure is logged and counted, not raised."""
        self.stopped = True
        if self._follows_default:
            remove_default_sink(self)
        for ref in self._executors:
            executor = ref()
            if executor is not None:
                executor.remove_sink(self)
        self._executors.clear()
        if not self._writer.stop(timeout):
            self._lost(TimeoutError("audit writer did not finish"), "stop")
        try:
            self.store.close()
        except Exception as exc:
            self._lost(exc, "seal")

    def _store_one(self, entry: AuditEntry) -> None:
        try:
            self.store.write(entry)
        except Exception as exc:
            self._lost(exc, entry.action)
            return
        with self._lock:
            self.written += 1

    def _lost(self, exc: BaseException, what: str) -> None:
        # A sink never breaks the traced call; a lost audit record is loud, not silent.
        with self._lock:
            self.failures += 1
            count = self.failures
        if count % _WARN_EVERY == 1:
            log_failure(
                logger, logging.ERROR, exc, "audit record lost (%s); %d so far", what, count
            )


def _stop_all() -> None:
    """Store every sink's queue and seal its log at interpreter exit."""
    for sink in list(_LIVE):
        if not sink.stopped:
            sink.stop(timeout=5.0)


atexit.register(_stop_all)


def audit(
    executor: Executor | None = None,
    *,
    folder: str | os.PathLike[str] = DEFAULT_FOLDER,
    key: str | bytes | None = None,
    store: AuditStore | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    fsync: bool = True,
    fail_closed: bool = False,
) -> AuditSink:
    """Attach a signed audit log of `executor` (or the default one) under `folder` or `store`.

    Raises:
        ConfigError: no key or a short one, a bad limit, another writer on the folder, or the
            folder already audited in the other `fail_closed` mode.
    """
    target = executor if executor is not None else default_executor()
    if store is None:
        resolved = Path(folder).resolve()
        for member in members(target.sink):
            if isinstance(member, AuditSink) and getattr(member.store, "folder", None) == resolved:
                if member.fail_closed != bool(fail_closed):
                    raise ConfigError(
                        f"{resolved} is already audited with fail_closed={member.fail_closed}"
                    )
                return member
        store = FileAuditStore(resolved, key=key, max_bytes=max_bytes, fsync=fsync)
    elif not isinstance(store, AuditStore):
        raise ConfigError(f"store must be an AuditStore, got {type(store).__name__}")
    sink = AuditSink(store, fail_closed=fail_closed)
    if executor is None:
        add_default_sink(sink)
        sink._follows_default = True
    else:
        try:
            executor.add_sink(sink)
        except ToolUsageError:
            store.close()
            raise
        sink._executors.append(weakref.ref(executor))
    logger.info("audit log attached (%s)", type(store).__name__)
    return sink
