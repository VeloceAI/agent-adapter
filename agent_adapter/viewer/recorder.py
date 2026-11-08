"""The run recorder: an EventSink that queues events for a background writer of JSONL files."""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
import time
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import Event
from agent_adapter.exec._settings import ToolPolicy
from agent_adapter.exec.defaults import add_default_sink, default_executor, remove_default_sink
from agent_adapter.exec.executor import Executor
from agent_adapter.runtime._fork import ForkSafeLock
from agent_adapter.tracing._content import ContentEvent, content_data
from agent_adapter.tracing.sinks import members
from agent_adapter.viewer._files import DEFAULT_FOLDER, append
from agent_adapter.viewer._segments import Options, Segments

logger = logging.getLogger(__name__)

_BATCH = 512  # items written per wake-up of the writer
_WARN_EVERY = 1_000  # after the first problem, warn once per this many
_STOP = object()
_LIVE: weakref.WeakSet[Recorder] = weakref.WeakSet()
RegistrySource = Registry | Callable[[], Registry] | None
PolicySource = Callable[[], ToolPolicy | None] | None


class Recorder:
    """Writes events under `folder`, one segment file per run and process (append-only).

    `emit` only queues: a background thread writes, in batches, so the event loop never waits
    on the disk. A full queue drops events (counted, logged). Queued events are written at
    `flush`, `close` and interpreter exit. Tool and LLM content keys are left out unless
    `capture_content`; secret-named values are always masked. Past `keep_runs` the oldest
    ended runs are deleted; a run stops recording after `max_events_per_run` events.

    Raises:
        ConfigError: a bad limit, or a folder that cannot be made.
    """

    def __init__(
        self,
        folder: str | os.PathLike[str] = DEFAULT_FOLDER,
        *,
        capture_content: bool = False,
        capture_confirm_content: bool = False,
        keep_runs: int = 200,
        max_events_per_run: int = 50_000,
        max_queue: int = 10_000,
        registry: RegistrySource = None,
        source: str = "",
        policy: PolicySource = None,
    ) -> None:
        limits = (
            ("keep_runs", keep_runs),
            ("max_events_per_run", max_events_per_run),
            ("max_queue", max_queue),
        )
        for label, value in limits:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"{label} must be an int >= 1, got {value!r}")
        self.folder = Path(folder).resolve()
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"cannot create the runs folder {self.folder}: {exc}") from exc
        self.capture_content = bool(capture_content)
        if capture_confirm_content and not capture_content:
            raise ConfigError("capture_confirm_content=True needs capture_content=True")
        self.capture_confirm_content = bool(capture_confirm_content)
        self.keep_runs = keep_runs
        self.max_events_per_run = max_events_per_run
        self.max_queue = max_queue
        self.source = str(source)
        self._registry = registry
        self._policy = policy  # the executor's policy, shown per tool in the Tools tab
        self._lock = ForkSafeLock()
        self._pid = -1
        self._queue: queue.Queue[Any] = queue.Queue(max_queue)
        self.dropped = 0
        self.failures = 0
        self.stopped = False
        self._executors: list[weakref.ref[Executor]] = []  # where record() attached it
        self._follows_default = False
        _LIVE.add(self)

    def emit(self, event: Event) -> None:
        if isinstance(event, ContentEvent):
            data = content_data(event, self)
            event = Event(event.type, event.run_id, event.call_id, event.parent_id, event.ts, data)
        self._put(event)

    def take_flow_graph(self, run_id: str, flow: str, graph: dict[str, Any]) -> None:
        """Write a flow's graph once into its run's file (FlowRunner calls this)."""
        self._put(("graph", run_id, flow, graph, time.time()))

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every queued event is written; False on timeout."""
        if self._pid != os.getpid():
            return True  # no writer in this process (none yet, or a forked child): nothing queued
        return _drained(self._queue, timeout)

    def close(self, timeout: float = 5.0) -> bool:
        """Write what is queued and stop the writer; later events start a new one."""
        with self._lock:
            if self._pid != os.getpid():
                return True
            self._pid = -1
            work = self._queue
        work.put(_STOP)  # blocks only while the queue is full
        return _drained(work, timeout)

    def stop(self, timeout: float = 5.0) -> bool:
        """Detach from every executor `record()` attached it to, write what is queued, and stop.

        Later events are ignored. False when the queue was not written within `timeout`.
        """
        with self._lock:  # no writer can start once this is set
            self.stopped = True
        if self._follows_default:
            remove_default_sink(self)
        for ref in self._executors:
            executor = ref()
            if executor is not None:
                executor.remove_sink(self)
        self._executors.clear()
        done = self.close(timeout)
        logger.info("stopped recording runs to %s", self.folder)
        return done

    @property
    def wants_content(self) -> bool:
        """Asks native calls for their content (arguments, results, messages, flow state)."""
        return self.capture_content

    @property
    def wants_confirm_content(self) -> bool:
        """Also asks for `confirm` tools' results (opt-in on top of `capture_content`)."""
        return self.capture_confirm_content

    def registry(self) -> Registry | None:
        """The registry whose catalogue each run file starts with."""
        source = self._registry
        return source() if callable(source) and not isinstance(source, Registry) else source

    def _put(self, item: Any) -> None:
        work = self._writer()
        if work is None:
            return  # stopped: dropped on purpose
        try:
            work.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            if self.dropped % _WARN_EVERY == 1:
                logger.warning(
                    "run recorder queue full (%d): %d event(s) dropped so far",
                    self.max_queue,
                    self.dropped,
                )

    def _writer(self) -> queue.Queue[Any] | None:
        """The writer's queue in this process (new after a fork), or None once stopped."""
        if self.stopped:
            return None
        pid = os.getpid()
        if self._pid == pid:
            return self._queue
        with self._lock:
            if self.stopped:  # stop() won the race: never start a writer after it
                return None
            if self._pid != pid:
                self._queue = queue.Queue(self.max_queue)  # a parent's queue may be locked
                options = Options(
                    self.folder,
                    self.capture_content,
                    self.keep_runs,
                    self.max_events_per_run,
                    self.registry,
                    self.source,
                    self._policy,
                )
                thread = threading.Thread(
                    target=self._run,
                    args=(self._queue, Segments(options)),
                    name=f"agent-adapter-recorder-{id(self):x}",
                    daemon=True,
                )
                thread.start()
                self._pid = pid
            return self._queue

    def _run(self, work: queue.Queue[Any], segments: Segments) -> None:
        while True:
            batch = [work.get()]
            while len(batch) < _BATCH:
                try:
                    batch.append(work.get_nowait())
                except queue.Empty:
                    break
            try:
                self._write(segments, [item for item in batch if item is not _STOP])
            finally:
                for _ in batch:
                    work.task_done()
            if any(item is _STOP for item in batch):
                return

    def _write(self, segments: Segments, items: list[Any]) -> None:
        out: dict[Path, list[bytes]] = {}
        for item in items:
            try:
                if isinstance(item, Event):
                    segments.event(item, out)
                else:
                    _, run_id, flow, graph, ts = item
                    segments.graph(run_id, flow, graph, ts, out)
            except Exception as exc:
                self._failed(exc, "cannot record an event")
        for path, lines in out.items():
            if not lines:
                continue
            try:
                append(path, lines)
            except Exception as exc:
                self._failed(exc, f"cannot write {path.name}")

    def _failed(self, exc: BaseException, what: str) -> None:
        # A recorder never breaks the traced call; failures are logged, rate-limited.
        self.failures += 1
        if self.failures % _WARN_EVERY == 1:
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "run recorder: %s; %d failure(s) so far",
                what,
                self.failures,
            )


def _drained(work: queue.Queue[Any], timeout: float) -> bool:
    """Wait until `work` has no unfinished items; False on timeout."""
    deadline = time.monotonic() + timeout
    with work.all_tasks_done:
        while work.unfinished_tasks:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            work.all_tasks_done.wait(left)
    return True


def flush_all(timeout: float = 5.0) -> bool:
    """Write every recorder's queued events (runs at interpreter exit)."""
    return all([recorder.flush(timeout) for recorder in list(_LIVE)])


def _after_fork_in_child() -> None:
    """A forked child has no writer thread: give each recorder a fresh, empty queue."""
    for recorder in list(_LIVE):
        recorder._pid = -1
        recorder._queue = queue.Queue(recorder.max_queue)


atexit.register(flush_all)
if hasattr(os, "register_at_fork"):  # POSIX only
    os.register_at_fork(after_in_child=_after_fork_in_child)


def record(
    executor: Executor | None = None,
    *,
    folder: str | os.PathLike[str] = DEFAULT_FOLDER,
    capture_content: bool = False,
    keep_runs: int = 200,
    source: str = "",
    capture_confirm_content: bool = False,
) -> Recorder:
    """Record every run of `executor` under `folder`.

    With no executor it follows the default executor, also after `aa.configure` replaces it.
    Calling it again for the same target and folder returns the recorder already attached;
    `recorder.stop()` detaches it again.

    Raises:
        ConfigError: a bad `keep_runs`, a folder that cannot be made, or a second call for
            the same target and folder with other options.
    """
    target = executor if executor is not None else default_executor()
    resolved = Path(folder).resolve()
    existing = next(
        (s for s in members(target.sink) if isinstance(s, Recorder) and s.folder == resolved),
        None,
    )
    if existing is not None:
        wanted = (capture_content, keep_runs, str(source), capture_confirm_content)
        have = (existing.capture_content, existing.keep_runs, existing.source)
        if (*have, existing.capture_confirm_content) != wanted:
            raise ConfigError(
                f"runs in {resolved} are already recorded with capture_content="
                f"{existing.capture_content}, keep_runs={existing.keep_runs}, "
                f"source={existing.source!r}"
            )
        return existing
    registry: RegistrySource = (
        target.registry if executor is not None else (lambda: default_executor().registry)
    )
    recorder = Recorder(
        resolved,
        capture_content=capture_content,
        capture_confirm_content=capture_confirm_content,
        keep_runs=keep_runs,
        registry=registry,
        source=source,
        policy=(lambda: target.policy) if executor is not None else _default_policy,
    )
    if executor is None:
        add_default_sink(recorder)
        recorder._follows_default = True
    else:
        executor.add_sink(recorder)
        recorder._executors.append(weakref.ref(executor))
    logger.info("recording runs to %s", resolved)
    return recorder


def _default_policy() -> ToolPolicy | None:
    """The default executor's policy, read at each catalogue (configure may replace it)."""
    return default_executor().policy
