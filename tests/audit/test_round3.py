"""Review fixes: head writes, live logs, anchors, streams, the writer thread, strict gates."""

from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.audit import (
    AuditSink,
    AuditUnavailable,
    CallableAuditStore,
    FileAuditStore,
    audit,
    verify,
    verify_records,
)
from agent_adapter.audit import sink as sink_module
from agent_adapter.audit import store as store_module
from agent_adapter.audit._actions import AuditEntry
from agent_adapter.audit._files import HEAD, segments
from agent_adapter.core.types import Event
from agent_adapter.errors import ConfigError, FlowError
from agent_adapter.memory import MemoryScope, SQLiteMemory
from tests.audit.helpers import KEY, entry, write_log
from tests.integrations.durable.kit import RAN, durable

REQUESTED = Event("approval.requested", data={"tool": "t", "approval_id": "a"})


def flaky_head(monkeypatch: pytest.MonkeyPatch, fail: set[int]) -> None:
    real, calls = store_module.write_head, {"n": 0}  # type: ignore[attr-defined]

    def head(*args: Any, **kw: Any) -> None:
        calls["n"] += 1
        if calls["n"] in fail:
            raise PermissionError("a reader holds the head")
        real(*args, **kw)

    monkeypatch.setattr(store_module, "write_head", head)


def test_a_failed_head_write_keeps_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flaky_head(monkeypatch, {2})  # open ok, first event's head fails
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    store.write(entry())  # does not raise: the line is stored
    store.write(entry())
    store.close()
    result = verify(tmp_path, KEY)
    assert result.ok and result.events == 2


def test_a_failed_head_write_at_the_seal_still_rotates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileAuditStore(tmp_path, key=KEY, max_bytes=1024, fsync=False)
    flaky_head(monkeypatch, {4})  # open, event, event... then the seal's head
    for _ in range(8):
        store.write(entry(note="x" * 300))
    store.close()
    result = verify(tmp_path, KEY)
    assert result.ok and result.events == 8 and result.segments > 1


def test_a_live_writer_ahead_of_its_head_verifies(tmp_path: Path) -> None:
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    store.write(entry())
    old_head = (tmp_path / HEAD).read_bytes()
    store.write(entry())
    (tmp_path / HEAD).write_bytes(old_head)  # verify read the head before this write
    ((_, path),) = segments(tmp_path)
    with path.open("ab") as tail:
        tail.write(b'{"kind":"ev')  # a line being written right now
    assert verify(tmp_path, KEY).ok
    store._folder_lock.release()
    assert store._file is not None
    store._file.close()


def test_an_anchor_catches_a_rollback_and_a_swapped_folder(tmp_path: Path) -> None:
    log = tmp_path / "log"
    write_log(log, 2)
    snapshot = tmp_path / "snapshot"
    shutil.copytree(log, snapshot)
    store = FileAuditStore(log, key=KEY, fsync=False)
    store.write(entry())
    store.close()
    anchor = verify(log, KEY).anchor
    assert verify(log, KEY, anchor=anchor).ok
    shutil.rmtree(log)
    shutil.copytree(snapshot, log)  # an older, genuine state
    assert verify(log, KEY).ok  # alone, a rollback looks intact
    rolled = verify(log, KEY, anchor=anchor)
    assert any("rolled back" in p for p in rolled.problems)
    other = tmp_path / "other"
    write_log(other, 3)
    assert any("another log" in p for p in verify(other, KEY, anchor=anchor).problems)
    with pytest.raises(ConfigError, match="anchor"):
        verify(log, KEY, anchor="nope")


def test_a_segment_from_another_log_shows(tmp_path: Path) -> None:
    write_log(tmp_path / "a", 1)
    write_log(tmp_path / "b", 1)
    ((_, a),) = segments(tmp_path / "a")
    ((_, b),) = segments(tmp_path / "b")
    a.write_bytes(b.read_bytes())
    problems = verify(tmp_path / "a", KEY).problems
    assert any("another log" in p for p in problems)


def test_streams_may_interleave_and_each_is_checked() -> None:
    sent: list[dict[str, Any]] = []
    one, two = CallableAuditStore(sent.append, key=KEY), CallableAuditStore(sent.append, key=KEY)
    one.write(entry())
    two.write(entry())
    one.write(entry())
    one.close()
    two.close()
    result = verify_records(sent, KEY)
    assert result.ok and result.segments == 2 and result.events == 3
    bad_seal = [dict(r) for r in sent]
    bad_seal[-1] = {**bad_seal[-1], "events": 9}
    assert any(
        "MAC" in p or "seal does not match" in p for p in verify_records(bad_seal, KEY).problems
    )
    after = [*sent, sent[1]]
    assert not verify_records(after, KEY).ok


def test_writes_happen_on_the_writer_thread() -> None:
    threads: list[str] = []

    class Spy:
        def write(self, entry: AuditEntry) -> None:
            threads.append(threading.current_thread().name)

        def close(self) -> None: ...

        def digest(self, value: object) -> str:
            return "d"

    sink = AuditSink(Spy())
    sink.emit(REQUESTED)
    assert sink.flush()
    sink.stop()
    assert threads and threads[0].startswith("agent-adapter-audit")
    assert threads[0] != threading.current_thread().name and sink.written == 1


def test_a_full_queue_loses_records_loudly() -> None:
    gate = threading.Event()

    class Slow:
        def write(self, entry: AuditEntry) -> None:
            gate.wait(5)

        def close(self) -> None: ...

        def digest(self, value: object) -> str:
            return "d"

    sink = AuditSink(Slow(), fail_closed=True, max_queue=1)
    for _ in range(4):
        sink.emit(REQUESTED)
    assert sink.failures >= 2 and not sink.settled()
    gate.set()
    with pytest.raises(AuditUnavailable):
        sink.require()
    sink.stop()
    with pytest.raises(ConfigError):
        AuditSink(Slow(), max_queue=0)


def test_strict_mode_waits_for_the_disk_and_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()

    class Stuck:
        def write(self, entry: AuditEntry) -> None:
            gate.wait(5)

        def close(self) -> None: ...

        def digest(self, value: object) -> str:
            return "d"

    monkeypatch.setattr(sink_module, "REQUIRE_TIMEOUT_S", 0.05)
    sink = AuditSink(Stuck(), fail_closed=True)
    sink.emit(REQUESTED)
    with pytest.raises(AuditUnavailable, match="not stored in time"):
        sink.require()
    gate.set()
    assert sink.flush() and sink.settled()
    sink.require()  # stored after all: nothing lost
    sink.stop()


def test_strict_mode_stops_a_memory_export(tmp_path: Path) -> None:
    from agent_adapter import Executor, Registry

    executor = Executor(Registry())
    audit(executor, store=CallableAuditStore(_fail, key=KEY), fail_closed=True)
    memory = SQLiteMemory(tmp_path / "m.db")
    ctx = executor.context(run_id="run_m")
    scope = MemoryScope(user_id="ann")
    asyncio.run(memory.export(scope, ctx=ctx))  # its record is the one that is lost
    with pytest.raises(AuditUnavailable):
        asyncio.run(memory.export(scope, ctx=ctx))
    with pytest.raises(AuditUnavailable):
        asyncio.run(memory.delete(scope=scope, ctx=ctx))


def _fail(record: dict[str, Any]) -> None:
    raise RuntimeError("collector down")


async def test_strict_mode_blocks_calls_inside_a_durable_flow() -> None:
    flows, _ = durable()
    sink = AuditSink(CallableAuditStore(_fail, key=KEY), fail_closed=True)
    flows.runner.executor.add_sink(sink)
    sink.emit(REQUESTED)  # a lost record
    assert sink.flush()
    try:
        report = await flows.advance("orders", "r_strict", {"order": "o-1"})
        assert report.status == "failed"
    except FlowError as exc:  # the node failed with the audit error: nothing was charged
        assert "audit-unavailable" in str(exc)
    assert RAN["charge"] == 0
    sink.stop()
