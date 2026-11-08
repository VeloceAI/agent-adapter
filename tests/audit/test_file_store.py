"""FileAuditStore: segments, rotation, seals, resume after a restart or a crash, one writer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from agent_adapter.audit import FileAuditStore, verify
from agent_adapter.audit._files import HEAD, segments
from agent_adapter.errors import ConfigError, IntegrationError, ToolUsageError
from tests.audit.helpers import KEY, entry, write_log


def lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def crash(store: FileAuditStore) -> None:
    """Leave the store as a killed process would: no seal, lock gone."""
    store._folder_lock.release()
    assert store._file is not None
    store._file.close()


def test_a_closed_log_is_one_sealed_segment_that_verifies(tmp_path: Path) -> None:
    write_log(tmp_path, 3)
    ((index, path),) = segments(tmp_path)
    kinds = [record["kind"] for record in lines(path)]
    assert (index, kinds) == (0, ["open", "event", "event", "event", "seal"])
    result = verify(tmp_path, KEY)
    assert result.ok and result.sealed and (result.events, result.records) == (3, 5)
    assert "intact" in result.summary()


def test_nothing_is_written_until_the_first_entry(tmp_path: Path) -> None:
    FileAuditStore(tmp_path, key=KEY, fsync=False).close()
    assert segments(tmp_path) == [] and not (tmp_path / HEAD).exists()


def test_segments_rotate_by_size(tmp_path: Path) -> None:
    store = FileAuditStore(tmp_path, key=KEY, max_bytes=1024, fsync=False)
    for n in range(12):
        store.write(entry(ts=float(n), note="x" * 150))
    store.close()
    assert len(segments(tmp_path)) > 1
    result = verify(tmp_path, KEY)
    assert result.ok and result.events == 12


def test_segments_rotate_at_a_new_utc_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1_800_000_000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    store.write(entry())
    now[0] += 86_400
    store.write(entry())
    store.close()
    names = [path.name for _, path in segments(tmp_path)]
    assert len(names) == 2 and names[0].split("-")[1] != names[1].split("-")[1]
    assert verify(tmp_path, KEY).ok


def test_a_reopened_folder_continues_the_chain(tmp_path: Path) -> None:
    write_log(tmp_path, 2)
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)  # last segment sealed: a new one
    store.write(entry())
    store.close()
    assert len(segments(tmp_path)) == 2
    result = verify(tmp_path, KEY)
    assert result.ok and result.events == 3


def test_an_unsealed_segment_is_continued_after_a_crash(tmp_path: Path) -> None:
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    store.write(entry())
    crash(store)
    again = FileAuditStore(tmp_path, key=KEY, fsync=False)
    again.write(entry())
    again.close()
    assert len(segments(tmp_path)) == 1
    assert verify(tmp_path, KEY).ok


def test_a_head_that_lags_one_signed_line_is_repaired(tmp_path: Path) -> None:
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    store.write(entry())
    stale = (tmp_path / HEAD).read_bytes()
    store.write(entry())
    crash(store)
    (tmp_path / HEAD).write_bytes(stale)  # a crash between the line and its head
    FileAuditStore(tmp_path, key=KEY, fsync=False).close()
    result = verify(tmp_path, KEY)
    assert result.ok and result.events == 2


def test_another_key_cannot_append(tmp_path: Path) -> None:
    write_log(tmp_path)
    with pytest.raises(ConfigError, match="another key"):
        FileAuditStore(tmp_path, key="a-different-key-0123456789", fsync=False)


@pytest.mark.parametrize("damage", ["cut", "partial", "head_gone", "segment_gone", "edited"])
def test_a_damaged_folder_is_refused(tmp_path: Path, damage: str) -> None:
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    store.write(entry())
    store.write(entry())
    crash(store)
    ((_, path),) = segments(tmp_path)
    raw = path.read_bytes()
    if damage == "cut":
        path.write_bytes(raw.split(b"\n")[0] + b"\n")
    elif damage == "partial":
        path.write_bytes(raw + b'{"kind":"ev')
    elif damage == "head_gone":
        (tmp_path / HEAD).unlink()
    elif damage == "segment_gone":
        path.unlink()
    else:
        path.write_bytes(raw.replace(b"refund", b"refunx"))
    with pytest.raises(ConfigError, match="verify"):
        FileAuditStore(tmp_path, key=KEY, fsync=False)


def test_one_writer_per_folder(tmp_path: Path) -> None:
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    with pytest.raises(ConfigError, match="another writer"):
        FileAuditStore(tmp_path, key=KEY, fsync=False)
    store.close()
    FileAuditStore(tmp_path, key=KEY, fsync=False).close()  # free again


def test_writes_after_close_are_refused_and_close_is_idempotent(tmp_path: Path) -> None:
    store = write_log(tmp_path, 1)
    store.close()
    with pytest.raises(ToolUsageError):
        store.write(entry())


@pytest.mark.parametrize("bad", [0, 1023, True, "big"])
def test_bad_max_bytes_is_refused(tmp_path: Path, bad: object) -> None:
    with pytest.raises(ConfigError):
        FileAuditStore(tmp_path, key=KEY, max_bytes=bad)  # type: ignore[arg-type]


def test_a_folder_that_cannot_be_made_is_refused(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ConfigError, match="cannot create"):
        FileAuditStore(blocker / "audit", key=KEY)


def test_a_failed_write_is_rolled_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FileAuditStore(tmp_path, key=KEY, fsync=True)
    store.write(entry())

    def boom(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        store.write(entry())
    monkeypatch.undo()
    store.write(entry())
    store.close()
    result = verify(tmp_path, KEY)
    assert result.ok and result.events == 2


def test_a_write_that_cannot_be_rolled_back_stops_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    store.write(entry())
    handle = store._file
    assert handle is not None

    def boom(*_: object) -> None:
        raise OSError("gone")

    monkeypatch.setattr(store, "_file", _Failing(handle, boom))
    monkeypatch.setattr(os, "truncate", boom)
    with pytest.raises(OSError):
        store.write(entry())
    with pytest.raises(IntegrationError):
        store.write(entry())
    store.close()  # a broken segment gets no seal
    assert verify(tmp_path, KEY).sealed is False


class _Failing:
    """A file whose writes and truncates fail."""

    def __init__(self, inner: object, fail: object) -> None:
        self.inner = inner
        self.write = fail

    def close(self) -> None:
        self.inner.close()  # type: ignore[attr-defined]
        raise OSError("close failed too")


def test_digest_uses_the_store_key(tmp_path: Path) -> None:
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    other = FileAuditStore(tmp_path / "b", key="another-key-0123456789", fsync=False)
    assert store.digest("ann") != other.digest("ann")
    assert len(store.key_id) == 16
    store.close()
    other.close()
