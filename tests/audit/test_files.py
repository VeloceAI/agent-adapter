"""The folder plumbing: the head's atomic replace, the lock, and sealing at exit."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from agent_adapter.audit import FileAuditStore, _files, store, verify
from agent_adapter.audit._check import _parse
from agent_adapter.audit._files import FolderLock, Head, read_head, write_head
from agent_adapter.audit.store import _start_of, close_all
from agent_adapter.errors import ConfigError
from tests.audit.helpers import KEY, entry

HEAD = Head(seq=1, last="ab", index=0, segment="audit-20261106-000000.jsonl", sealed=False)


def test_the_head_replace_retries_a_briefly_busy_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real, calls = os.replace, {"n": 0}

    def busy(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(os, "replace", busy)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    write_head(tmp_path, KEY.encode(), HEAD, fsync=True)
    assert read_head(tmp_path, KEY.encode()) == HEAD


def test_a_head_that_stays_busy_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(src: str, dst: str) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(os, "replace", busy)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        write_head(tmp_path, KEY.encode(), HEAD, fsync=False)


def test_a_lock_releases_once_and_logs_a_failed_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    lock = FolderLock(tmp_path)
    lock.release()
    lock.release()
    other = FolderLock(tmp_path)

    def broken(handle: object) -> None:
        raise OSError("unlock failed")

    monkeypatch.setattr(_files, "_unlock", broken)
    with caplog.at_level(logging.WARNING):
        other.release()
    assert "audit lock release failed" in caplog.text
    FolderLock(tmp_path).release()  # closing the handle freed it


def test_open_stores_are_sealed_at_exit(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    good = FileAuditStore(tmp_path / "a", key=KEY, fsync=False)
    good.write(entry())
    bad = FileAuditStore(tmp_path / "b", key=KEY, fsync=False)

    def fail() -> None:
        raise OSError("disk gone")

    bad.close = fail  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        close_all()
    assert verify(tmp_path / "a", KEY).sealed
    assert "not sealed" in caplog.text
    store._OPEN.discard(bad)
    FileAuditStore.close(bad)


def test_damaged_first_records_and_nan_lines_are_not_records() -> None:
    with pytest.raises(ConfigError, match="damaged"):
        _start_of(b"{}")
    assert _parse(b'{"a":NaN}') is None
