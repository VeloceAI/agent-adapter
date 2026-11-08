"""verify, read_entries and export_to read one line at a time and never load a whole log."""

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.audit import FileAuditStore, export_to, read_entries, verify
from agent_adapter.audit._files import segments
from agent_adapter.errors import ConfigError
from tests.audit.helpers import KEY, entry

verify_module = importlib.import_module("agent_adapter.audit.verify")


def big_log(folder: Path, count: int = 300) -> None:
    store = FileAuditStore(folder, key=KEY, max_bytes=4096, fsync=False)
    for n in range(count):
        store.write(entry(ts=1_800_000_000.0 + n, note="x" * 40))
    store.close()


def test_verify_never_reads_a_whole_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    big_log(tmp_path)
    real = Path.read_bytes

    def no_whole_segments(self: Path) -> bytes:
        assert self.suffix != ".jsonl", f"{self.name} read whole"
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", no_whole_segments)
    result = verify(tmp_path, KEY)
    assert result.ok and result.events == 300 and result.segments > 10


def test_read_entries_streams_every_event_in_order(tmp_path: Path) -> None:
    big_log(tmp_path)
    seqs = [record["seq"] for record in read_entries(tmp_path, KEY)]
    assert len(seqs) == 300 and seqs == sorted(seqs)


def test_read_entries_refuses_a_log_that_is_not_intact(tmp_path: Path) -> None:
    big_log(tmp_path, 3)
    ((_, path),) = segments(tmp_path)
    path.write_bytes(path.read_bytes().replace(b'"note":"x', b'"note":"y', 1))
    with pytest.raises(ConfigError, match="not intact"):
        next(read_entries(tmp_path, KEY))


def test_a_log_changed_between_the_check_and_the_read_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big_log(tmp_path, 3)
    ((_, path),) = segments(tmp_path)
    real_walk = verify_module.walk

    def walk_then_tamper(*args: Any, **kw: Any) -> Any:
        result = real_walk(*args, **kw)
        path.write_bytes(path.read_bytes().replace(b'"note":"x', b'"note":"y', 1))
        return result

    monkeypatch.setattr(verify_module, "walk", walk_then_tamper)
    with pytest.raises(ConfigError, match="changed while it was read"):
        list(read_entries(tmp_path, KEY))


def test_export_to_writes_as_it_reads(tmp_path: Path) -> None:
    big_log(tmp_path)
    out = io.StringIO()
    assert export_to(out, tmp_path, since=1_800_000_100, key=KEY) == 200
    first = json.loads(out.getvalue().splitlines()[0])
    assert first["ts"] == 1_800_000_100.0


def test_an_empty_window_still_has_a_csv_header(tmp_path: Path) -> None:
    big_log(tmp_path, 2)
    out = io.StringIO()
    assert export_to(out, tmp_path, since="2100-01-01", format="csv", key=KEY) == 0
    assert out.getvalue() == "seq,time,action,run_id,call_id,data,prev,mac\n"
