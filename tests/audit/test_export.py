"""export: verified records only, a time window, JSONL kept byte for byte, safe CSV."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from agent_adapter.audit import FileAuditStore, export
from agent_adapter.audit._chain import check_mac
from agent_adapter.audit._files import segments
from agent_adapter.audit.export import moment
from agent_adapter.errors import ConfigError
from tests.audit.helpers import KEY, entry

DAY = 1_800_000_000.0


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    store.write(entry(ts=DAY, tool="refund"))
    store.write(entry(ts=DAY + 10, tool="=HYPERLINK(1)"))
    store.write(entry(action="budget.exceeded", ts=DAY + 20, scope="user"))
    store.close()
    return tmp_path


def test_jsonl_keeps_each_signed_line(folder: Path) -> None:
    rows = [json.loads(line) for line in export(folder, key=KEY).splitlines()]
    assert [row["action"] for row in rows] == ["approval.requested"] * 2 + ["budget.exceeded"]
    assert all(check_mac(KEY.encode(), row) for row in rows)


def test_a_window_is_since_inclusive_until_exclusive(folder: Path) -> None:
    text = export(folder, since=DAY + 10, until=DAY + 20, key=KEY)
    assert [json.loads(line)["seq"] for line in text.splitlines()] == [2]


def test_csv_has_a_header_iso_times_and_no_formulas(folder: Path) -> None:
    rows = list(csv.reader(io.StringIO(export(folder, format="csv", key=KEY))))
    assert tuple(rows[0]) == ("seq", "time", "action", "run_id", "call_id", "data", "prev", "mac")
    assert rows[1][1].endswith("+00:00") and len(rows) == 4
    assert json.loads(rows[2][5])["tool"] == "=HYPERLINK(1)"  # inside JSON: not a formula
    assert all(not cell.startswith("=") for row in rows for cell in row)


def test_a_formula_cell_is_quoted(tmp_path: Path) -> None:
    store = FileAuditStore(tmp_path, key=KEY, fsync=False)
    store.write(entry(action="=cmd", ts=DAY))
    store.close()
    rows = list(csv.reader(io.StringIO(export(tmp_path, format="csv", key=KEY))))
    assert rows[1][2] == "'=cmd"


def test_a_tampered_log_is_not_exported(folder: Path) -> None:
    ((_, path),) = segments(folder)
    path.write_bytes(path.read_bytes().replace(b"refund", b"refunx"))
    with pytest.raises(ConfigError, match="not intact"):
        export(folder, key=KEY)


def test_a_bad_format_is_refused(folder: Path) -> None:
    with pytest.raises(ConfigError, match="format"):
        export(folder, format="xml", key=KEY)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (5, 5.0),
        (2.5, 2.5),
        ("2026-11-01", datetime(2026, 11, 1, tzinfo=UTC).timestamp()),
        ("2026-11-01T10:00:00+01:00", datetime(2026, 11, 1, 9, tzinfo=UTC).timestamp()),
        (datetime(2026, 11, 1), datetime(2026, 11, 1, tzinfo=UTC).timestamp()),
        (date(2026, 11, 1), datetime(2026, 11, 1, tzinfo=UTC).timestamp()),
    ],
)
def test_moment_accepts_common_times(value: object, expected: float | None) -> None:
    assert moment(value) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [True, "yesterday", [1]])
def test_moment_refuses_anything_else(bad: object) -> None:
    with pytest.raises(ConfigError):
        moment(bad)  # type: ignore[arg-type]
