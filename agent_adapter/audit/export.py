"""Export a verified audit log, or a time window of it, as JSONL or CSV."""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any, Literal, TextIO

from agent_adapter.audit.verify import signed_events
from agent_adapter.core.errors import ConfigError

Format = Literal["jsonl", "csv"]
Moment = datetime | date | float | int | str | None
CSV_COLUMNS = ("seq", "time", "action", "run_id", "call_id", "data", "prev", "mac")
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")  # spreadsheet formula triggers


def export(
    path: str | os.PathLike[str],
    *,
    since: Moment = None,
    until: Moment = None,
    format: Format = "jsonl",
    key: str | bytes | None = None,
) -> str:
    """Verified event records with `since <= ts < until` as text (large logs: `export_to`).

    Raises:
        ConfigError: no key, a bad format or time, or a log that is not intact.
    """
    buffer = io.StringIO()
    export_to(buffer, path, since=since, until=until, format=format, key=key)
    return buffer.getvalue()


def export_to(
    out: TextIO,
    path: str | os.PathLike[str],
    *,
    since: Moment = None,
    until: Moment = None,
    format: Format = "jsonl",
    key: str | bytes | None = None,
) -> int:
    """Stream `export`'s records to `out` after a full check; the count written.

    Raises:
        ConfigError: no key, a bad format or time, or a log that is not intact (a change
            found while reading raises too: discard what was written).
    """
    if format not in ("jsonl", "csv"):
        raise ConfigError(f"format must be 'jsonl' or 'csv', got {format!r}")
    start, end = moment(since), moment(until)
    writer = csv.writer(out, lineterminator="\n") if format == "csv" else None
    count = 0
    for record, line in signed_events(path, key):
        if not _inside(record, start, end):
            continue
        if writer is None:
            out.write(line.decode("utf-8") + "\n")
        else:
            if count == 0:
                writer.writerow(CSV_COLUMNS)
            writer.writerow(_row(record))
        count += 1
    if writer is not None and count == 0:
        writer.writerow(CSV_COLUMNS)
    return count


def moment(value: Moment) -> float | None:
    """Epoch seconds for a datetime, a date (UTC midnight), a number or an ISO string.

    Raises:
        ConfigError: anything else, or a string that is not ISO 8601.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError("a time must be a datetime, a date, epoch seconds or ISO text")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"not an ISO 8601 time: {value!r}") from exc
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.timestamp()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp()
    raise ConfigError("a time must be a datetime, a date, epoch seconds or ISO text")


def _inside(record: Mapping[str, Any], start: float | None, end: float | None) -> bool:
    ts = record.get("ts")
    if not isinstance(ts, int | float):
        return False
    return (start is None or ts >= start) and (end is None or ts < end)


def _row(record: Mapping[str, Any]) -> list[str]:
    when = datetime.fromtimestamp(float(record["ts"]), UTC).isoformat()
    data = json.dumps(record.get("data", {}), sort_keys=True, ensure_ascii=False)
    row = [
        record.get("seq"),
        when,
        record.get("action"),
        record.get("run_id"),
        record.get("call_id"),
        data,
        record.get("prev"),
        record.get("mac"),
    ]
    return [_cell(value) for value in row]


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return f"'{text}" if text.startswith(_FORMULA_START) else text
