"""Read recorded runs (`<start>.<run>.<pid>.jsonl` segments and tag sidecars) without the viewer.

The layering keeps evals below the viewer, so this is a small, read-only copy of its file
format; `tests/evals/test_from_runs_files.py` pins it to the viewer's.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

SEGMENT = re.compile(r"^(\d{13})\.([A-Za-z0-9_-]{1,64})\.(\d{1,10})\.jsonl$")
TAG_PREFIX, TAG_SUFFIX = "tags.", ".json"
MAX_SEGMENT_BYTES = 64 * 1024 * 1024  # read at most this much of one segment
MAX_TAG_FILE = 8_192
MAX_TAGS = 20
_SPAN_FAMILIES = frozenset({"tool", "llm", "embedding", "rerank", "node", "flow"})
_SPAN_ENDS = frozenset({"finished", "failed", "cached"})
_PENDING = frozenset({"tool.pending", "node.paused"})
_FAILED = frozenset(
    {
        "tool.failed",
        "llm.failed",
        "node.failed",
        "embedding.failed",
        "retrieval.failed",
        "sandbox.failed",
    }
)


@dataclass(frozen=True)
class RecordedRun:
    """One recorded run: its key, id, status, whether content was kept, events and tags."""

    key: str
    run_id: str
    status: str
    capture_content: bool
    events: tuple[dict[str, Any], ...]
    tags: tuple[str, ...] = ()


def read_runs(folder: str | os.PathLike[str]) -> Iterator[RecordedRun]:
    """Every run in a runs folder (or one run file), oldest first, read one run at a time.

    Raises:
        ConfigError: the folder is missing, or a file that is not a run segment.
        OSError: the folder cannot be listed.
    """
    target = Path(folder)
    if target.is_file():
        if SEGMENT.match(target.name) is None:
            raise ConfigError(f"{target} is not a run file (<start>.<run>.<pid>.jsonl)")
        segments, home = [target], target.parent
    elif target.is_dir():
        segments, home = _segments(target), target
    else:
        raise ConfigError(f"{target}: no such runs folder")
    groups: dict[str, list[Path]] = {}
    for path in segments:
        groups.setdefault(_key(path), []).append(path)
    for key, paths in groups.items():
        records = [record for path in paths for record in _records(path)]
        yield _run(key, records, _tags(home, key))


def _segments(folder: Path) -> list[Path]:
    found = [Path(e.path) for e in os.scandir(folder) if e.is_file() and SEGMENT.match(e.name)]
    return sorted(found, key=lambda p: (p.name.split(".", 1)[0], p.name))


def _key(path: Path) -> str:
    match = SEGMENT.match(path.name)
    return match.group(2) if match else ""


def _records(path: Path) -> Iterator[dict[str, Any]]:
    """The parsed JSON objects of one segment; a malformed or partial line is skipped."""
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_SEGMENT_BYTES)
    except FileNotFoundError:
        return  # rotated away between listing and reading
    lines = data.split(b"\n")
    if not data.endswith(b"\n"):
        lines.pop()  # a line still being written
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            logger.debug("skipped a malformed line in %s", path.name)
            continue
        if isinstance(record, dict):
            yield record


def _run(key: str, records: list[dict[str, Any]], tags: tuple[str, ...]) -> RecordedRun:
    meta = next((r for r in records if r.get("kind") == "meta"), {})
    events = tuple(_event(r) for r in records if r.get("kind") == "event")
    valid = tuple(e for e in events if e is not None)
    return RecordedRun(
        key=key,
        run_id=str(meta.get("run_id") or key),
        status=run_status(valid),
        capture_content=meta.get("capture_content") is True,
        events=valid,
        tags=tags,
    )


def _event(record: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(record.get("type"), str):
        return None
    data = record.get("data")
    return {**record, "data": data if isinstance(data, dict) else {}}


def run_status(events: tuple[dict[str, Any], ...]) -> str:
    """paused, ok, failed or unfinished, from a run's events (as the viewer decides it)."""
    waiting: dict[str, bool] = {}
    spans: dict[str, bool] = {}
    finished: list[bool] = []
    failed = scoped = False
    for event in events:
        kind, data = event["type"], event["data"]
        call = str(event.get("call_id") or "")
        if kind in _PENDING:
            waiting[call] = True
        elif call and kind.startswith(("tool.", "node.")):
            waiting[call] = False
        if kind == "run.started":
            scoped = True
        elif kind == "run.finished":
            finished.append(bool(data.get("ok")))
        failed = failed or (kind in _FAILED and not event.get("parent_id"))
        family, _, step = kind.partition(".")
        if family in _SPAN_FAMILIES:
            span = f"{family}:{call}" if call else f"{family}:{data.get(family, '')}"
            if step == "started":
                spans[span] = True
            elif step in _SPAN_ENDS or kind == "tool.pending":
                spans[span] = False
    if any(waiting.values()):
        return "paused"
    if finished:
        return "ok" if all(finished) else "failed"
    if not scoped and spans and not any(spans.values()):
        return "failed" if failed else "ok"  # calls outside a run scope, all ended
    return "unfinished"


def _tags(folder: Path, key: str) -> tuple[str, ...]:
    """The run's tags from its sidecar; a missing, huge or bad one gives none."""
    path = folder / f"{TAG_PREFIX}{key}{TAG_SUFFIX}"
    try:
        if not path.is_file() or path.stat().st_size > MAX_TAG_FILE:
            return ()
        tags = json.loads(path.read_bytes())["tags"]
    except FileNotFoundError:
        return ()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("ignored the tags in %s (%s)", path.name, type(exc).__name__)
        return ()
    if not isinstance(tags, list):
        return ()
    clean = [t for t in tags if isinstance(t, str) and t.strip() and len(t) <= 40]
    return tuple(dict.fromkeys(clean))[:MAX_TAGS]
