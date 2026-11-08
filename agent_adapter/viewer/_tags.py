"""Run tags: one small `tags.<run key>.json` sidecar per run, next to its segments."""

from __future__ import annotations

import json
import logging
import os
import re
import stat as stat_mode
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

from agent_adapter.core.errors import ToolUsageError

logger = logging.getLogger(__name__)

PREFIX, SUFFIX = "tags.", ".json"  # the key never starts a name (Windows device names)
MAX_TAGS = 20  # per run
MAX_TAG = 40  # characters of one tag
MAX_FILE = 8_192  # bytes of one sidecar read; a bigger one is ignored
_TAG = re.compile(r"[\w][\w .:/+-]*", re.UNICODE)
_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")


def clean_tag(value: object) -> str:
    """A tag trimmed and lower-cased.

    Raises:
        ToolUsageError: not a string, empty, too long, or with other characters than letters,
            digits, space and `_ . : / + -`.
    """
    if not isinstance(value, str):
        raise ToolUsageError("a tag must be a string")
    tag = " ".join(value.split()).lower()
    if not tag or len(tag) > MAX_TAG or not _TAG.fullmatch(tag):
        raise ToolUsageError(
            f"a tag holds 1-{MAX_TAG} letters, digits, spaces or _ . : / + - characters"
        )
    return tag


class TagStore:
    """Reads and writes the tag sidecars of one runs folder; cached by file modification time."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._cache: dict[str, tuple[float, list[str]]] = {}
        self._lock = threading.Lock()

    def all(self) -> dict[str, list[str]]:
        """Tags by run key, for every run that has any.

        Raises:
            OSError: the folder exists but cannot be listed.
        """
        try:
            entries = [
                e
                for e in os.scandir(self.folder)
                if e.name.startswith(PREFIX) and e.name.endswith(SUFFIX)
            ]
        except FileNotFoundError:
            return {}
        found: dict[str, list[str]] = {}
        with self._lock:
            for entry in entries:
                key = entry.name.removeprefix(PREFIX).removesuffix(SUFFIX)
                if not _KEY.fullmatch(key):
                    continue
                tags = self._read(key, Path(entry.path))
                if tags:
                    found[key] = tags
            for key in set(self._cache) - set(found):
                del self._cache[key]
        return found

    def get(self, key: str) -> list[str]:
        """The tags of one run (none when it has no sidecar or an unreadable one).

        Raises:
            ToolUsageError: `key` is not a run key.
        """
        with self._lock:
            return list(self._read(key, self._path(key)))

    def update(
        self, key: str, add: Sequence[object] = (), remove: Sequence[object] = ()
    ) -> list[str]:
        """Add then remove tags of run `key`; the sidecar goes when no tag is left.

        Raises:
            ToolUsageError: a bad key or tag, or more than `MAX_TAGS` tags.
            OSError: the sidecar cannot be written.
        """
        added = [clean_tag(tag) for tag in add]
        removed = {clean_tag(tag) for tag in remove}
        path = self._path(key)
        with self._lock:
            tags = _merged(self._read(key, path), added, removed)
            if len(tags) > MAX_TAGS:
                raise ToolUsageError(f"a run holds at most {MAX_TAGS} tags")
            if tags:
                _write(path, tags)
            else:
                path.unlink(missing_ok=True)
            self._cache.pop(key, None)
        logger.info("viewer: run %s now has %d tag(s)", key, len(tags))
        return tags

    def _path(self, key: str) -> Path:
        if not _KEY.fullmatch(key):
            raise ToolUsageError("not a run key")
        return self.folder / f"{PREFIX}{key}{SUFFIX}"

    def _read(self, key: str, path: Path) -> list[str]:
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._cache.pop(key, None)
            return []
        if not stat_mode.S_ISREG(stat.st_mode):
            logger.warning("viewer: ignored %s: not a regular file", path.name)
            return []
        cached = self._cache.get(key)
        if cached is not None and cached[0] == stat.st_mtime:
            return cached[1]
        tags = _load(path, stat.st_size)
        self._cache[key] = (stat.st_mtime, tags)
        return tags


def _merged(tags: Iterable[str], added: list[str], removed: set[str]) -> list[str]:
    out: list[str] = []
    for tag in [*tags, *added]:
        if tag not in out and tag not in removed:
            out.append(tag)
    return out


def _load(path: Path, size: int) -> list[str]:
    """The tags of a sidecar; a bad one is ignored with a WARNING."""
    if size > MAX_FILE:
        logger.warning("viewer: ignored %s: over %d bytes", path.name, MAX_FILE)
        return []
    try:
        data = json.loads(path.read_bytes())
        tags = [clean_tag(tag) for tag in data["tags"]][:MAX_TAGS]
    except (OSError, ValueError, KeyError, TypeError, ToolUsageError) as exc:
        logger.warning("viewer: ignored the tags in %s (%s)", path.name, type(exc).__name__)
        return []
    return list(dict.fromkeys(tags))


def _write(path: Path, tags: list[str]) -> None:
    """Write the sidecar whole, through a temporary file, so readers never see half of it."""
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temp.write_text(json.dumps({"v": 1, "tags": tags}, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        temp.unlink(missing_ok=True)
        raise
