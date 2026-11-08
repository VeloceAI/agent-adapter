"""Catch sqlite3 connections freed while still open, on every Python (3.13 only warns)."""

from __future__ import annotations

import contextlib
import gc
import sqlite3
import sqlite3.dbapi2
import traceback
import weakref
from pathlib import Path
from typing import Any

_ROOT = str(Path(__file__).resolve().parents[1])
_VENV = ".venv"
_real_connect = sqlite3.connect
_leaks: list[str] = []
_current = {"test": ""}
_opened: list[weakref.ref[TrackedConnection]] = []


class TrackedConnection(sqlite3.Connection):
    """A connection that records where it was opened and reports being freed unclosed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._open_site = ""
        self._closed = True  # a failed open has nothing to close
        super().__init__(*args, **kwargs)
        self._closed = False
        frames = traceback.extract_stack(limit=16)[:-2]
        ours = [f for f in frames if f.filename.startswith(_ROOT) and _VENV not in f.filename]
        lines = [f"{f.filename}:{f.lineno} in {f.name}" for f in (ours or frames)[-4:]]
        self._open_site = f"opened in {_current['test'] or 'collection'} at " + " <- ".join(
            reversed(lines)
        )
        _opened.append(weakref.ref(self))

    def close(self) -> None:
        """Close and remember it, so the finaliser stays quiet."""
        super().close()
        self._closed = True

    def __del__(self) -> None:
        if self._closed:
            return
        _leaks.append(self._open_site)
        # Report once here, not as a random unraisable warning later; a connection freed on
        # another thread cannot be closed from here, and sqlite closes it itself.
        with contextlib.suppress(sqlite3.ProgrammingError):
            self.close()


def _connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
    if len(args) <= 5 and "factory" not in kwargs:  # factory is the sixth parameter
        kwargs["factory"] = TrackedConnection
    return _real_connect(*args, **kwargs)


def install() -> None:
    """Route every `sqlite3.connect` (and SQLAlchemy's `dbapi2.connect`) through the tracker."""
    sqlite3.connect = _connect  # type: ignore[assignment]
    sqlite3.dbapi2.connect = _connect  # type: ignore[assignment]


def enter(test: str) -> None:
    """Name the running test, for the open site of new connections."""
    _current["test"] = test
    _opened.clear()


def take_leaks() -> list[str]:
    """The leaks found since the last call (collecting first if this test left any open)."""
    if any(_still_open(ref) for ref in _opened):
        gc.collect()  # an unclosed connection in a reference cycle: free it now, in its own test
    _opened.clear()
    found = list(_leaks)
    del _leaks[: len(found)]
    return found


def _still_open(ref: weakref.ref[TrackedConnection]) -> bool:
    conn = ref()
    return conn is not None and not conn._closed
