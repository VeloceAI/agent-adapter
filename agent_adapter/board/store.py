"""The `BoardStore` port and the in-process `MemoryBoardStore`."""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_adapter.board.entries import Entry
from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)


@runtime_checkable
class BoardStore(Protocol):
    """Where boards keep their entries and claims, one `scope` (run or thread) per board.

    `append` gives the entry the board's next `seq` and refuses it (None) when the board
    would pass `max_entries` or `max_bytes`; `claim` records the first holder of a key and
    returns (holder, True when this call recorded it). Both are atomic across every
    process the store serves. Backend errors propagate.
    """

    async def append(
        self, scope: str, entry: Entry, *, max_entries: int, max_bytes: int
    ) -> Entry | None: ...

    async def read(self, scope: str, *, since: int = 0) -> list[Entry]: ...

    async def claim(self, scope: str, key: str, holder: str) -> tuple[str, bool]: ...

    async def drop(self, scope: str) -> None: ...


@dataclass
class _Scope:
    entries: list[Entry] = field(default_factory=list)
    claims: dict[str, str] = field(default_factory=dict)
    size: int = 0


class MemoryBoardStore:
    """A `BoardStore` in this process (threads and event loops share it), the default.

    Keeps the boards of the last `max_scopes` runs; the least recently used is dropped.

    Raises:
        ConfigError: `max_scopes` is not a positive int.
    """

    def __init__(self, *, max_scopes: int = 256) -> None:
        if isinstance(max_scopes, bool) or not isinstance(max_scopes, int) or max_scopes < 1:
            raise ConfigError(f"max_scopes must be a positive int, got {max_scopes!r}")
        self.max_scopes = max_scopes
        self._scopes: OrderedDict[str, _Scope] = OrderedDict()
        self._lock = threading.Lock()

    async def append(
        self, scope: str, entry: Entry, *, max_entries: int, max_bytes: int
    ) -> Entry | None:
        """Add `entry` as the board's next one, or None when a cap would be passed."""
        size = entry.size()
        with self._lock:
            board = self._scope(scope)
            if len(board.entries) >= max_entries or board.size + size > max_bytes:
                return None
            stored = dataclasses.replace(entry, seq=len(board.entries) + 1)
            board.entries.append(stored)
            board.size += size
            return stored

    async def read(self, scope: str, *, since: int = 0) -> list[Entry]:
        """The board's entries after `seq` `since`, oldest first."""
        with self._lock:
            board = self._scopes.get(scope)
            return [] if board is None else board.entries[max(since, 0) :]

    async def claim(self, scope: str, key: str, holder: str) -> tuple[str, bool]:
        """The key's holder: `holder` when it was free (True) or already theirs."""
        with self._lock:
            claims = self._scope(scope).claims
            if key in claims:
                return claims[key], False
            claims[key] = holder
            return holder, True

    async def drop(self, scope: str) -> None:
        """Forget a board's entries and claims."""
        with self._lock:
            self._scopes.pop(scope, None)

    def _scope(self, scope: str) -> _Scope:
        board = self._scopes.get(scope)
        if board is None:
            board = self._scopes[scope] = _Scope()
            while len(self._scopes) > self.max_scopes:
                dropped, _ = self._scopes.popitem(last=False)
                logger.info("memory board store full: dropped the board of %s", dropped)
        self._scopes.move_to_end(scope)
        return board
