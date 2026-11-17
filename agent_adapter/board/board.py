"""`Board`: a run's shared, append-only notes: statuses, findings, claims and handoffs."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from typing import Any

from agent_adapter.board._author import current_author
from agent_adapter.board._prompt import render
from agent_adapter.board.entries import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_ENTRIES,
    MAX_BY,
    MAX_KEY,
    Claim,
    Entry,
    check_caps,
    check_data,
    check_kind,
    check_short,
    cut,
)
from agent_adapter.board.store import BoardStore, MemoryBoardStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolError, ToolUsageError
from agent_adapter.core.ports import EventSink
from agent_adapter.runtime.current import current_context
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)


class Board:
    """One board: what the agents of a run (or thread) are doing and have found.

    Entries are only added, never changed. `by` defaults to the agent or flow node that is
    running (`posting_as`); every post and claim is a `board.posted` / `board.claimed`
    event on `sink`. A post that would pass `max_entries` or `max_bytes` is refused.

    Raises:
        ToolUsageError: an empty `scope`, a store that is not a BoardStore, or bad caps.
    """

    def __init__(
        self,
        store: BoardStore | None = None,
        *,
        scope: str,
        run_id: str = "",
        sink: EventSink | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        store = store if store is not None else MemoryBoardStore(max_scopes=1)
        if not isinstance(store, BoardStore):
            raise ToolUsageError(f"store must be a BoardStore, got {type(store).__name__}")
        check_short("scope", scope, 300, empty=False)
        check_caps(max_entries, max_bytes)
        self.store = store
        self.scope = scope
        self.run_id = run_id
        self.sink = sink
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._clock = clock

    def __repr__(self) -> str:
        return f"Board(scope={self.scope!r}, store={type(self.store).__name__})"

    async def post(
        self,
        kind: str,
        text: str,
        *,
        key: str = "",
        data: Mapping[str, Any] | None = None,
        by: str | None = None,
    ) -> Entry:
        """Add an entry; text past `MAX_TEXT` characters is cut.

        Raises:
            ToolUsageError: an unknown kind, a bad key, author or data.
            ToolError: the board is full (the model sees why).
        """
        entry = self._entry(check_kind(kind), text, key=key, data=data, by=by)
        stored = await self._append(entry)
        self._emit(EventType.BOARD_POSTED, stored, seq=stored.seq)
        return stored

    async def claim(
        self,
        key: str,
        *,
        by: str | None = None,
        text: str = "",
        data: Mapping[str, Any] | None = None,
    ) -> Claim:
        """Take `key` if nobody holds it; the first caller wins, later ones see the holder.

        The winner's claim is also posted as a `claim` entry. Claiming again a key one
        already holds wins again and posts nothing.

        Raises:
            ToolUsageError: an empty key or author.
            ToolError: the board is full when the winner's entry is posted.
        """
        check_short("key", key, MAX_KEY, empty=False)
        author = check_short("by", self._by(by), MAX_BY, empty=False)
        holder, made = await self.store.claim(self.scope, key, author)
        claim = Claim(key=key, holder=holder, won=holder == author)
        if made:
            entry = self._entry("claim", text or f"claimed {key}", key=key, data=data, by=author)
            await self._append(entry)
        self._emit_claim(claim, author)
        return claim

    async def read(
        self,
        *,
        kind: str | Iterable[str] | None = None,
        by: str | None = None,
        key: str | None = None,
        since: int = 0,
        limit: int | None = None,
    ) -> list[Entry]:
        """Entries after `seq` `since` (oldest first), filtered; `limit` keeps the newest."""
        kinds = {kind} if isinstance(kind, str) else set(kind) if kind is not None else None
        found = [
            entry
            for entry in await self.store.read(self.scope, since=since)
            if (kinds is None or entry.kind in kinds)
            and (by is None or entry.by == by)
            and (key is None or entry.key == key)
        ]
        return found[-limit:] if limit is not None and limit >= 0 else found

    async def watch(
        self,
        *,
        since: int = 0,
        kind: str | Iterable[str] | None = None,
        poll_s: float = 0.1,
    ) -> AsyncIterator[Entry]:
        """New entries as they arrive, forever; stop by breaking out or cancelling.

        Every store is polled each `poll_s` seconds, so it works across processes too.
        """
        last = since
        while True:
            fresh = await self.store.read(self.scope, since=last)
            for entry in fresh:
                last = max(last, entry.seq)
                if kind is None or entry.kind in ({kind} if isinstance(kind, str) else set(kind)):
                    yield entry
            if not fresh:
                await asyncio.sleep(poll_s)

    async def summary(
        self,
        max_chars: int = 1_500,
        *,
        exclude_by: str | None = None,
        include_own: bool = False,
    ) -> str:
        """The board as short prompt text, fenced and marked as data ("" when empty).

        The running agent's own entries are left out (`exclude_by` names another author;
        `include_own=True` keeps them), only each author's latest status is kept, the
        newest entries win the space, and a long text ends with a pointer to read it whole
        (`board_read(entry=n)`).
        """
        if exclude_by is None and not include_own:
            exclude_by = current_author() or None
        entries = await self.store.read(self.scope)
        return render(entries, max_chars, exclude_by=exclude_by)

    async def snapshot(self) -> list[dict[str, Any]]:
        """Every entry as plain data, for a checkpoint."""
        return [entry.to_dict() for entry in await self.store.read(self.scope)]

    async def restore(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Put back snapshot entries the store lacks (by id); returns how many were added."""
        have = {entry.id for entry in await self.store.read(self.scope)}
        added = 0
        for row in rows:
            entry = Entry.from_dict(row)
            if entry.id in have:
                continue
            caps = {"max_entries": self.max_entries, "max_bytes": self.max_bytes}
            if await self.store.append(self.scope, entry, **caps) is None:
                logger.warning("board %s full: %d snapshot entries not restored", self.scope, added)
                break
            have.add(entry.id)
            added += 1
        return added

    def _by(self, by: str | None) -> str:
        return current_author() if by is None else by

    def _entry(
        self,
        kind: str,
        text: str,
        *,
        key: str,
        data: Mapping[str, Any] | None,
        by: str | None,
    ) -> Entry:
        if not isinstance(text, str):
            raise ToolUsageError(f"board entry text must be a str, got {type(text).__name__}")
        return Entry(
            id=new_id("bd_"),
            at=self._clock(),
            by=check_short("by", self._by(by), MAX_BY),
            kind=kind,
            key=check_short("key", key, MAX_KEY),
            text=cut(text),
            data=check_data(data),
        )

    async def _append(self, entry: Entry) -> Entry:
        caps = {"max_entries": self.max_entries, "max_bytes": self.max_bytes}
        stored = await self.store.append(self.scope, entry, **caps)
        if stored is None:
            logger.warning("board %s is full: a %s entry was refused", self.scope, entry.kind)
            raise ToolError(
                f"the shared board is full ({self.max_entries} entries or "
                f"{self.max_bytes} bytes); read it instead of posting more"
            )
        return stored

    def _ctx(self) -> ToolContext:
        current = current_context()
        if current is not None and current.run_id == self.run_id:
            return current
        return ToolContext(run_id=self.run_id)

    def _emit(self, type: str, entry: Entry, **extra: Any) -> None:
        data: dict[str, Any] = {
            "kind": entry.kind,
            "key": entry.key,
            "by": entry.by,
            "chars": len(entry.text),
        }
        emit(self.sink, make_event(type, self._ctx(), entry_id=entry.id, **data, **extra))

    def _emit_claim(self, claim: Claim, by: str) -> None:
        event = make_event(
            EventType.BOARD_CLAIMED,
            self._ctx(),
            key=claim.key,
            by=by,
            holder=claim.holder,
            won=claim.won,
        )
        emit(self.sink, event)
