"""One board per run: the executor's settings, `board_of(ctx)` and the automatic posts."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_adapter.board._author import current_author
from agent_adapter.board._prompt import note, seen_upto
from agent_adapter.board.board import Board
from agent_adapter.board.entries import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_ENTRIES,
    Entry,
    check_caps,
    cut,
)
from agent_adapter.board.store import BoardStore, MemoryBoardStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolUsageError, log_failure
from agent_adapter.core.messages import Message
from agent_adapter.core.ports import EventSink
from agent_adapter.runtime.current import current_context

logger = logging.getLogger(__name__)

BOARD_KEY = "aa.board"  # ctx.extras: the run's Board, shared by every nested call
NOTE_CHARS = 600
MIN_NOTE_CHARS = 200


@dataclass(frozen=True)
class BoardSettings:
    """How an executor makes each run's board: the store, the scope and the caps.

    `scope="thread"` lets every run of one `thread_id` share a board (a run without a
    thread keeps its own).

    Raises:
        ToolUsageError: a store that is not a BoardStore, an unknown scope or bad caps.
    """

    store: BoardStore = field(default_factory=MemoryBoardStore)
    scope: Literal["run", "thread"] = "run"
    max_entries: int = DEFAULT_MAX_ENTRIES
    max_bytes: int = DEFAULT_MAX_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.store, BoardStore):
            raise ToolUsageError(f"store must be a BoardStore, got {type(self.store).__name__}")
        if self.scope not in ("run", "thread"):
            raise ToolUsageError(f"board scope must be 'run' or 'thread', got {self.scope!r}")
        check_caps(self.max_entries, self.max_bytes)

    def board(self, *, run_id: str, thread_id: str = "", sink: EventSink | None = None) -> Board:
        """The board of a run (or of its thread, with `scope="thread"`)."""
        threaded = self.scope == "thread" and thread_id
        scope = f"thread:{thread_id}" if threaded else f"run:{run_id}"
        return Board(
            self.store,
            scope=scope,
            run_id=run_id,
            sink=sink,
            max_entries=self.max_entries,
            max_bytes=self.max_bytes,
        )


def board_settings(option: BoardStore | BoardSettings | None) -> BoardSettings:
    """An executor's `board=` option as settings (None: a new in-memory store).

    Raises:
        ToolUsageError: neither a BoardStore nor BoardSettings.
    """
    if option is None:
        return BoardSettings()
    if isinstance(option, BoardSettings):
        return option
    if isinstance(option, BoardStore):
        return BoardSettings(store=option)
    raise ToolUsageError(f"board must be a BoardStore or BoardSettings, got {option!r}")


def board_of(ctx: ToolContext | None = None) -> Board:
    """The board of `ctx`'s run (default: the call running now), made on first use.

    Contexts from `Executor.context()` carry the executor's board; any other context gets
    an in-memory board that its nested calls share.

    Raises:
        ToolUsageError: no `ctx` and no call is running.
    """
    ctx = ctx if ctx is not None else current_context()
    if ctx is None:
        raise ToolUsageError("board_of() needs a ctx outside a running tool call or flow node")
    found = ctx.extras.get(BOARD_KEY)
    if isinstance(found, Board):
        return found
    board = Board(scope=f"run:{ctx.run_id or 'local'}", run_id=ctx.run_id, sink=ctx.sink)
    ctx.extras[BOARD_KEY] = board
    return board


def _found(ctx: ToolContext) -> Board | None:
    found = ctx.extras.get(BOARD_KEY)
    return found if isinstance(found, Board) else None


async def post_status(ctx: ToolContext, text: str, *, by: str) -> None:
    """Post an automatic status entry; never fails the caller (a full board is skipped)."""
    board = _found(ctx)
    if board is None:
        return
    entry = board._entry("status", text, key="", data=None, by=cut(by, 100))
    caps = {"max_entries": board.max_entries, "max_bytes": board.max_bytes}
    try:
        if await board.store.append(board.scope, entry, **caps) is None:
            logger.debug("board %s full: status of %s skipped", board.scope, by)
    except Exception as exc:  # any backend's error: a status is never worth failing a run
        log_failure(logger, logging.WARNING, exc, "board status of %s not posted", by)


async def snapshot(ctx: ToolContext) -> tuple[dict[str, Any], ...]:
    """The run's board entries for a checkpoint; () when it has no board or no entries."""
    board = _found(ctx)
    return () if board is None else tuple(await board.snapshot())


async def restore(ctx: ToolContext, rows: Iterable[Mapping[str, Any]]) -> None:
    """Put a checkpoint's board entries back on the run's board (made if missing)."""
    rows = list(rows)
    if rows:
        added = await board_of(ctx).restore(rows)
        logger.debug("board of run %s: %d entries restored", ctx.run_id, added)


async def board_note(
    ctx: ToolContext, history: Sequence[Message], max_chars: int = NOTE_CHARS
) -> Message | None:
    """What the other agents posted since this loop's last note, or None when nothing is new.

    The note goes into the history, so each entry reaches a model once; its first line
    records how far the board was read (`up to #n`), which also survives a checkpoint.
    """
    board = _found(ctx)
    if board is None:
        return None
    me = current_author() or None
    seen = seen_upto([m.content for m in history if m.role == "user"])
    entries = await board.store.read(board.scope)
    fresh = [e for e in entries if e.seq > seen and e.by != me]
    if not fresh:
        return None
    claims = [e for e in entries if e.kind == "claim" and e.seq <= seen and e.by != me]
    upto = max(e.seq for e in entries)
    text = note(fresh, claims, upto=upto, max_chars=max_chars)
    return Message(role="user", content=text) if text else None


def check_note_chars(value: object) -> int:
    """`board_chars=`: an int of at least MIN_NOTE_CHARS.

    Raises:
        ToolUsageError: anything else.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < MIN_NOTE_CHARS:
        raise ToolUsageError(f"board_chars must be an int >= {MIN_NOTE_CHARS}, got {value!r}")
    return value


def entries_of(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Snapshot rows checked as entries (a checkpoint being loaded).

    Raises:
        ValueError: a row is not an entry.
    """
    return tuple(Entry.from_dict(row).to_dict() for row in rows)
