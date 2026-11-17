"""`board_tools()`: the board as model-visible tools (read, post, claim)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal

from agent_adapter.board._author import current_author
from agent_adapter.board._prompt import READ_LINE, render
from agent_adapter.board.context import board_of
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.tool import ToolSpec, build_spec

READ_CHARS = 4_000
FULL_CHARS = 2_600
MAX_LIMIT = 50
_TAGS = frozenset({"board"})


def _author() -> str:
    return current_author() or "main"


async def board_read(
    ctx: ToolContext,
    kind: Literal["status", "finding", "claim", "handoff"] | None = None,
    by: str | None = None,
    since: int = 0,
    entry: int = 0,
    limit: int = 10,
) -> str:
    """Read the shared board, newest first: what the agents of this run do and found.

    Args:
        kind: Only entries of this kind.
        by: Only entries by this agent.
        since: Only entries after this entry number (#n), to see what is new.
        entry: One entry (#n) in full, for a line that was cut.
        limit: At most this many entries (1 to 50).
    """
    board = board_of(ctx)
    if entry > 0:
        found = await board.read(since=entry - 1, limit=None)
        match = [e for e in found if e.seq == entry]
        if not match:
            return f"No entry #{entry} on the board."
        return render(match, FULL_CHARS, line_chars=FULL_CHARS, merge=False)
    wanted = min(max(limit, 1), MAX_LIMIT)
    entries = await board.read(kind=kind, by=by, since=since, limit=wanted)
    text = render(entries, READ_CHARS, line_chars=READ_LINE, newest_first=True, merge=False)
    return text or "No board entries match."


async def board_post(
    ctx: ToolContext,
    kind: Literal["status", "finding", "handoff"],
    text: str,
    key: str = "",
) -> str:
    """Post to the shared board so the other agents know what you are doing or found.

    Args:
        kind: "status" for what you are working on, "finding" for a result, "handoff" for
            a summary the next agent needs.
        text: Short and specific: one fact or one status.
        key: The topic, so others can find it (e.g. "pricing:oslo").
    """
    entry = await board_of(ctx).post(kind, text, key=key, by=_author())
    return f"posted as #{entry.seq}"


async def board_claim(ctx: ToolContext, key: str, note: str = "") -> str:
    """Claim a piece of work before doing it, so no other agent does it too.

    Args:
        key: The work's name, the same one other agents would use (e.g. "search:oslo hotels").
        note: What you will do.
    """
    claim = await board_of(ctx).claim(key, by=_author(), text=note)
    if claim.won:
        return f"claimed {key!r}: it is yours"
    return f"{claim.holder} already does {key!r}: pick other work and read the board later"


def board_tools(include: Iterable[str] = ("read", "post", "claim")) -> list[ToolSpec]:
    """`board_read`, `board_post` and `board_claim` (or the ones in `include`).

    They only write to the run's board, so they need no approval; add them to a registry
    and they run through the executor like any tool (guards and redaction apply).
    """
    made: dict[str, Callable[..., Any]] = {
        "read": board_read,
        "post": board_post,
        "claim": board_claim,
    }
    wanted = list(include)
    unknown = sorted(set(wanted) - set(made))
    if unknown:
        raise ToolUsageError(f"board_tools: unknown tools {unknown}; pick from {sorted(made)}")
    return [build_spec(made[name], tags=_TAGS, parallel_safe=True) for name in wanted]
