"""Budget counter keys: who a call is charged to, and which UTC window it falls in."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from urllib.parse import quote

from agent_adapter.core.context import ToolContext

Per = Literal["day", "month"]
PERIODS: tuple[Per, ...] = ("day", "month")
# A counter outlives its window by this long, so a late settle still lands on it.
KEEP_AFTER_S = 3600.0


@dataclass(frozen=True)
class Window:
    """One UTC budget window: `label` names it in keys; `end` is when it resets (epoch s)."""

    label: str
    start: float
    end: float | None


ALL_TIME = Window("all", 0.0, None)


def window_at(per: Per | None, now: float) -> Window:
    """The window holding `now`: a UTC day, a UTC month, or all time for `per=None`."""
    if per is None:
        return ALL_TIME
    day = datetime.fromtimestamp(now, UTC).date()
    if per == "day":
        return Window(f"day:{day.isoformat()}", _epoch(day), _epoch(day + timedelta(days=1)))
    first = day.replace(day=1)
    following = (first + timedelta(days=32)).replace(day=1)
    return Window(f"month:{first:%Y-%m}", _epoch(first), _epoch(following))


def _epoch(day: date) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()


def counter_key(name: str, scope: str, scope_id: str, window: Window) -> str:
    """The store key of one counter; every part is percent-encoded, so parts never collide."""
    parts = (name, scope, scope_id, window.label)
    return "|".join(quote(part, safe="") for part in parts)


def principal_of(ctx: ToolContext) -> str:
    """Who a call is charged to: the authenticated principal's id, else `ctx.user_id`."""
    return _principal_id(ctx) or ctx.user_id


def user_differs(ctx: ToolContext) -> bool:
    """True when both a principal and a different `ctx.user_id` are set."""
    principal = _principal_id(ctx)
    return bool(principal and ctx.user_id and ctx.user_id != principal)


def _principal_id(ctx: ToolContext) -> str:
    principal = ctx.extras.get("principal")
    if isinstance(principal, str):
        return principal
    found = principal.get("id") if isinstance(principal, Mapping) else getattr(principal, "id", "")
    return found if isinstance(found, str) else ""
