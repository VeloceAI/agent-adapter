"""`BudgetStatus`: one budget's limit, spend, holds and reset time, as callers see it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BudgetStatus:
    """One budget as a caller sees it: limit, spend, live holds and when it resets."""

    name: str
    scope: str
    scope_id: str
    window: str
    limit_usd: float
    spent_usd: float
    reserved_usd: float
    resets_at: float | None  # epoch seconds (UTC window end); None: never resets
    on_exceed: str

    @property
    def remaining_usd(self) -> float:
        """What is left before the limit, never below 0."""
        return max(0.0, self.limit_usd - self.spent_usd - self.reserved_usd)

    def view(self) -> dict[str, Any]:
        """A JSON-ready dict, `remaining_usd` included."""
        return {
            "name": self.name,
            "scope": self.scope,
            "key": self.scope_id,
            "window": self.window,
            "limit_usd": self.limit_usd,
            "spent_usd": round(self.spent_usd, 9),
            "reserved_usd": round(self.reserved_usd, 9),
            "remaining_usd": round(self.remaining_usd, 9),
            "resets_at": self.resets_at,
            "on_exceed": self.on_exceed,
        }
