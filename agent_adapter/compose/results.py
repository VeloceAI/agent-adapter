"""What composing a document returns: each blank's value and checks, the editor's revisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.compose.blanks import Kind


@dataclass(frozen=True)
class BlankResult:
    """One filled blank: its value, its Markdown and how the checks went."""

    name: str
    kind: Kind
    value: Any
    markdown: str
    problems: tuple[str, ...] = ()
    attempts: int = 1
    cached: bool = False
    edited: bool = False
    cost_usd: float = 0.0

    @property
    def ok(self) -> bool:
        """True when the value passed every check."""
        return not self.problems


@dataclass(frozen=True)
class Revision:
    """One change the editor proposed: before, after, and whether the checks let it in."""

    blank: str
    before: str
    after: str
    reason: str = ""
    accepted: bool = True
    problems: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """The revision as plain data (kept in the flow state)."""
        return {
            "blank": self.blank,
            "before": self.before,
            "after": self.after,
            "reason": self.reason,
            "accepted": self.accepted,
            "problems": list(self.problems),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Revision:
        """A revision from `to_dict()` output."""
        return cls(
            blank=str(data["blank"]),
            before=str(data["before"]),
            after=str(data["after"]),
            reason=str(data.get("reason", "")),
            accepted=bool(data.get("accepted", True)),
            problems=tuple(data.get("problems", ())),
        )


@dataclass(frozen=True)
class ComposeResult:
    """The finished document: `.markdown`, `.blanks`, `.revisions` and `.cost` (USD)."""

    markdown: str
    blanks: Mapping[str, BlankResult]
    revisions: tuple[Revision, ...] = ()
    cost: float = 0.0
    run_id: str = ""
    problems: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when every blank passed its checks."""
        return not self.problems
