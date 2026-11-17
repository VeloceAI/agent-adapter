"""What composing a document returns: blanks, revisions, citations, issues and usage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.compose.blanks import Kind
from agent_adapter.retrieval.documents import Document


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
class Citation:
    """One cited phrase: its number, its source, the blank it is in and the words it backs."""

    number: int
    source: Document
    blank: str
    text: str


@dataclass(frozen=True)
class StageUsage:
    """Model calls, tokens and USD for one stage: "blanks", "editor" or "verifier"."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class ComposeResult:
    """The finished document (`.text`, in the template's format) and how it was made.

    `.issues` lists what is still wrong after the checks and the verifier, by blank
    ("document" for the whole text); `.usage` gives calls, tokens and cost per stage.
    """

    text: str
    blanks: Mapping[str, BlankResult]
    revisions: tuple[Revision, ...] = ()
    cost: float = 0.0
    run_id: str = ""
    format: str = "markdown"
    citations: tuple[Citation, ...] = ()
    issues: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    usage: Mapping[str, StageUsage] = field(default_factory=dict)

    @property
    def markdown(self) -> str:
        """The document text (Markdown for a Markdown template)."""
        return self.text

    @property
    def problems(self) -> Mapping[str, tuple[str, ...]]:
        """The blanks that failed their checks, with why."""
        return {name: found.problems for name, found in self.blanks.items() if found.problems}

    @property
    def ok(self) -> bool:
        """True when nothing is left in `issues`."""
        return not self.issues
