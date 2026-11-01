"""Prompt layout: a stable prefix first, so provider prefix caches hit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from agent_adapter.cache.keys import digest
from agent_adapter.core.messages import Message

_JOIN = "\n\n"


@dataclass(frozen=True)
class PromptLayout:
    """Static prefix (system prompt, tool catalogue, few-shots) plus per-request text."""

    static: str
    dynamic: str = ""  # per-request context, placed in the final user turn
    # Named static sections appended after `static`, in insertion order.
    sections: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", MappingProxyType(dict(self.sections)))

    def system_text(self) -> str:
        """`static` then every non-empty section, separated by blank lines."""
        return _JOIN.join(part for part in (self.static, *self.sections.values()) if part)

    def messages(
        self, user: str | None = None, *, history: Sequence[Message] = ()
    ) -> tuple[Message, ...]:
        """System turn (cache breakpoint), `history`, then `dynamic` + `user` as one user turn."""
        out: list[Message] = []
        system = self.system_text()
        if system:
            out.append(Message(role="system", content=system, cache_breakpoint=True))
        out.extend(history)
        last = _JOIN.join(part for part in (self.dynamic, user) if part)
        if last:
            out.append(Message(role="user", content=last))
        return tuple(out)

    def prefix_digest(self) -> str:
        """Digest of `system_text()`; changes exactly when the cacheable prefix changes."""
        return digest(self.system_text())

    def with_static(self, text: str) -> PromptLayout:
        """A copy with `static` replaced."""
        return replace(self, static=text)

    def with_section(self, name: str, text: str) -> PromptLayout:
        """A copy with section `name` set (replaced in place, or appended)."""
        return replace(self, sections={**self.sections, name: text})
