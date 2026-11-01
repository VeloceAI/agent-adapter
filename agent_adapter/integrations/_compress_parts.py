"""A converter base for framework messages made of parts, plus items the model never sees."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

from agent_adapter.context.formats._shared import Converter, Group, Present
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message

Owned = tuple[int, ...] | None  # part positions a Message owns; None: the whole native
Piece = tuple[Message, Owned]


class PartsConverter(Converter):
    """Natives become Messages that each own the whole native or some of its parts.

    A native with no model-visible content (a rider: telemetry, provider state) joins the
    group of a neighbour and is kept while any part of that group is kept.
    """

    @abstractmethod
    def pieces(self, index: int, item: Any) -> list[Piece]:
        """The Messages of one native, in order; empty for a rider.

        Raises:
            ConfigError: the item is not a message of this format.
        """

    def retext(self, item: Any, message: Message) -> Any:
        """`item` (a whole-native Message) with `message.content` as its text."""
        raise NotImplementedError  # pragma: no cover - only formats with whole natives call it

    def parts(self, item: Any) -> Sequence[Any]:
        """The parts of `item` that positions index."""
        raise NotImplementedError  # pragma: no cover - only formats with parts call it

    def with_parts(self, item: Any, parts: list[Any]) -> Any:
        """`item` holding only `parts`."""
        raise NotImplementedError  # pragma: no cover - only formats with parts call it

    def retext_parts(self, item: Any, parts: list[Any], message: Message) -> list[Any]:
        """`parts` (one Message's) with `message.content` as their text."""
        raise NotImplementedError  # pragma: no cover - only formats with parts call it

    def rider_target(self, rider: int, before: int | None, after: int | None) -> int | None:
        """The native a rider joins: the next message native, else the previous one.

        Riders between the same two natives must all get the same answer, so order holds.
        """
        return after if after is not None else before

    def groups(self) -> list[Group]:
        found = [self.pieces(index, item) for index, item in enumerate(self.items)]
        after: list[int | None] = [None] * len(found)  # the next visible native, per index
        upcoming: int | None = None
        for index in range(len(found) - 1, -1, -1):
            after[index] = upcoming
            if found[index]:
                upcoming = index
        members: dict[int, list[int]] = {}
        before: int | None = None
        for index, pieces in enumerate(found):
            if pieces:
                members.setdefault(index, []).append(index)
                before = index
                continue
            target = self.rider_target(index, before, after[index])
            if target is None:
                raise ConfigError(f"no item of this {self.name} list is sent to the model")
            members.setdefault(target, []).append(index)
        return [
            Group(
                tuple(sorted(members[head])),
                tuple(message for message, _ in found[head]),
                tuple((head, owned) for _, owned in found[head]),
            )
            for head in sorted(members)
        ]

    def rebuild(self, group: Group, present: list[Present]) -> list[Any]:
        kept = {part.part: part for part in present}
        out: list[Any] = []
        for native in group.natives:
            owners = [(k, owned) for k, (head, owned) in enumerate(group.layout) if head == native]
            item = self.items[native]
            if not owners or all(k in kept and kept[k].unchanged for k, _ in owners):
                out.append(item)  # a rider, or every part as it was
                continue
            rebuilt = self._rebuilt(item, [(owned, kept.get(k)) for k, owned in owners])
            if rebuilt is not None:
                out.append(rebuilt)
        return out

    def _rebuilt(self, item: Any, owners: list[tuple[Owned, Present | None]]) -> Any:
        """`item` with only its present Messages, changed text written back; None if none."""
        if owners[0][0] is None:
            part = owners[0][1]
            return None if part is None else self.retext(item, part.message)
        all_parts = list(self.parts(item))
        out: list[Any] = []
        for owned, part in owners:
            if part is None or owned is None:
                continue
            mine = [all_parts[p] for p in owned]
            out.extend(mine if part.unchanged else self.retext_parts(item, mine, part.message))
        return self.with_parts(item, out) if out else None
