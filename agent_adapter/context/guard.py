"""The invariant guard: undo any part of a step's output that breaks a compression rule."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from difflib import SequenceMatcher

from agent_adapter.context.conversation import Protection, split_units
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message

logger = logging.getLogger(__name__)


def enforce_invariants(
    before: Sequence[Message],
    after: object,
    protection: Protection,
    *,
    step: str,
    allow_system_prompt_shortening: bool = False,
) -> list[Message]:
    """`after` with every change that breaks an invariant skipped (the input message kept).

    Rules: protected messages stay as they were; a tool round is kept whole if any of it is
    kept; a step may not add system or tool messages or tool calls; added messages land after
    the leading system messages, outside tool rounds and before the last unit. With
    `allow_system_prompt_shortening`, an unpinned system message may get shorter content in
    place (nothing else about it may change, and it is never removed).

    Raises:
        ConfigError: `after` is not a list of `Message`.
    """
    if not isinstance(after, list | tuple) or not all(isinstance(m, Message) for m in after):
        raise ConfigError(f"compression step {step!r} must return a list of Message")
    plan = _Plan(before, after, protection, allow_system_prompt_shortening)
    plan.revert_protected()
    plan.revert_moved_tool_messages()
    plan.keep_units_whole()
    result = plan.assemble()
    if plan.skipped:
        logger.warning(
            "compression step %s: skipped %d change(s) that would break an invariant",
            step,
            plan.skipped,
        )
    return result


def _input_indices(before: Sequence[Message], after: Sequence[Message]) -> list[int | None]:
    """The input index each output message is (same object, else an equal copy); None if new."""
    positions: dict[int, list[int]] = defaultdict(list)  # one object may appear twice
    for i, message in enumerate(before):
        positions[id(message)].append(i)
    keys: list[int | None] = [None] * len(after)
    used: set[int] = set()
    for j, message in enumerate(after):
        index = next((i for i in positions.get(id(message), ()) if i not in used), None)
        if index is not None:
            keys[j] = index
            used.add(index)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, message in enumerate(before):
        buckets[(message.role, message.content)].append(i)
    for j, message in enumerate(after):
        if keys[j] is not None:
            continue
        for index in buckets.get((message.role, message.content), ()):
            if index not in used and before[index] == message:
                keys[j] = index
                used.add(index)
                break
    return keys


def is_system_prompt_shortening(old: Message, new: Message) -> bool:
    """True when `new` is `old`, an unpinned system message, with only shorter content."""
    return (
        old.role == "system"
        and not old.pinned
        and len(new.content) < len(old.content)
        and new == old.model_copy(update={"content": new.content})
    )


def _same_slot(old: Message, new: Message) -> bool:
    """True when `new` can replace `old` in place: only content, raw or flags differ."""
    return (
        old.role == new.role
        and old.tool_call_id == new.tool_call_id
        and old.name == new.name
        and old.tool_calls == new.tool_calls
    )


class _Plan:
    """Per input message: what to emit in its place (None: dropped); plus added messages."""

    def __init__(
        self,
        before: Sequence[Message],
        after: Sequence[Message],
        protection: Protection,
        shorten_system: bool,
    ) -> None:
        self.before = list(before)
        self.protection = protection
        self.shorten_system = shorten_system
        self.units = split_units(self.before, protection)
        self.skipped = 0
        self.emit: list[Message | None] = [None] * len(self.before)
        self.added: list[tuple[int, int, Message, int | None]] = []  # anchor, order, msg, origin
        keys = _input_indices(self.before, after)
        sequence_a = list(range(len(self.before)))
        sequence_b = [key if key is not None else -1 - j for j, key in enumerate(keys)]
        matcher = SequenceMatcher(None, sequence_a, sequence_b, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for i in range(i1, i2):
                    self.emit[i] = self.before[i]
            elif tag == "replace" and self._in_place(i1, i2, j1, j2, keys, after):
                for i, j in zip(range(i1, i2), range(j1, j2), strict=True):
                    self.emit[i] = after[j]
            elif tag != "delete":
                self._replace(i1, i2, j1, j2, keys, after)

    def _replace(
        self,
        i1: int,
        i2: int,
        j1: int,
        j2: int,
        keys: list[int | None],
        after: Sequence[Message],
    ) -> None:
        """Uneven change: a shortened system message keeps its slot; the rest are additions."""
        systems = [i for i in range(i1, i2) if self.before[i].role == "system"]
        for j in range(j1, j2):
            message = after[j]
            if self.shorten_system and keys[j] is None and systems and message.role == "system":
                index = systems.pop(0)
                if is_system_prompt_shortening(self.before[index], message):
                    self.emit[index] = message
                    continue
            self.added.append((i1, j, message, keys[j]))

    def _in_place(
        self,
        i1: int,
        i2: int,
        j1: int,
        j2: int,
        keys: list[int | None],
        after: Sequence[Message],
    ) -> bool:
        return i2 - i1 == j2 - j1 and all(
            keys[j] is None and _same_slot(self.before[i], after[j])
            for i, j in zip(range(i1, i2), range(j1, j2), strict=True)
        )

    def _keep(self, index: int) -> None:
        self.emit[index] = self.before[index]
        self.skipped += 1

    def revert_protected(self) -> None:
        """Protected messages come back unchanged."""
        for i, message in enumerate(self.before):
            new = self.emit[i]
            if not self.protection.is_protected(message) or new is message:
                continue
            if (
                new is not None
                and self.shorten_system
                and is_system_prompt_shortening(message, new)
            ):
                continue
            self._keep(i)

    def revert_moved_tool_messages(self) -> None:
        """A tool call or result moved elsewhere stays where it was."""
        for _, _, message, origin in self.added:
            if origin is not None and self.emit[origin] is None and _tool_like(message):
                self._keep(origin)

    def keep_units_whole(self) -> None:
        """A tool round partly kept is kept whole: no orphan call or result."""
        for unit in self.units:
            members = range(unit.start, unit.stop)
            if len(members) < 2:
                continue
            if any(self.emit[i] is not None for i in members):
                for i in members:
                    if self.emit[i] is None:
                        self._keep(i)

    def _anchor(self, anchor: int) -> int:
        """Where an added message may go: after the system lead, outside rounds, and before
        the latest turn (which starts at the latest user message)."""
        if self.units:
            anchor = min(anchor, self.units[-1].start)
        latest_user = next(
            (unit.start for unit in reversed(self.units) if unit.kind == "user"), None
        )
        if latest_user is not None:
            anchor = min(anchor, latest_user)
        lead = 0
        while lead < len(self.before) and self.before[lead].role == "system":
            lead += 1
        anchor = max(anchor, lead)
        for unit in self.units:
            if unit.start < anchor < unit.stop:
                anchor = unit.stop
        return anchor

    def assemble(self) -> list[Message]:
        """The output: kept or changed input messages, with allowed additions in place."""
        placed: list[tuple[int, int, Message]] = []
        for anchor, order, message, origin in self.added:
            duplicate = origin is not None and self.emit[origin] is not None
            if duplicate or _tool_like(message) or message.role == "system":
                self.skipped += 1
                continue
            placed.append((self._anchor(anchor), order, message))
        placed.sort(key=lambda item: (item[0], item[1]))
        result: list[Message] = []
        cursor = 0
        for i, kept in enumerate(self.emit):
            while cursor < len(placed) and placed[cursor][0] == i:
                result.append(placed[cursor][2])
                cursor += 1
            if kept is not None:
                result.append(kept)
        result.extend(item[2] for item in placed[cursor:])
        return result


def _tool_like(message: Message) -> bool:
    return message.role == "tool" or bool(message.tool_calls)
