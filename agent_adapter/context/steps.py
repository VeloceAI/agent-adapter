"""Non-AI steps: remove old messages, remove duplicates, cut long messages, your own function."""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from agent_adapter.cache.keys import canonical_json
from agent_adapter.context.conversation import Unit
from agent_adapter.context.scorer import Scorer
from agent_adapter.context.scoring import ScoringSession, unit_shape, unit_text
from agent_adapter.context.step import BaseCompressionStep, StepContext, check_count, check_flag
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message

StepFunction = Callable[[list[Message]], list[Message] | Awaitable[list[Message]]]
CUT_REFINE_ROUNDS = 5  # re-measure after cutting; a counter is not linear in characters


def drop_units(
    messages: list[Message], units: Sequence[Unit], candidates: Sequence[Unit], context: StepContext
) -> list[Message]:
    """`messages` without `candidates` (in their order), stopping once under the target."""
    total = context.tokens(messages)
    removed: set[int] = set()
    for unit in candidates:
        if context.under_target(total):
            break
        if unit.protected:
            continue
        removed.add(unit.start)
        total -= sum(context.message_tokens(m) for m in unit.messages)
    if not removed:
        return messages
    return [m for unit in units if unit.start not in removed for m in unit.messages]


@dataclass(frozen=True, kw_only=True)
class RemoveOldMessages(BaseCompressionStep):
    """Drop the oldest messages between the first and last ones kept, only as many as needed.

    `keep_first` and `keep_last` count non-system messages; a tool round on the edge is kept.
    """

    name: ClassVar[str] = "remove_old_messages"
    keep_first: int = 1
    keep_last: int = 8

    def __post_init__(self) -> None:
        check_count("keep_first", self.keep_first, minimum=0)
        check_count("keep_last", self.keep_last, minimum=0)

    async def apply(self, messages: list[Message], context: StepContext) -> list[Message]:
        """`messages` with the oldest middle units removed until under the target."""
        if context.under_target(context.tokens(messages)):
            return messages
        others = [i for i, m in enumerate(messages) if m.role != "system"]
        if self.keep_first > len(others) or self.keep_last > len(others):
            return messages  # everything is kept
        head = others[self.keep_first - 1] + 1 if self.keep_first else 0
        tail = others[-self.keep_last] if self.keep_last else len(messages)
        units = context.units(messages)
        middle = [u for u in units if u.start >= head and u.stop <= tail]
        return drop_units(messages, units, middle, context)


@dataclass(frozen=True, kw_only=True)
class RemoveDuplicates(BaseCompressionStep):
    """Drop an older copy of a later message or tool round, oldest first, only as many as needed.

    `match="exact"` compares role, text and tool calls (not call ids); `"similar"` asks
    `scorer` and treats texts scoring above `same_if_score_above` as the same.
    """

    name: ClassVar[str] = "remove_duplicates"
    match: Literal["exact", "similar"] = "exact"
    scorer: Scorer | None = None
    same_if_score_above: float = 0.92

    def __post_init__(self) -> None:
        if self.match not in ("exact", "similar"):
            raise ConfigError(f"match must be 'exact' or 'similar', got {self.match!r}")
        if self.match == "similar" and not isinstance(self.scorer, Scorer):
            raise ConfigError("RemoveDuplicates(match='similar') needs a scorer")
        threshold = self.same_if_score_above
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ConfigError(f"same_if_score_above must be in 0..1, got {threshold!r}")

    async def apply(self, messages: list[Message], context: StepContext) -> list[Message]:
        """`messages` without older duplicates until under the target."""
        if context.under_target(context.tokens(messages)):
            return messages
        units = context.units(messages)
        if self.match == "similar":
            older = await self._similar_duplicates(units, context)
        else:
            older = _exact_duplicates(units)
        return drop_units(messages, units, older, context)

    async def _similar_duplicates(self, units: list[Unit], context: StepContext) -> list[Unit]:
        """Unprotected units the scorer finds near-identical to a later kept one, oldest first.

        Newest first, each unit is compared with the later units of the same shape that stay,
        so a removed unit is never the only copy of what it said.

        Raises:
            ConfigError: no scorer, or it returned bad scores.
        """
        if self.scorer is None:
            raise ConfigError("RemoveDuplicates(match='similar') needs a scorer")
        texts = [unit_text(unit) for unit in units]
        shapes = [unit_shape(unit) for unit in units]
        kept: list[int] = []
        older: list[Unit] = []
        async with ScoringSession(self.scorer, context, step=self.name) as session:
            for index in range(len(units) - 1, -1, -1):
                unit = units[index]
                later = [j for j in kept if shapes[j] == shapes[index] and texts[j]]
                if unit.protected or not texts[index] or not later:
                    kept.append(index)
                    continue
                scores = await session.score(texts[index], [texts[j] for j in later])
                if any(score > self.same_if_score_above for score in scores):
                    older.append(unit)
                else:
                    kept.append(index)
        older.reverse()
        return older


def _arguments(arguments: dict[str, Any]) -> str:
    try:
        return canonical_json(arguments)
    except TypeError:  # values JSON cannot hold still compare by their repr
        return repr(sorted(arguments.items(), key=lambda item: item[0]))


def unit_signature(unit: Unit) -> tuple[Any, ...]:
    """What makes two units the same: roles, names, texts and calls, not call ids."""
    return tuple(
        (
            m.role,
            m.name,
            m.content,
            tuple((call.name, _arguments(call.arguments)) for call in m.tool_calls),
        )
        for m in unit.messages
    )


def _exact_duplicates(units: Sequence[Unit]) -> list[Unit]:
    """Unprotected units with an exact copy later on, oldest first."""
    seen: set[tuple[Any, ...]] = set()
    older: list[Unit] = []
    for unit in reversed(units):
        signature = unit_signature(unit)
        if signature in seen and not unit.protected:
            older.append(unit)
        seen.add(signature)
    older.reverse()
    return older


@dataclass(frozen=True, kw_only=True)
class CutLongMessages(BaseCompressionStep):
    """Cut the text of messages over `max_tokens_per_message`, oldest first, keeping one end.

    Only text is cut, never tool calls; a marker says how many characters went. With
    `include_system_prompt=True`, an unpinned system message is cut too, after the others.
    """

    name: ClassVar[str] = "cut_long_messages"
    may_include_system_prompt: ClassVar[bool] = True
    max_tokens_per_message: int = 2000
    keep: Literal["start", "end", "start_and_end"] = "start"
    include_system_prompt: bool = False

    def __post_init__(self) -> None:
        check_count("max_tokens_per_message", self.max_tokens_per_message, minimum=1)
        check_flag("include_system_prompt", self.include_system_prompt)
        if self.keep not in ("start", "end", "start_and_end"):
            raise ConfigError(f"keep must be 'start', 'end' or 'start_and_end', got {self.keep!r}")

    async def apply(self, messages: list[Message], context: StepContext) -> list[Message]:
        """`messages` with long texts cut until under the target."""
        total = context.tokens(messages)
        if context.under_target(total):
            return messages
        out = list(messages)
        slots = [
            unit.start + offset
            for unit in context.units(messages)
            if not unit.protected
            for offset in range(unit.stop - unit.start)
        ]
        if self.include_system_prompt:  # last: the system prompt is cut only if still needed
            slots += [i for i, m in enumerate(messages) if m.role == "system" and not m.pinned]
        for index in slots:
            if context.under_target(total):
                break
            message = messages[index]
            tokens = context.message_tokens(message)
            if tokens <= self.max_tokens_per_message or not message.content:
                continue
            cut = self._cut(message, tokens, context)
            cut_tokens = context.message_tokens(cut)
            if cut_tokens < tokens:
                out[index] = cut
                total += cut_tokens - tokens
        return out

    def _cut(self, message: Message, tokens: int, context: StepContext) -> Message:
        content = message.content
        keep_chars = len(content) * self.max_tokens_per_message // tokens
        candidate = message
        for _ in range(CUT_REFINE_ROUNDS):
            text = shorten(content, keep_chars, self.keep)
            candidate = with_text(message, text)
            measured = context.message_tokens(candidate)
            if measured <= self.max_tokens_per_message or keep_chars == 0:
                break
            excess = measured - self.max_tokens_per_message
            keep_chars = max(0, keep_chars - max(1, math.ceil(excess * len(content) / tokens)))
        return candidate


def with_text(message: Message, text: str) -> Message:
    """`message` with new text; provider items go too, as ports replay them over the text."""
    if message.role == "system":
        return message.model_copy(update={"content": text})
    return message.model_copy(update={"content": text, "raw": None})


def shorten(content: str, keep_chars: int, keep: str) -> str:
    """`content` cut to `keep_chars` characters at the kept end(s), with a marker."""
    cut = len(content) - keep_chars
    if cut <= 0:
        return content
    marker = f"[... {cut} characters cut to save context ...]"
    if keep == "start":
        return f"{content[:keep_chars]}\n{marker}"
    if keep == "end":
        return f"{marker}\n{content[len(content) - keep_chars :]}"
    head = (keep_chars + 1) // 2
    tail = keep_chars - head
    return f"{content[:head]}\n{marker}\n{content[len(content) - tail :]}"


@dataclass(frozen=True)
class MyOwnStep(BaseCompressionStep):
    """Your function as a step: `list[Message] -> list[Message]`, sync or async.

    The invariant guard still applies to what it returns; with `include_system_prompt=True`
    it may also shorten the system message's text (never remove it or change anything else).
    """

    name: ClassVar[str] = "my_own_step"
    may_include_system_prompt: ClassVar[bool] = True
    function: StepFunction
    include_system_prompt: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise ConfigError("MyOwnStep needs a function(list[Message]) -> list[Message]")
        check_flag("include_system_prompt", self.include_system_prompt)

    async def apply(self, messages: list[Message], context: StepContext) -> list[Message]:
        """What the function returns for a copy of `messages`.

        Raises:
            ConfigError: the function returned something other than a list of Message.
        """
        result = self.function(list(messages))
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, list | tuple) or not all(isinstance(m, Message) for m in result):
            raise ConfigError("MyOwnStep's function must return a list of Message")
        return list(result)
