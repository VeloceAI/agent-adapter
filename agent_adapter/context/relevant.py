"""KeepMostRelevant: keep the units a scorer finds most relevant, drop the least relevant first."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar

from agent_adapter.context.scorer import Scorer
from agent_adapter.context.scoring import ScoringSession, unit_text
from agent_adapter.context.step import BaseCompressionStep, StepContext, check_count
from agent_adapter.context.steps import drop_units
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message

logger = logging.getLogger(__name__)

LATEST_USER_MESSAGE = "latest_user_message"


@dataclass(frozen=True)
class KeepMostRelevant(BaseCompressionStep):
    """Score every unprotected unit against `compare_to`; keep the `keep_count` best ones.

    The rest are dropped least relevant first (older first on a tie), only as many as needed.
    `compare_to` is the latest user message's text by default, or any text you give.
    """

    name: ClassVar[str] = "keep_most_relevant"
    scorer: Scorer
    compare_to: str = field(default=LATEST_USER_MESSAGE, kw_only=True)
    keep_count: int = field(default=10, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.scorer, Scorer):
            raise ConfigError("KeepMostRelevant needs a scorer, e.g. EmbeddingScorer(embedder)")
        if not isinstance(self.compare_to, str) or not self.compare_to.strip():
            raise ConfigError("compare_to must be 'latest_user_message' or a non-empty text")
        check_count("keep_count", self.keep_count, minimum=0)

    async def apply(self, messages: list[Message], context: StepContext) -> list[Message]:
        """`messages` without the least relevant units until under the target.

        Raises:
            ConfigError: the scorer returned bad scores.
        """
        if context.under_target(context.tokens(messages)):
            return messages
        units = context.units(messages)
        eligible = [unit for unit in units if not unit.protected]
        if len(eligible) <= self.keep_count:
            return messages
        query = self._query(messages)
        if not query:
            logger.debug("step %s: no text to compare with; nothing dropped", self.name)
            return messages
        async with ScoringSession(self.scorer, context, step=self.name) as session:
            scores = await session.score(query, [unit_text(unit) for unit in eligible])
        ranked = sorted(zip(scores, eligible, strict=True), key=lambda p: (p[0], p[1].start))
        candidates = [unit for _, unit in ranked[: len(eligible) - self.keep_count]]
        return drop_units(messages, units, candidates, context)

    def _query(self, messages: list[Message]) -> str:
        if self.compare_to != LATEST_USER_MESSAGE:
            return self.compare_to
        latest = next((m for m in reversed(messages) if m.role == "user"), None)
        return latest.content if latest is not None else ""
