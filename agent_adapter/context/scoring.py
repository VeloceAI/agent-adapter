"""How steps score units with a Scorer: unit texts, checked scores and one ledger entry."""

from __future__ import annotations

import json
import logging
import math
import numbers
import time
from collections.abc import Sequence
from contextlib import nullcontext
from types import TracebackType
from typing import Any

from agent_adapter.context.conversation import Unit
from agent_adapter.context.entries import record, run_context, scoring_entry
from agent_adapter.context.scorer import Scorer
from agent_adapter.context.step import StepContext
from agent_adapter.core.errors import ConfigError, qualified_name
from agent_adapter.core.types import ErrorInfo
from agent_adapter.runtime.current import use_context

logger = logging.getLogger(__name__)

SCORE_SLACK = 1e-6  # float noise tolerated just outside 0..1


def unit_text(unit: Unit) -> str:
    """The text a scorer sees for a unit: message texts and tool calls, one per line."""
    lines = []
    for message in unit.messages:
        if message.content:
            lines.append(message.content)
        for call in message.tool_calls:
            arguments = json.dumps(call.arguments, ensure_ascii=False, default=str, sort_keys=True)
            lines.append(f"{call.name}({arguments})")
    return "\n".join(lines)


def unit_shape(unit: Unit) -> tuple[Any, ...]:
    """What two units must share to be compared: kind, roles and tool names."""
    return (
        unit.kind,
        tuple((m.role, m.name, tuple(c.name for c in m.tool_calls)) for m in unit.messages),
    )


class ScoringSession:
    """Scores texts for one step run; on exit writes one ledger entry if the scorer was called."""

    def __init__(self, scorer: Scorer, context: StepContext, *, step: str) -> None:
        self.scorer = scorer
        self.context = context
        self.step = step
        self.calls = 0
        self.texts = 0
        self._started_at = time.time()

    async def __aenter__(self) -> ScoringSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self.calls:
            return
        ctx = run_context(self.context)
        entry = scoring_entry(
            ctx,
            step=self.step,
            scorer=qualified_name(type(self.scorer)),
            calls=self.calls,
            texts=self.texts,
            started_at=self._started_at,
            error_kind="" if exc is None else ErrorInfo.from_exception(exc).kind,
        )
        record(ctx.ledger, entry)

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One score in 0..1 per text; empty texts score 0 and are not sent.

        Raises:
            ConfigError: the scorer returned the wrong count or a value outside 0..1.
        """
        sent = [text for text in texts if text]
        scores: list[float] = []
        if sent:
            self.calls += 1
            self.texts += len(sent)
            scope = use_context(self.context.ctx) if self.context.ctx is not None else nullcontext()
            with scope:
                scores = await self.scorer.score(query, sent)
            scores = self._checked(scores, len(sent))
        found = iter(scores)
        return [next(found) if text else 0.0 for text in texts]

    def _checked(self, scores: object, expected: int) -> list[float]:
        name = qualified_name(type(self.scorer))
        if not isinstance(scores, list | tuple) or len(scores) != expected:
            raise ConfigError(f"scorer {name} must return {expected} scores")
        checked = []
        for value in scores:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ConfigError(f"scorer {name} returned a score that is not a number")
            number = float(value)
            if not math.isfinite(number) or not -SCORE_SLACK <= number <= 1 + SCORE_SLACK:
                raise ConfigError(f"scorer {name} returned a score outside 0..1")
            checked.append(min(1.0, max(0.0, number)))
        logger.debug("step %s: scored %d texts with %s", self.step, expected, name)
        return checked
