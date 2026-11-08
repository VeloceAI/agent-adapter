"""PresidioGuard: Microsoft Presidio's analyzer finds PII, its anonymizer redacts it."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.hooks.builtin.guardrail import Stage
from agent_adapter.hooks.builtin.guards.base import DEFAULT_STAGES, Action, GuardVerdict, TextGuard

INSTALL = 'pip install "agent-adapter[presidio]"'


class PresidioGuard(TextGuard):
    """Presidio as a guard; pass your own `analyzer` and `anonymizer` engines or get defaults."""

    name = "presidio_guard"

    def __init__(
        self,
        *,
        analyzer: Any = None,
        anonymizer: Any = None,
        entities: Iterable[str] | None = None,
        language: str = "en",
        threshold: float = 0.5,
        on: Iterable[Stage] | Stage = DEFAULT_STAGES,
        action: Action = "block",
    ) -> None:
        super().__init__(on=on, action=action)
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(f"threshold must be between 0 and 1, got {threshold}")
        self.analyzer = analyzer if analyzer is not None else _default_analyzer()
        self.anonymizer = anonymizer
        if anonymizer is None and action == "redact":
            self.anonymizer = _default_anonymizer()
        self.entities = list(entities) if entities is not None else None
        self.language = language
        self.threshold = threshold

    async def review(self, text: str, stage: str) -> GuardVerdict:
        return await asyncio.to_thread(self._review, text)

    def _review(self, text: str) -> GuardVerdict:
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.threshold,
        )
        if not results:
            return GuardVerdict(True)
        types = sorted({str(r.entity_type) for r in results})
        reason = "found " + ", ".join(types)
        if self.anonymizer is None:
            return GuardVerdict(False, reason)
        redacted = self.anonymizer.anonymize(text=text, analyzer_results=results)
        kinds = tuple(kind.lower() for kind in types)
        return GuardVerdict(False, reason, str(redacted.text), kinds=kinds, count=len(results))


def _default_analyzer() -> Any:
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError as exc:
        raise MissingDependency(f"PresidioGuard needs presidio-analyzer: {INSTALL}") from exc
    return AnalyzerEngine()


def _default_anonymizer() -> Any:
    try:
        from presidio_anonymizer import AnonymizerEngine
    except ImportError as exc:
        raise MissingDependency(f"PresidioGuard needs presidio-anonymizer: {INSTALL}") from exc
    return AnonymizerEngine()
