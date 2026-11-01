"""What one OpenAPI import left out or renamed, summarised in one WARNING per kind."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from agent_adapter.integrations.openapi._refs import FIELDS_PER_VALUE, MAX_FIELDS, MIN_FIELDS

_SHOWN = 5  # entries named in each summary


@dataclass
class ImportReport:
    """What one import left out or renamed, logged as one summary WARNING per kind."""

    fields_budget: int  # top-level input fields all tools may have
    fields: int = 0
    over_budget: bool = False  # an operation was skipped for the input-fields budget
    ids: Counter[str] = field(default_factory=Counter)  # operationId -> operations using it
    skipped: list[str] = field(default_factory=list)  # "key: reason"
    _skipped_keys: set[str] = field(default_factory=set)
    dropped: list[str] = field(default_factory=list)  # keys whose response schema was dropped

    def skip(self, key: str, reason: str) -> None:
        """Record that the operation `key` is left out, and why."""
        self.skipped.append(f"{key}: {reason}")
        self._skipped_keys.add(key)

    def skipped_keys(self) -> set[str]:
        """The keys of the skipped operations."""
        return set(self._skipped_keys)

    def fits(self, key: str, count: int) -> bool:
        """Charge an operation's `count` input fields; False (and skipped) when over budget."""
        if self.fields + count > self.fields_budget:
            self.skip(key, f"{count} input fields over the budget")
            self.over_budget = True
            return False
        self.fields += count
        return True

    def log(self, logger: logging.Logger) -> None:
        """One WARNING per kind on `logger`: duplicated ids, skipped operations, dropped schemas."""
        duplicated = [key for key, count in self.ids.items() if count > 1]
        if duplicated:
            logger.warning(
                "%d OpenAPI operationIds are used by more than one operation (their tools "
                "get numbered names): %s",
                len(duplicated),
                _shown(duplicated, ", "),
            )
        if self.skipped:
            budget = (
                f" (the input-fields budget for all tools is {self.fields_budget}: "
                f"{FIELDS_PER_VALUE} per document value, {MIN_FIELDS} to {MAX_FIELDS})"
                if self.over_budget
                else ""
            )
            shown = _shown(self.skipped, "; ")
            logger.warning("%d OpenAPI operations skipped: %s%s", len(self.skipped), shown, budget)
        if self.dropped:
            shown = _shown(self.dropped, ", ")
            logger.warning(
                "%d OpenAPI response schemas dropped (too large): %s", len(self.dropped), shown
            )


def _shown(entries: list[str], separator: str) -> str:
    more = f"{separator}..." if len(entries) > _SHOWN else ""
    return separator.join(entries[:_SHOWN]) + more
