"""The in-memory cost ledger: prices entries, keeps them per run, forwards them to sinks."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

from agent_adapter.core.errors import ConfigError, ToolUsageError, log_failure
from agent_adapter.core.types import LedgerEntry, Usage
from agent_adapter.cost.prices import Prices
from agent_adapter.cost.sinks import LedgerSink
from agent_adapter.cost.summary import LedgerSummary, summarise

logger = logging.getLogger(__name__)


def _entry_date(entry: LedgerEntry) -> date | None:
    """The UTC day the call started; None (today) when the start time is unknown."""
    if entry.started_at <= 0:
        return None
    return datetime.fromtimestamp(entry.started_at, UTC).date()


def _usage(entry: LedgerEntry) -> Usage:
    return Usage(
        input_tokens=entry.input_tokens,
        cached_input_tokens=entry.cached_input_tokens,
        cache_write_input_tokens=entry.cache_write_input_tokens,
        cache_write_by_ttl=entry.cache_write_by_ttl,
        output_tokens=entry.output_tokens,
        reasoning_tokens=entry.reasoning_tokens,
        units=entry.units,
    )


class Ledger:
    """A `LedgerPort`: idempotent on (run_id, entry_id); keeps the `max_runs` latest runs."""

    def __init__(
        self,
        *,
        prices: Prices | None = None,
        sinks: Sequence[LedgerSink] = (),
        max_runs: int = 1000,
    ) -> None:
        if max_runs < 1:
            raise ConfigError(f"max_runs must be >= 1, got {max_runs}")
        self._prices = prices if prices is not None else Prices.default()
        self._sinks = tuple(sinks)
        self._max_runs = max_runs
        self._runs: OrderedDict[str, dict[str, LedgerEntry]] = OrderedDict()
        self._lock = threading.Lock()

    def __reduce__(self) -> tuple[Any, ...]:
        """Refuse pickling with a clear error (it holds locks and loop-bound state).

        Raises:
            ToolUsageError: always.
        """
        raise ToolUsageError(
            f"{type(self).__name__} cannot be pickled: build it on the worker with a "
            "module-level functools.cache factory (see docs/backends/pipelines.md)"
        )

    @property
    def prices(self) -> Prices:
        """The price table used for entries recorded without a cost."""
        return self._prices

    @property
    def sinks(self) -> tuple[LedgerSink, ...]:
        """Sinks every recorded entry is forwarded to, in order."""
        return self._sinks

    @property
    def max_runs(self) -> int:
        """How many runs the in-memory index keeps."""
        return self._max_runs

    def record(self, entry: LedgerEntry) -> None:
        """Price `entry` if its cost is unknown, store it (replacing a same-id entry), fan out."""
        try:
            priced = self._priced(entry)
            self._store(priced)
        except Exception as exc:
            # The ledger port never raises: a cost-tracking fault must not fail the call.
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "ledger failed to record %s (run %s)",
                entry.entry_id,
                entry.run_id,
            )
            return
        for sink in self._sinks:
            try:
                sink.write(priced)
            except Exception as exc:
                log_failure(
                    logger,
                    logging.WARNING,
                    exc,
                    "ledger sink %s failed for %s (run %s)",
                    type(sink).__name__,
                    priced.entry_id,
                    priced.run_id,
                )

    def entries(
        self, run_id: str | None = None, *, thread_id: str | None = None
    ) -> list[LedgerEntry]:
        """Entries by run, least recently recorded run first; first-record order within a run."""
        with self._lock:
            if run_id is None:
                found = [e for run in self._runs.values() for e in run.values()]
            else:
                found = list(self._runs.get(run_id, {}).values())
        if thread_id is None:
            return found
        return [e for e in found if e.thread_id == thread_id]

    def total_usd(self, run_id: str | None = None, *, thread_id: str | None = None) -> float:
        """Sum of known costs; unknown costs count as 0."""
        return sum(e.cost_usd or 0.0 for e in self.entries(run_id, thread_id=thread_id))

    def summary(self, run_id: str | None = None) -> LedgerSummary:
        """Cost, cache and parallelism summary for one run, or for every run kept."""
        return summarise(self.entries(run_id), run_id)

    def clear(self, run_id: str | None = None) -> None:
        """Forget one run's entries, or everything (sinks are not touched)."""
        with self._lock:
            if run_id is None:
                self._runs.clear()
            else:
                self._runs.pop(run_id, None)

    def _priced(self, entry: LedgerEntry) -> LedgerEntry:
        if entry.cost_usd is not None:
            return entry
        if entry.kind == "llm" and entry.cache_hit:
            # Served from the response cache: no provider call, nothing billed, no price used.
            return entry.model_copy(update={"cost_usd": 0.0})
        try:
            cost = self._cost(entry)
        except Exception as exc:
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "pricing failed for %s %s (run %s); cost left unknown",
                entry.kind,
                entry.model,
                entry.run_id,
            )
            return entry
        if cost is None:
            return entry
        return entry.model_copy(update={"cost_usd": cost, "price_version": self._prices.version})

    def _cost(self, entry: LedgerEntry) -> float | None:
        if entry.kind == "llm":
            if not entry.model:
                return None
            return self._prices.cost(
                entry.model,
                _usage(entry),
                at=_entry_date(entry),
                seconds=entry.latency_ms / 1000,  # instrumented model calls arrive priced
                attempts=max(1, entry.attempts),  # each attempt is a billed request
            )
        if entry.kind == "api":
            item = f"{entry.provider}:{entry.model}"
            return self._prices.unit_cost(item, entry.units, at=_entry_date(entry))
        return None

    def _store(self, entry: LedgerEntry) -> None:
        with self._lock:
            run = self._runs.setdefault(entry.run_id, {})
            run[entry.entry_id] = entry
            self._runs.move_to_end(entry.run_id)
            while len(self._runs) > self._max_runs:
                evicted, _ = self._runs.popitem(last=False)
                logger.debug("ledger evicted run %s (max_runs=%d)", evicted, self._max_runs)
