"""Ledger entries of AI compression steps: ids, recording and the cost of a summary call."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from agent_adapter.context.step import StepContext
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import log_failure
from agent_adapter.core.ports import LedgerPort
from agent_adapter.core.types import LedgerEntry, Usage
from agent_adapter.cost.prices import Prices
from agent_adapter.runtime.current import resolve_context

logger = logging.getLogger(__name__)

STAGE = "compress"
SEQ_KEY = "aa.compress_seq"  # ctx.extras: the next compression entry number of this context
PROVIDER = "agent_adapter"  # provider of the entries that record scoring, not a model call


def run_context(context: StepContext) -> ToolContext:
    """The run a step's model and scorer calls belong to: the bound, current or a fresh one."""
    return resolve_context(context.ctx)


def next_entry_id(ctx: ToolContext) -> str:
    """`{call or run id}:compress:{n}`, numbered per context."""
    current = ctx.extras.get(SEQ_KEY)
    number = current if isinstance(current, int) and not isinstance(current, bool) else 0
    ctx.extras[SEQ_KEY] = number + 1
    return f"{ctx.call_id or ctx.run_id}:{STAGE}:{number}"


def record(ledger: LedgerPort | None, entry: LedgerEntry) -> None:
    """Record `entry`; a failing ledger is logged and never fails the compression."""
    if ledger is None:
        return
    try:
        ledger.record(entry)
    except Exception as exc:
        log_failure(logger, logging.WARNING, exc, "ledger.record failed for %s", entry.entry_id)


def find(ledger: LedgerPort | None, run_id: str, entry_id: str) -> LedgerEntry | None:
    """The entry `entry_id` of run `run_id`, if `ledger` has it."""
    if ledger is None:
        return None
    try:
        entries = ledger.entries(run_id)
    except Exception as exc:
        log_failure(logger, logging.WARNING, exc, "ledger.entries failed for run %s", run_id)
        return None
    return next((e for e in entries if e.entry_id == entry_id), None)


@dataclass
class SummaryEntry:
    """One summariser call's ledger entry; skipped when an instrumented model wrote it already."""

    ctx: ToolContext
    entry_id: str
    step: str
    provider: str
    model: str
    started_at: float = field(default_factory=time.time)
    _started: float = field(default_factory=time.perf_counter)

    def settle(
        self,
        usage: Usage,
        *,
        model: str = "",
        ok: bool = True,
        error_kind: str = "",
        cache_hit: bool = False,
    ) -> float:
        """Record the entry unless present; its USD cost (0 when unknown or not ok)."""
        ledger = self.ctx.ledger
        entry = find(ledger, self.ctx.run_id, self.entry_id)  # an instrumented model's own
        if entry is None:
            entry = self._entry(usage, model or self.model, ok, error_kind, cache_hit)
            if ledger is None:
                return _default_price(entry, usage) if ok and not cache_hit else 0.0
            record(ledger, entry)
            entry = find(ledger, self.ctx.run_id, self.entry_id) or entry
        if not ok or entry.cost_usd is None:
            return 0.0
        return entry.cost_usd

    def _entry(
        self, usage: Usage, model: str, ok: bool, error_kind: str, cache_hit: bool
    ) -> LedgerEntry:
        metadata: Mapping[str, str] = {"step": self.step}
        if self.ctx.call_id:
            metadata = {**metadata, "parent_call_id": self.ctx.call_id}
        return LedgerEntry.from_usage(
            usage,
            kind="llm",
            run_id=self.ctx.run_id,
            call_id=self.entry_id,
            entry_id=self.entry_id,
            stage=STAGE,
            provider=self.provider,
            model=model,
            cost_usd=0.0 if cache_hit else None,
            latency_ms=(time.perf_counter() - self._started) * 1000,
            started_at=self.started_at,
            ended_at=time.time(),
            ok=ok,
            error_kind=error_kind,
            cache_hit=cache_hit,
            thread_id=self.ctx.thread_id,
            metadata=dict(metadata),
        )


def scoring_entry(
    ctx: ToolContext,
    *,
    step: str,
    scorer: str,
    calls: int,
    texts: int,
    started_at: float,
    error_kind: str = "",
) -> LedgerEntry:
    """A $0 entry saying a step scored `texts` texts; the scorer's own port bills its calls."""
    entry_id = next_entry_id(ctx)
    return LedgerEntry(
        ok=not error_kind,
        error_kind=error_kind,
        kind="api",
        run_id=ctx.run_id,
        call_id=entry_id,
        entry_id=entry_id,
        stage=STAGE,
        provider=PROVIDER,
        model=f"context:{step}",
        cost_usd=0.0,
        started_at=started_at,
        ended_at=time.time(),
        latency_ms=(time.time() - started_at) * 1000,
        thread_id=ctx.thread_id,
        metadata={"step": step, "scorer": scorer, "calls": str(calls), "texts": str(texts)},
    )


def _default_price(entry: LedgerEntry, usage: Usage) -> float:
    """The call's cost by the default price table, for a run without a ledger; 0 if unpriced."""
    try:
        cost = Prices.default().cost(entry.model, usage, seconds=entry.latency_ms / 1000)
    except Exception as exc:
        log_failure(logger, logging.WARNING, exc, "pricing failed for %s", entry.entry_id)
        return 0.0
    return cost or 0.0
