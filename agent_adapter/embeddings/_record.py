"""One metered embed (or rerank) call's ledger entry id, ledger entry and trace events."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import log_failure
from agent_adapter.core.ports import EventSink, LedgerPort
from agent_adapter.core.types import ErrorInfo, LedgerEntry, Usage
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.tracing.sinks import first_sink

logger = logging.getLogger(__name__)

SEQ_KEY = "aa.embed_seq"  # ctx.extras: the next embed entry number of this context
STAGE = "embed"
CANCELLED = "cancelled"
_PER_UNIT = 1_000_000  # ledger units are millions of tokens, as unit prices are


@dataclass(frozen=True)
class CallKind:
    """What a metered call is: its entry-id label (also its default stage) and event types."""

    label: str
    started: str
    finished: str
    failed: str


EMBED = CallKind(
    STAGE, EventType.EMBEDDING_STARTED, EventType.EMBEDDING_FINISHED, EventType.EMBEDDING_FAILED
)
RERANK = CallKind(
    "rerank", EventType.RERANK_STARTED, EventType.RERANK_FINISHED, EventType.RERANK_FAILED
)


def next_entry_id(ctx: ToolContext, label: str = STAGE) -> str:
    """`{call or run id}:{label}:{n}`, numbered per context and label."""
    key = SEQ_KEY if label == STAGE else f"aa.{label}_seq"
    current = ctx.extras.get(key)
    number = current if isinstance(current, int) and not isinstance(current, bool) else 0
    ctx.extras[key] = number + 1
    return f"{ctx.call_id or ctx.run_id}:{label}:{number}"


def billed_units(usage: Usage) -> float:
    """The fn's own `units`, else input tokens in millions (the unit-price unit)."""
    return usage.units if usage.units else usage.input_tokens / _PER_UNIT


class EmbedCall:
    """One metered call: collects batch usage, records one ledger entry, emits its events.

    `kind` picks the entry-id label and events (embed by default); `stage` defaults to the label.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        ledger: LedgerPort | None,
        sink: EventSink | None,
        ctx: ToolContext,
        texts: int,
        kind: CallKind = EMBED,
        stage: str = "",
    ) -> None:
        self.kind = kind
        self.stage = stage or kind.label
        self.provider = provider
        self.model = model
        self.ctx = ctx
        self.ledger = ledger if ledger is not None else ctx.ledger
        self.sink = first_sink(sink, ctx.sink)
        self.entry_id = next_entry_id(ctx, kind.label)
        self.texts = texts
        self.usage = Usage()
        self.billed_model = ""  # the model name the fn reported, if any
        self.batches = 0
        self.cache_hits = 0
        self._started = time.perf_counter()
        self._started_at = time.time()

    def started(self) -> None:
        """Emit `embedding.started`."""
        self._emit(self.kind.started)

    def paid(self, usage: Usage, model: str) -> None:
        """Add one finished batch."""
        self.batches += 1
        self.usage = self.usage + usage
        self.billed_model = self.billed_model or model

    def finished(self) -> None:
        """Record the entry and emit the finished event; a call served from the cache costs 0."""
        logger.debug(
            "%s %s (%s %s, run %s): %d texts, %d cached, %d batches",
            self.kind.label,
            self.entry_id,
            self.provider,
            self.model,
            self.ctx.run_id,
            self.texts,
            self.cache_hits,
            self.batches,
        )
        entry = self._entry(cost_usd=None if self.batches else 0.0)
        self._record(entry)
        self._emit(
            self.kind.finished,
            cache_hits=self.cache_hits,
            batches=self.batches,
            latency_ms=entry.latency_ms,
            usage=self.usage.model_dump(mode="json"),
            units=entry.units,
            cost_usd=self._cost(entry),
            response_model=entry.model,
        )

    def failed(self, exc: BaseException) -> None:
        """Record a failed entry and emit the failed event.

        Cancellation records an entry only after a paid batch.
        """
        if not isinstance(exc, Exception):
            kind = CANCELLED
            if self.batches:
                self._record(self._entry(cost_usd=None, ok=False, error_kind=kind))
        else:
            kind = ErrorInfo.from_exception(exc).kind
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "%s %s (%s %s, run %s) failed after %d batches: %s",
                self.kind.label,
                self.entry_id,
                self.provider,
                self.model,
                self.ctx.run_id,
                self.batches,
                kind,
            )
            cost = None if self.batches else 0.0
            self._record(self._entry(cost_usd=cost, ok=False, error_kind=kind))
        self._emit(
            self.kind.failed,
            error_kind=kind,
            latency_ms=(time.perf_counter() - self._started) * 1000,
        )

    def _entry(
        self, *, cost_usd: float | None, ok: bool = True, error_kind: str = ""
    ) -> LedgerEntry:
        metadata = {"parent_call_id": self.ctx.call_id} if self.ctx.call_id else {}
        metadata.update(
            texts=str(self.texts), batches=str(self.batches), cache_hits=str(self.cache_hits)
        )
        return LedgerEntry.from_usage(
            self.usage.model_copy(update={"units": billed_units(self.usage)}),
            kind="api",
            run_id=self.ctx.run_id,
            call_id=self.entry_id,
            entry_id=self.entry_id,
            stage=self.stage,
            provider=self.provider,
            model=self.billed_model or self.model,
            cost_usd=cost_usd,
            latency_ms=(time.perf_counter() - self._started) * 1000,
            started_at=self._started_at,
            ended_at=time.time(),
            ok=ok,
            error_kind=error_kind,
            cache_hit=self.batches == 0 and ok,
            thread_id=self.ctx.thread_id,
            metadata=metadata,
        )

    def _record(self, entry: LedgerEntry) -> None:
        ledger = self.ledger
        if ledger is None:
            return
        try:
            ledger.record(entry)
        except Exception as exc:
            # The port must not raise; a broken one must not fail the call it measures.
            log_failure(logger, logging.WARNING, exc, "ledger.record failed for %s", self.entry_id)

    def _emit(self, type: str, /, **data: Any) -> None:
        ids = {"provider": self.provider, "model": self.model, "stage": self.stage}
        event = make_event(type, self.ctx, call_id=self.entry_id, **ids, texts=self.texts, **data)
        emit(self.sink, event)

    def _cost(self, entry: LedgerEntry) -> float | None:
        """The entry's USD cost as the ledger prices it; None without a pricing ledger."""
        if entry.cost_usd is not None:
            return entry.cost_usd
        unit_cost = getattr(getattr(self.ledger, "prices", None), "unit_cost", None)
        if unit_cost is None:
            return None
        day = datetime.fromtimestamp(entry.started_at, UTC).date()
        try:
            value = unit_cost(f"{entry.provider}:{entry.model}", entry.units, at=day)
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "pricing failed for %s", self.entry_id)
            return None
        return value if isinstance(value, int | float) else None
