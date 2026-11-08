"""Price a model call for a budget: an estimate before it runs, the real cost after."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any, Protocol

from agent_adapter.core.errors import log_failure
from agent_adapter.core.messages import LLMRequest
from agent_adapter.core.types import LedgerEntry, Usage

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # a rough, provider-neutral prompt size for the estimate
DEFAULT_OUTPUT_TOKENS = 1024  # assumed when the request sets no max_output_tokens


class PriceTable(Protocol):
    """What a budget needs of `agent_adapter.cost.Prices`."""

    def cost(
        self,
        model: str,
        usage: Usage,
        *,
        at: date | None = ...,
        seconds: float = ...,
        attempts: int = ...,
    ) -> float | None: ...


def prices_of(own: PriceTable | None, ledger: Any) -> PriceTable | None:
    """The hook's own table, else the ledger's (`Ledger.prices`), else None."""
    if own is not None:
        return own
    found = getattr(ledger, "prices", None)
    return found if callable(getattr(found, "cost", None)) else None


def estimate_usd(prices: PriceTable | None, model: str, request: LLMRequest, now: float) -> float:
    """A conservative cost of `request` on `model`: prompt chars / 4 in, max output out."""
    if prices is None or not model:
        return 0.0
    chars = sum(len(str(message.content or "")) for message in request.messages)
    usage = Usage(
        input_tokens=max(1, chars // CHARS_PER_TOKEN),
        output_tokens=request.max_output_tokens or DEFAULT_OUTPUT_TOKENS,
    )
    return _priced(prices, model, usage, now, 0.0) or 0.0


def billed_usd(prices: PriceTable | None, entry: LedgerEntry) -> float | None:
    """What the ledger bills for `entry` (cache hits and hook replies are $0); None if unpriced."""
    if entry.cost_usd is not None:
        return entry.cost_usd
    if prices is None or not entry.model:
        return None
    usage = Usage(
        input_tokens=entry.input_tokens,
        cached_input_tokens=entry.cached_input_tokens,
        cache_write_input_tokens=entry.cache_write_input_tokens,
        cache_write_by_ttl=entry.cache_write_by_ttl,
        output_tokens=entry.output_tokens,
        reasoning_tokens=entry.reasoning_tokens,
        units=entry.units,
    )
    started = entry.started_at if entry.started_at > 0 else None
    seconds = entry.latency_ms / 1000
    return _priced(prices, entry.model, usage, started, seconds, max(1, entry.attempts))


def _priced(
    prices: PriceTable,
    model: str,
    usage: Usage,
    now: float | None,
    seconds: float,
    attempts: int = 1,
) -> float | None:
    at = None if now is None else _day(now)
    try:
        cost = prices.cost(model, usage, at=at, seconds=max(0.0, seconds), attempts=attempts)
    except Exception as exc:
        # A pricing fault must not fail the call; the budget then counts the call as unpriced.
        log_failure(logger, logging.WARNING, exc, "budget pricing failed for model %s", model)
        return None
    if cost is None or cost < 0:
        return None
    return float(cost)


def _day(now: float) -> date:
    return datetime.fromtimestamp(now, UTC).date()
