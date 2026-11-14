"""Bill what is not a model or a tool call: call minutes, speech-to-text seconds, TTS characters."""

from __future__ import annotations

import logging
import time

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolUsageError, log_failure
from agent_adapter.core.ports import LedgerPort
from agent_adapter.core.types import LedgerEntry
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)


def record_units(
    ledger: LedgerPort | None, ctx: ToolContext, item: str, units: float
) -> LedgerEntry | None:
    """Record `units` of `item` ("provider:name") for the run; the ledger's unit price bills it.

    Returns the entry recorded, or None without a ledger.

    Raises:
        ToolUsageError: `item` has no "provider:" part, or `units` is negative.
    """
    provider, _, name = item.partition(":")
    if not provider or not name:
        raise ToolUsageError(f"a metered item is 'provider:name', got {item!r}")
    if units < 0:
        raise ToolUsageError(f"metered units must be >= 0, got {units!r}")
    if ledger is None:
        logger.debug("no ledger: %s units of %s not recorded", units, item)
        return None
    now = time.time()
    entry = LedgerEntry(
        kind="api",
        run_id=ctx.run_id,
        call_id=ctx.call_id,
        entry_id=new_id("meter_"),
        stage=item,
        provider=provider,
        model=name,
        units=units,
        started_at=now,
        ended_at=now,
        thread_id=ctx.thread_id,
    )
    try:
        ledger.record(entry)
    except Exception as exc:
        log_failure(logger, logging.WARNING, exc, "ledger.record failed for %s", item)
    return entry
