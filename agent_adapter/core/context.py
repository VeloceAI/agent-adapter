"""Per-call context injected into tools and hidden from the LLM."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.core.errors import log_failure
from agent_adapter.core.ports import CachePort, EventSink, LedgerPort
from agent_adapter.core.types import ApprovalDecision, Event

logger = logging.getLogger(__name__)

CUSTOM_PREFIX = "custom."


@dataclass
class ToolContext:
    """Run-scoped data a tool may read; never part of the tool's schema."""

    run_id: str = ""
    call_id: str = ""
    state: Mapping[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)  # keys starting "aa." are reserved
    thread_id: str = ""
    ledger: LedgerPort | None = None
    cache: CachePort | None = None
    sink: EventSink | None = None
    approval: ApprovalDecision | None = None  # pre-approval granted by a host framework

    def emit(self, type: str, /, **data: Any) -> None:
        """Send a `custom.<type>` event to the sink; a no-op without one."""
        if self.sink is None:
            return
        name = type if type.startswith(CUSTOM_PREFIX) else CUSTOM_PREFIX + type
        event = Event(name, run_id=self.run_id, call_id=self.call_id, data=data)
        try:
            self.sink.emit(event)
        except Exception as exc:
            # Sinks must never break a tool; a failing one is reported, not propagated.
            log_failure(
                logger, logging.WARNING, exc, "event sink failed for %s (run %s)", name, self.run_id
            )

    def child(self, *, call_id: str, **changes: Any) -> ToolContext:
        """A copy for a nested call; `extras` is copied and `approval` is not inherited."""
        changes.setdefault("extras", dict(self.extras))
        changes.setdefault("approval", None)
        return dataclasses.replace(self, call_id=call_id, **changes)
