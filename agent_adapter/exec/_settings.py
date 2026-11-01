"""`_Settings`: everything the per-call pipeline needs from an Executor."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent_adapter.core.ports import ApprovalPort, CachePort, EventSink, LedgerPort
from agent_adapter.exec.guards import RunLimits, _RunTracker
from agent_adapter.exec.retry import Backoff
from agent_adapter.hooks.base import Hook


@dataclass(frozen=True)
class _Settings:
    """Everything the pipeline needs from an Executor."""

    hooks: tuple[Hook, ...]
    approval: ApprovalPort | None
    cache: CachePort | None
    ledger: LedgerPort | None
    sink: EventSink | None
    limits: RunLimits
    backoff: Backoff
    namespace: str
    clock: Callable[[], float]
    sleep: Callable[[float], Awaitable[None]]
    rng: Callable[[], float]
    runs: _RunTracker
