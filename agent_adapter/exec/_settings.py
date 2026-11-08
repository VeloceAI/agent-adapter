"""`_Settings`: everything the per-call pipeline needs from an Executor."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent_adapter.core.ports import ApprovalPort, CachePort, EventSink, LedgerPort
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec.guards import RunLimits, _RunTracker
from agent_adapter.exec.retry import Backoff
from agent_adapter.hooks.base import Hook


@runtime_checkable
class ToolPolicy(Protocol):
    """An executor setting no `hooks=` can drop: its hooks run first, its specs and limits win."""

    def hooks(self) -> tuple[Hook, ...]:
        """Hooks run before the executor's own, on every call (the access gate first)."""
        ...

    def limits(self, current: RunLimits) -> RunLimits:
        """`current` made no looser than the policy allows."""
        ...

    def effective(self, spec: ToolSpec) -> ToolSpec:
        """The spec the call runs with; never less strict than `spec`."""
        ...

    def describe(self, spec: ToolSpec) -> list[str]:
        """Short lines naming the rules that may apply to `spec`."""
        ...


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
    policy: ToolPolicy | None = None
