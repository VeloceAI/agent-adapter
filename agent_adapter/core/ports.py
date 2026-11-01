"""Port protocols: what the adapter needs from caches, ledgers, sinks, approvers and models."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable

from agent_adapter.core.messages import LLMRequest, LLMResponse, StreamEvent
from agent_adapter.core.types import (
    ApprovalDecision,
    ApprovalRequest,
    Event,
    LedgerEntry,
    ModelCaps,
)


@runtime_checkable
class CachePort(Protocol):
    """Byte cache. Implementations log backend failures and act as a miss; never raise."""

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...


@runtime_checkable
class LedgerPort(Protocol):
    """Cost ledger; `record` is idempotent on (run_id, entry_id) and never raises.

    `run_id` and `thread_id` filter together; neither means every entry.
    """

    def record(self, entry: LedgerEntry) -> None: ...

    def total_usd(  # unknown costs count as 0
        self, run_id: str | None = None, *, thread_id: str | None = None
    ) -> float: ...

    def entries(
        self, run_id: str | None = None, *, thread_id: str | None = None
    ) -> list[LedgerEntry]: ...


@runtime_checkable
class EventSink(Protocol):
    """Receives trace events; `emit` is sync, non-blocking and never raises."""

    def emit(self, event: Event) -> None: ...


@runtime_checkable
class ApprovalPort(Protocol):
    """Decides whether a tool call may run; may raise ApprovalPending."""

    async def request(self, request: ApprovalRequest) -> ApprovalDecision: ...


@runtime_checkable
class LLMPort(Protocol):
    """A model provider; `generate` raises ModelError subclasses."""

    provider: str
    model: str

    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    # An async generator (`async def` + `yield`) whose last item is StreamDone.
    def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]: ...

    def cache_key_fields(self) -> dict[str, Any]: ...  # extra LLM-cache key material


CapabilityLookup = Callable[[str], ModelCaps]


@runtime_checkable
class PromptCachePlugin(Protocol):
    """Edits provider SDK kwargs to use that provider's prompt cache."""

    name: str
    providers: frozenset[str]

    def key_fields(self) -> dict[str, Any]: ...

    async def apply(
        self, request: LLMRequest, kwargs: dict[str, Any], *, model: str, client: Any
    ) -> dict[str, Any]: ...
