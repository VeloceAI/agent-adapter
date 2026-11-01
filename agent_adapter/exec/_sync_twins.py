"""SyncTwins: blocking versions of the Executor's async entry points, for sync code."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision, ToolCall, ToolResult
from agent_adapter.exec._sugar import ToolRef, blocking
from agent_adapter.exec.parallel import OnError


class _AsyncRunner(Protocol):
    """The async methods the twins block on (the Executor's)."""

    async def run(
        self,
        tool: ToolRef,
        args: Mapping[str, Any] | None = None,
        ctx: ToolContext | None = None,
        *,
        call_id: str | None = None,
    ) -> Any: ...

    async def run_call(
        self, call: ToolCall, ctx: ToolContext | None = None, *, registry: Registry | None = None
    ) -> ToolResult: ...

    async def run_many(
        self,
        calls: Iterable[ToolCall],
        ctx: ToolContext | None = None,
        *,
        concurrency: int | None = None,
        on_error: OnError = "collect",
        registry: Registry | None = None,
    ) -> list[ToolResult]: ...

    async def resume(
        self,
        approval_id: str,
        decision: bool | ApprovalDecision | None = None,
        ctx: ToolContext | None = None,
        *,
        decided_by: str = "",
        reason: str = "",
    ) -> ToolResult: ...


class SyncTwins:
    """`run_sync`, `run_call_sync`, `run_many_sync` and `resume_sync` for the Executor.

    Each blocks until done on one long-lived loop shared by the process (a
    `runtime.LoopRunner`), so loop-bound state such as clients and semaphores stays valid
    across calls. In async code, await the async method instead.
    """

    def run_sync(
        self: _AsyncRunner,
        tool: ToolRef,
        args: Mapping[str, Any] | None = None,
        ctx: ToolContext | None = None,
        *,
        call_id: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        """`run` from sync code; returns the value and raises like `run`.

        Raises:
            TimeoutError: past `timeout` seconds.
            ToolUsageError: called from a tool already running on the shared loop.
        """
        return blocking(lambda: self.run(tool, args, ctx, call_id=call_id), timeout)

    def run_call_sync(
        self: _AsyncRunner,
        call: ToolCall,
        ctx: ToolContext | None = None,
        *,
        registry: Registry | None = None,
        timeout: float | None = None,
    ) -> ToolResult:
        """`run_call` from sync code; failures are in the `ToolResult`."""
        return blocking(lambda: self.run_call(call, ctx, registry=registry), timeout)

    def run_many_sync(
        self: _AsyncRunner,
        calls: Iterable[ToolCall],
        ctx: ToolContext | None = None,
        *,
        concurrency: int | None = None,
        on_error: OnError = "collect",
        registry: Registry | None = None,
        timeout: float | None = None,
    ) -> list[ToolResult]:
        """`run_many` from sync code; results keep input order."""
        batch = list(calls)
        return blocking(
            lambda: self.run_many(
                batch, ctx, concurrency=concurrency, on_error=on_error, registry=registry
            ),
            timeout,
        )

    def resume_sync(
        self: _AsyncRunner,
        approval_id: str,
        decision: bool | ApprovalDecision | None = None,
        ctx: ToolContext | None = None,
        *,
        decided_by: str = "",
        reason: str = "",
        timeout: float | None = None,
    ) -> ToolResult:
        """`resume` from sync code: decide a parked call and run it once."""
        return blocking(
            lambda: self.resume(approval_id, decision, ctx, decided_by=decided_by, reason=reason),
            timeout,
        )
