"""Batches: the Executor's many-call methods (`run_many`, `as_completed`, plans, `parallel`)."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Sequence
from contextlib import aclosing
from typing import Literal, Protocol, TypeVar, overload

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ToolCall, ToolResult
from agent_adapter.exec._settings import _Settings
from agent_adapter.exec.parallel import (
    OnError,
    _check_concurrency,
    _check_on_error,
    _check_unique_ids,
    _close_unstarted,
    _run_batch,
    _run_parallel,
    _stream_completed,
)
from agent_adapter.exec.plan import Step, _run_waves

T = TypeVar("T")


class _Core(Protocol):
    """The Executor state and helpers the batch methods use."""

    _settings: _Settings

    def _run_context(self, ctx: ToolContext | None) -> tuple[ToolContext, bool]: ...

    def _registry(self, registry: Registry | None) -> Registry: ...

    def end_run(self, run_id: str) -> None: ...


async def _batch(
    core: _Core,
    calls: list[ToolCall],
    ctx: ToolContext,
    concurrency: int,
    on_error: OnError,
    registry: Registry | None,
) -> list[ToolResult]:
    return await _run_batch(
        core._settings, calls, ctx, concurrency=concurrency, on_error=on_error, registry=registry
    )


async def _as_completed(
    core: _Core,
    calls: list[ToolCall],
    ctx: ToolContext | None,
    concurrency: int,
    registry: Registry | None,
) -> AsyncGenerator[ToolResult, None]:
    run_ctx, owned = core._run_context(ctx)
    stream = _stream_completed(
        core._settings, calls, run_ctx, concurrency=concurrency, registry=registry
    )
    try:
        async with aclosing(stream):
            async for result in stream:
                yield result
    finally:
        if owned:
            core.end_run(run_ctx.run_id)


class Batches:
    """`run_many`, `as_completed`, `run_plan` and `parallel` for the Executor."""

    async def run_many(
        self: _Core,
        calls: Iterable[ToolCall],
        ctx: ToolContext | None = None,
        *,
        concurrency: int | None = None,
        on_error: OnError = "collect",
        registry: Registry | None = None,
    ) -> list[ToolResult]:
        """Run independent calls concurrently; results keep input order.

        `ctx.approval` covers none of the calls; each asks the approval port.
        """
        mode = _check_on_error(on_error)
        limit = _check_concurrency(concurrency, self._settings.limits.max_concurrency)
        batch = list(calls)
        ctx, owned = self._run_context(ctx)
        try:
            return await _batch(self, batch, ctx, limit, mode, self._registry(registry))
        finally:
            if owned:
                self.end_run(ctx.run_id)

    def as_completed(
        self: _Core,
        calls: Iterable[ToolCall],
        ctx: ToolContext | None = None,
        *,
        concurrency: int | None = None,
        registry: Registry | None = None,
    ) -> AsyncGenerator[ToolResult, None]:
        """Yield each call's ToolResult as it finishes; leaving early cancels the rest.

        `ctx.approval` covers none of the calls; each asks the approval port.
        """
        limit = _check_concurrency(concurrency, self._settings.limits.max_concurrency)
        batch = list(calls)
        _check_unique_ids(batch)
        return _as_completed(self, batch, ctx, limit, self._registry(registry))

    async def run_plan(
        self: _Core,
        steps: Sequence[Step],
        ctx: ToolContext | None = None,
        *,
        concurrency: int | None = None,
        on_error: OnError = "collect",
        registry: Registry | None = None,
    ) -> dict[str, ToolResult]:
        """Run steps in dependency waves; returns {step.id: ToolResult} in step order.

        Raises:
            PlanInvalid: duplicate ids, unknown dependencies or a cycle, before anything runs.
        """
        mode = _check_on_error(on_error)
        limit = _check_concurrency(concurrency, self._settings.limits.max_concurrency)
        tools = self._registry(registry)
        ctx, owned = self._run_context(ctx)
        try:
            return await _run_waves(
                steps,
                lambda calls: _batch(self, calls, ctx, limit, mode, tools),
                on_error=mode,
                run_id=ctx.run_id,
            )
        finally:
            if owned:
                self.end_run(ctx.run_id)

    @overload
    async def parallel(
        self: _Core,
        *work: Callable[[], Awaitable[T]] | Awaitable[T],
        concurrency: int | None = None,
        timeout: float | None = None,
        retries: int = 0,
        on_error: Literal["fail_fast"] = "fail_fast",
    ) -> list[T]: ...

    @overload
    async def parallel(
        self: _Core,
        *work: Callable[[], Awaitable[T]] | Awaitable[T],
        concurrency: int | None = None,
        timeout: float | None = None,
        retries: int = 0,
        on_error: Literal["collect"],
    ) -> list[T | Exception]: ...

    async def parallel(
        self: _Core,
        *work: Callable[[], Awaitable[T]] | Awaitable[T],
        concurrency: int | None = None,
        timeout: float | None = None,
        retries: int = 0,
        on_error: OnError = "fail_fast",
    ) -> list[T] | list[T | Exception]:
        """Run non-tool work concurrently with timeout and retries; results keep input order.

        Raises:
            ToolUsageError: bad options, or `retries > 0` with an awaitable (not retryable).
        """
        settings = self._settings
        try:
            mode = _check_on_error(on_error)
            limit = _check_concurrency(concurrency, settings.limits.max_concurrency)
        except ToolUsageError:
            _close_unstarted(work)
            raise
        return await _run_parallel(
            work,
            concurrency=limit,
            timeout=timeout,
            retries=retries,
            on_error=mode,
            backoff=settings.backoff,
            rng=settings.rng,
            sleep=settings.sleep,
        )
