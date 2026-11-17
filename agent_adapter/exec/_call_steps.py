"""One call's cache step (lookup, store) and its success result and closing event."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel

from agent_adapter.cache._inflight import NOTHING, share
from agent_adapter.cache.codec import decode, encode
from agent_adapter.cache.keys import digest
from agent_adapter.cache.tool_cache import ToolResultCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolResult
from agent_adapter.exec._settings import _Settings
from agent_adapter.exec.guards import _RunState
from agent_adapter.hooks.base import ToolInvocation
from agent_adapter.tracing.events import EventType

NOT_STORED = object()  # step 11 sentinel: nothing to put in the cache


class _Run(Protocol):
    """The state of one call these steps read."""

    settings: _Settings
    ctx: ToolContext
    name: str
    call_id: str
    attempts: int

    def emit(self, type: str, /, **data: Any) -> None: ...

    def emit_finished(self, value: Any, /, **data: Any) -> None: ...

    def latency_ms(self) -> float: ...


class _Steps(_Run, Protocol):
    landing: Callable[[], None] | None

    async def _lookup(self, spec: ToolSpec, inv: ToolInvocation) -> Any: ...

    async def _run(
        self, spec: ToolSpec, state: _RunState, validated: BaseModel, inv: ToolInvocation
    ) -> tuple[ToolResult, Any]: ...

    def _ok(self, value: Any, *, cached: bool = False) -> ToolResult: ...


class CallSteps:
    """Cache lookup and store, the ok result and the finishing event of one call."""

    landing: Callable[[], None] | None = None  # releases a shared call once it is stored

    async def _cached_or_run(
        self: _Steps, spec: ToolSpec, state: _RunState, validated: BaseModel, inv: ToolInvocation
    ) -> tuple[ToolResult, Any]:
        """A cache hit, else the tool's run, shared with callers of the same key meanwhile."""
        cached = await self._lookup(spec, inv)
        if cached is not NOT_STORED:
            return self._ok(cached, cached=True), NOT_STORED
        cache = _result_cache(self, spec)
        key = "" if cache is None else _key(cache, spec, inv, self.ctx)

        def follow(payload: bytes, waited_ms: float) -> tuple[ToolResult, Any]:
            self.emit(EventType.TOOL_CACHED, tool=spec.name)
            self.emit(EventType.INFLIGHT_SHARED, kind="tool", name=spec.name, waited_ms=waited_ms)
            envelope = decode(payload, "tool")  # each waiter gets its own copy, as from a hit
            return self._ok(envelope.payload if envelope else None, cached=True), NOT_STORED

        def hold(release: Callable[[], None]) -> None:
            self.landing = release

        return await share(
            self.ctx.cache,
            key,
            lambda: self._run(spec, state, validated, inv),
            handout=_handout,
            follow=follow,
            hold=hold,
        )

    def land(self) -> None:
        """End this call's share: later callers read the cache (the store is done)."""
        release, self.landing = self.landing, None
        if release is not None:
            release()

    async def _lookup(self: _Run, spec: ToolSpec, inv: ToolInvocation) -> Any:
        cache = _result_cache(self, spec)
        if cache is None:
            return NOT_STORED
        envelope = await cache.get(
            _cache_name(spec), inv.call.arguments, scope=spec.cache_scope, ctx=self.ctx
        )
        if envelope is None:
            return NOT_STORED
        self.emit(EventType.TOOL_CACHED, tool=spec.name)
        return envelope.payload

    async def _store(self: _Run, spec: ToolSpec, stored: tuple[dict[str, Any], Any]) -> None:
        cache = _result_cache(self, spec)
        if cache is None:
            return
        arguments, value = stored
        await cache.set(
            _cache_name(spec),
            arguments,
            value,
            ttl=spec.cache_ttl,
            scope=spec.cache_scope,
            ctx=self.ctx,
        )

    def _ok(self: _Run, value: Any, *, cached: bool = False) -> ToolResult:
        return ToolResult(
            call_id=self.call_id,
            name=self.name,
            status="ok",
            value=value,
            attempts=0 if cached else self.attempts,
            cached=cached,
            latency_ms=self.latency_ms(),
        )

    def _finish(self: _Run, result: ToolResult) -> None:
        if result.ok:
            self.emit_finished(
                result.value,
                tool=self.name,
                latency_ms=result.latency_ms,
                attempts=result.attempts,
                cached=result.cached,
            )
        elif result.status == "pending":
            approval_id = result.error.details.get("approval_id", "") if result.error else ""
            self.emit(EventType.TOOL_PENDING, tool=self.name, approval_id=approval_id)
        else:
            self.emit(
                EventType.TOOL_FAILED,
                tool=self.name,
                error_kind=result.error.kind if result.error is not None else "",
                attempts=result.attempts,
                latency_ms=result.latency_ms,
            )


def _result_cache(run: _Run, spec: ToolSpec) -> ToolResultCache | None:
    if spec.cache_ttl is None or run.ctx.cache is None:
        return None
    return ToolResultCache(run.ctx.cache, namespace=run.settings.namespace)


def _key(cache: ToolResultCache, spec: ToolSpec, inv: ToolInvocation, ctx: ToolContext) -> str:
    return cache.key(_cache_name(spec), inv.call.arguments, scope=spec.cache_scope, ctx=ctx)


def _handout(ran: tuple[ToolResult, Any]) -> Any:
    """What the waiters get: the stored value, encoded once; nothing if it is not stored."""
    stored = ran[1]
    if stored is NOT_STORED:
        return NOTHING
    try:
        return encode("tool", stored[1])
    except TypeError:
        return NOTHING


def _cache_name(spec: ToolSpec) -> str:
    """Tool part of the cache key; global keys add a schema digest so a changed tool never hits."""
    if spec.cache_scope == "global":
        return f"{spec.name}@{digest(spec.json_schema())[:8]}"
    return spec.name
