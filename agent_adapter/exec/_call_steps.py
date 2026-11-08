"""One call's cache step (lookup, store) and its success result and closing event."""

from __future__ import annotations

from typing import Any, Protocol

from agent_adapter.cache.keys import digest
from agent_adapter.cache.tool_cache import ToolResultCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolResult
from agent_adapter.exec._settings import _Settings
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


class CallSteps:
    """Cache lookup and store, the ok result and the finishing event of one call."""

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


def _cache_name(spec: ToolSpec) -> str:
    """Tool part of the cache key; global keys add a schema digest so a changed tool never hits."""
    if spec.cache_scope == "global":
        return f"{spec.name}@{digest(spec.json_schema())[:8]}"
    return spec.name
