"""`memory_tools`: `remember`, `recall` and `forget` tools whose scope comes from the context."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ToolError
from agent_adapter.core.tool import ConfirmRule, ToolSpec, build_spec
from agent_adapter.memory._base import GuardedMemory
from agent_adapter.memory.records import MAX_K, MemoryRecord, MemoryScope

logger = logging.getLogger(__name__)

ScopeFrom = Callable[[ToolContext], MemoryScope | None]
REMEMBER, RECALL, FORGET = "remember", "recall", "forget"


def scope_from_context(ctx: ToolContext) -> MemoryScope | None:
    """The signed-in caller's scope (`ctx.user_id`, else the host's principal), or None."""
    user = ctx.user_id or _principal(ctx.extras.get("principal"))
    if not user:
        return None
    return MemoryScope(user_id=user, session_id=ctx.thread_id)


def thread_scope(ctx: ToolContext) -> MemoryScope | None:
    """Opt-in memory for anonymous chats: the thread only; safe only with unguessable ids."""
    if ctx.user_id or _principal(ctx.extras.get("principal")):
        return scope_from_context(ctx)
    return MemoryScope(session_id=ctx.thread_id) if ctx.thread_id else None


def user_scope(ctx: ToolContext) -> MemoryScope | None:
    """The caller's user-only scope (for backends that partition by user), or None."""
    scope = scope_from_context(ctx)
    if scope is None or not scope.user_id:
        return None
    return MemoryScope(user_id=scope.user_id)


def memory_tools(
    memory: GuardedMemory,
    *,
    scope_from: ScopeFrom | None = None,
    confirm_writes: ConfirmRule = False,
    forget: bool = True,
    across_sessions: bool = True,
    k: int = 5,
) -> list[ToolSpec]:
    """`remember`, `recall` and `forget` tools whose scope is `scope_from(ctx)`, never an argument.

    Raises:
        ConfigError: `memory` is not a memory backend, `scope_from` is not callable, or a
            bad `k`.
    """
    if not isinstance(memory, GuardedMemory):
        raise ConfigError(
            f"memory must be one of agent_adapter.memory's backends, got "
            f"{type(memory).__name__}: wrap your own store in CallableMemory"
        )
    if scope_from is None:
        scope_from = user_scope if memory.user_only else scope_from_context
    if not callable(scope_from):
        raise ConfigError("scope_from must be callable: (ToolContext) -> MemoryScope | None")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise ConfigError(f"k must be an int from 1 to {MAX_K}, got {k!r}")
    scopes = _Scopes(scope_from, across_sessions)

    async def remember(text: str, ctx: ToolContext) -> str:
        """Save one short fact about the user or the task, to recall in later turns or chats.

        Args:
            text: The fact in one or two plain sentences.
        """
        record = _record(scopes.write(ctx, REMEMBER), text, ctx.run_id)
        (stored,) = await memory.add([record], ctx=ctx)
        return f"Saved as {stored.id}."

    async def recall(query: str, ctx: ToolContext) -> list[dict[str, Any]]:
        """Search saved facts about the user; an empty query lists the newest.

        Args:
            query: Words to look for.
        """
        hits = await memory.search(query, scope=scopes.read(ctx, RECALL), k=k, ctx=ctx)
        return [
            {"id": hit.record.id, "text": hit.record.text, "saved": _day(hit.record)}
            for hit in hits
        ]

    async def forget_memory(id: str, ctx: ToolContext) -> str:
        """Delete one saved fact by its id (from `recall`).

        Args:
            id: The id of the fact, such as "mem_0123abcd...".
        """
        if not await memory.delete([id], scope=scopes.read(ctx, FORGET), ctx=ctx):
            raise ToolError(f"no saved fact with id {id!r}", tool=FORGET)
        return f"Deleted {id}."

    specs = [
        build_spec(remember, name=REMEMBER, effects="additive", confirm=confirm_writes),
        build_spec(recall, name=RECALL, effects="read_only", parallel_safe=True),
    ]
    if forget:
        specs.append(
            build_spec(forget_memory, name=FORGET, effects="destructive", confirm=confirm_writes)
        )
    return specs


class _Scopes:
    """Resolves the write and read scopes of one call, failing closed."""

    def __init__(self, scope_from: ScopeFrom, across_sessions: bool) -> None:
        self.scope_from = scope_from
        self.across_sessions = across_sessions

    def write(self, ctx: ToolContext, tool: str) -> MemoryScope:
        """The scope new records go to.

        Raises:
            ToolError: the context names no user or session, or `scope_from` failed.
        """
        try:
            scope = self.scope_from(ctx)
        except Exception as exc:
            logger.warning("memory scope_from failed (%s)", type(exc).__name__)
            raise ToolError("memory is not available for this caller", tool=tool) from exc
        if not isinstance(scope, MemoryScope):
            raise ToolError("memory needs a user or session id in the context", tool=tool)
        return scope

    def read(self, ctx: ToolContext, tool: str) -> MemoryScope:
        """The scope searched: the user's, across sessions, when allowed."""
        scope = self.write(ctx, tool)
        if self.across_sessions and scope.user_id and scope.session_id:
            return scope.model_copy(update={"session_id": ""})
        return scope


def _record(scope: MemoryScope, text: str, source: str) -> MemoryRecord:
    try:
        return MemoryRecord(scope=scope, text=text.strip(), source=source[:256])
    except ValidationError as exc:
        raise ToolError(
            "the fact must be non-empty and under 20,000 characters", tool=REMEMBER
        ) from exc


def _day(record: MemoryRecord) -> str:
    return record.created_at.date().isoformat() if record.created_at else ""


def _principal(value: object) -> str:
    """A host principal's id: a plain str, or an object with a str `id`."""
    if isinstance(value, str):
        return value
    found = getattr(value, "id", "")
    return found if isinstance(found, str) else ""
