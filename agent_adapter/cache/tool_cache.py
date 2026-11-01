"""Tool-result cache over any CachePort; backend failures become misses."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from agent_adapter.cache.codec import Envelope, decode, encode
from agent_adapter.cache.keys import tool_key
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import log_failure
from agent_adapter.core.ports import CachePort
from agent_adapter.core.types import CacheScope

logger = logging.getLogger(__name__)


class ToolResultCache:
    """Stores tool results under `tool_key`, scoped to a run, a thread or globally."""

    def __init__(self, port: CachePort, *, namespace: str = "aa") -> None:
        self.port = port
        self.namespace = namespace

    async def get(
        self, tool: str, args: Mapping[str, Any], *, scope: CacheScope, ctx: ToolContext
    ) -> Envelope | None:
        """The cached envelope, or None on a miss, an uncacheable call or a backend error."""
        key = self._key(tool, args, scope, ctx)
        if not key:
            return None
        try:
            data = await self.port.get(key)
        except Exception as exc:
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "cache get failed for tool %s (run %s)",
                tool,
                ctx.run_id,
            )
            return None
        if data is None:
            logger.debug("cache miss for tool %s", tool)
            return None
        return decode(data, "tool")

    async def set(
        self,
        tool: str,
        args: Mapping[str, Any],
        value: Any,
        *,
        ttl: float | None,
        scope: CacheScope,
        ctx: ToolContext,
    ) -> bool:
        """Store `value`; False when it cannot be cached or the backend fails."""
        key = self._key(tool, args, scope, ctx)
        if not key:
            return False
        try:
            data = encode("tool", value)
        except TypeError as exc:
            logger.debug("tool %s: result not cached (%s)", tool, exc)
            return False
        try:
            await self.port.set(key, data, ttl)
        except Exception as exc:
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "cache set failed for tool %s (run %s)",
                tool,
                ctx.run_id,
            )
            return False
        return True

    def _key(self, tool: str, args: Mapping[str, Any], scope: CacheScope, ctx: ToolContext) -> str:
        try:
            return tool_key(
                self.namespace,
                tool,
                args,
                scope=scope,
                run_id=ctx.run_id,
                thread_id=ctx.thread_id,
            )
        except TypeError as exc:
            logger.debug("tool %s: arguments are not cacheable (%s)", tool, exc)
            return ""
