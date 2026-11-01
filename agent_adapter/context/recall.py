"""Saved tool results: their ids and cache keys, and the `recall_context` tool that reads them.

Each saved result lives under its owner's scope (the thread, else the run), so a
`recall_context` call in one conversation cannot read another conversation's results.
"""

from __future__ import annotations

import logging
import re

from agent_adapter.cache.codec import decode, encode
from agent_adapter.cache.keys import KEY_VERSION, digest
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolError, log_failure
from agent_adapter.core.messages import Message
from agent_adapter.core.ports import CachePort
from agent_adapter.core.tool import ToolSpec, build_spec

logger = logging.getLogger(__name__)

RECALL_TOOL_NAME = "recall_context"
_ID = re.compile(r"ctx_[0-9a-f]{16}")


def saved_result_id(message: Message) -> str:
    """The stable id of one tool result: `ctx_` and 16 hex digits of its digest."""
    return "ctx_" + digest([message.tool_call_id, message.name, message.content])[:16]


def owner_scope(ctx: ToolContext | None) -> str:
    """Who a saved result belongs to: `thread=<id>`, else `run=<id>`, else `global`."""
    if ctx is not None and ctx.thread_id:
        return f"thread={ctx.thread_id}"
    if ctx is not None and ctx.run_id:
        return f"run={ctx.run_id}"
    return "global"


def saved_result_key(result_id: str, scope: str = "global") -> str:
    """The cache key a saved tool result lives under, inside its owner's scope."""
    return f"aa:ctx:v{KEY_VERSION}:tool_result:{scope}:{result_id}"


async def save_result(
    store: CachePort, message: Message, *, scope: str = "global", ttl: float | None = None
) -> str:
    """Save a tool result's full text for `ttl` seconds; its id, or "" if the store failed."""
    result_id = saved_result_id(message)
    payload = {
        "id": result_id,
        "tool": message.name,
        "tool_call_id": message.tool_call_id,
        "content": message.content,
    }
    try:
        await store.set(saved_result_key(result_id, scope), encode("tool", payload), ttl=ttl)
    except Exception as exc:
        log_failure(logger, logging.WARNING, exc, "could not save tool result %s", result_id)
        return ""
    return result_id


def recall_tool(store: CachePort) -> ToolSpec:
    """The `recall_context(id)` tool: the full text of a result saved in `store` by this run.

    It reads only results saved under the calling context's owner scope (see `owner_scope`).
    """

    async def recall_context(id: str, ctx: ToolContext) -> str:
        """Read the full text of a tool result that was shortened to save context.

        Args:
            id: The id shown in the shortened result, such as "ctx_0123456789abcdef".

        Raises:
            ToolError: the id is unknown or the saved result cannot be read.
        """
        if not _ID.fullmatch(id):
            raise ToolError(f"{id!r} is not a saved result id", tool=RECALL_TOOL_NAME)
        try:
            data = await store.get(saved_result_key(id, owner_scope(ctx)))
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "could not read saved tool result %s", id)
            raise ToolError("the saved result cannot be read now", tool=RECALL_TOOL_NAME) from exc
        envelope = decode(data, "tool") if data is not None else None
        payload = envelope.payload if envelope is not None else None
        if not isinstance(payload, dict) or not isinstance(payload.get("content"), str):
            raise ToolError(f"no saved result with id {id}", tool=RECALL_TOOL_NAME)
        content: str = payload["content"]
        return content

    return build_spec(
        recall_context,
        name=RECALL_TOOL_NAME,
        effects="read_only",
        idempotent=True,
        parallel_safe=True,
    )
