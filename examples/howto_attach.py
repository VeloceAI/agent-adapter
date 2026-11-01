"""Attach one tool to plain code, a sync caller and two model APIs, with the same safety."""

import asyncio
import json

from agent_adapter import (
    ApprovalDenied,
    AutoDeny,
    Executor,
    LoggingHook,
    Registry,
    RunLimits,
    ToolCall,
    run_sync,
)
from agent_adapter.integrations.anthropic import anthropic_tool
from agent_adapter.integrations.openai import chat_tool

# --8<-- [start:define]
registry = Registry()


@registry.tool(kind="search", parallel_safe=True, timeout=10)
def find_article(article: str) -> str:
    """Return the text of a help-center article.

    Args:
        article: Article id, e.g. HC-101.
    """
    return f"Article {article}: how to fix it"


def delete_notes(user_id: str) -> int:
    """Delete a user's saved notes; returns how many were deleted."""
    return 3


registry.wrap(delete_notes, confirm=True, effects="destructive")  # existing code, unchanged
# --8<-- [end:define]

# --8<-- [start:executor]
executor = Executor(
    registry,
    hooks=[LoggingHook()],  # one redacted log line per call, once logging is configured
    approval=AutoDeny(reason="no reviewer in this script"),  # confirm tools fail closed
    limits=RunLimits(max_calls=20),  # at most 20 calls per run
)
# --8<-- [end:executor]


async def main() -> None:
    # --8<-- [start:run]
    print(await executor.run("find_article", {"article": "HC-101"}))
    calls = [ToolCall(name="find_article", arguments={"article": a}) for a in ("HC-102", "HC-201")]
    print([r.value for r in await executor.run_many(calls)])
    try:
        await executor.run("delete_notes", {"user_id": "u-1"})
    except ApprovalDenied as denied:
        print("denied:", denied.reason)
    # --8<-- [end:run]

    # --8<-- [start:surfaces]
    spec = registry.get("find_article")
    print(json.dumps(chat_tool(spec)["function"]["parameters"]["required"]))
    print(json.dumps(sorted(anthropic_tool(spec))))
    # --8<-- [end:surfaces]


def sync_caller() -> str:
    # --8<-- [start:sync]
    # A Flask view, a Celery task or a script: no event loop of its own.
    return str(run_sync(executor.run("find_article", {"article": "HC-301"}), timeout=30))
    # --8<-- [end:sync]


if __name__ == "__main__":
    asyncio.run(main())
    print(sync_caller())
