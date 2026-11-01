"""OpenAI strict mode: every field required, and null means "use the default"."""

import asyncio
import json
from typing import Any

from agent_adapter import Executor, Registry
from agent_adapter.integrations.openai import chat_tools

registry = Registry()


# --8<-- [start:tool]
@registry.tool
def search(query: str, top_k: int = 8, region: str | None = None) -> list[str]:
    """Search articles.

    Args:
        query: Search text.
        top_k: How many hits.
        region: Optional region filter.
    """
    return [f"{query} #{i} ({region})" for i in range(top_k)]


# --8<-- [end:tool]


async def main() -> None:
    # --8<-- [start:render]
    parameters = chat_tools(registry)[0]["function"]["parameters"]
    print(json.dumps(parameters, indent=2))
    # --8<-- [end:render]

    # --8<-- [start:call]
    executor = Executor(registry)  # model tool calls always go through the executor
    # Strict mode sends every field; null stands for "not given".
    hits = await executor.run("search", {"query": "returns", "top_k": None, "region": None})
    print(len(hits))  # 8: top_k's default applied
    print(hits[0])  # returns #0 (None): region's default is None, so null is kept
    # --8<-- [end:call]


def loose_tools() -> list[dict[str, Any]]:
    """The same tools rendered without strict mode."""
    # --8<-- [start:loose]
    tools = chat_tools(registry, strict=False)
    # --8<-- [end:loose]
    return tools


if __name__ == "__main__":
    asyncio.run(main())
