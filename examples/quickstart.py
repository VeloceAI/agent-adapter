"""Quickstart: define tools once, render them for two providers, run them with the executor."""

import asyncio
import json

from agent_adapter import AutoDeny, Executor, Registry, ToolCall, ToolContext
from agent_adapter.integrations.gemini import function_declarations
from agent_adapter.integrations.openai import chat_tools

registry = Registry()


@registry.tool(kind="retrieval", parallel_safe=True)
async def lookup_article(article: str, lang: str = "en", ctx: ToolContext | None = None) -> str:
    """Return the text of a help-center article.

    Args:
        article: Article topic, e.g. returns.
        lang: Language code.
    """
    return f"Article {article} ({lang})"


def legacy_rate(item: str, region: str) -> float:
    """Return the shipping rate for an item, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


registry.wrap(legacy_rate, name="shipping_rate", fixed={"region": "us"})

# One executor per process: approval, budgets, guards, cache and ledger live here.
executor = Executor(registry, approval=AutoDeny())


async def main() -> None:
    print(json.dumps(chat_tools(registry), indent=2))
    print(json.dumps(function_declarations(registry), indent=2))
    print(await executor.run("lookup_article", {"article": "returns"}))
    # --8<-- [start:model-call]
    # A tool call from a model: name, parsed arguments and the provider's call id.
    call = ToolCall(name="shipping_rate", arguments={"item": "laptop"}, id="call_1")
    result = await executor.run_call(call)  # a ToolResult; failures are in result.error
    print(result.value if result.ok else result.error)
    # --8<-- [end:model-call]


if __name__ == "__main__":
    asyncio.run(main())
