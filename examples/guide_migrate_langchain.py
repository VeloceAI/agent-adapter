"""Migration: from LangChain `@tool` functions to a registry, keeping LangChain as a caller."""

import asyncio
import importlib.util
from typing import Any

from agent_adapter import Executor, Registry, ToolCall

registry = Registry()


# --8<-- [start:native]
@registry.tool(kind="search", parallel_safe=True, timeout=10)  # was: @tool from langchain_core
def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())


executor = Executor(registry)  # approval, budgets, cache and ledger live here now
# --8<-- [end:native]


def import_existing() -> Any:
    """Step 1, no rewrite yet: bring LangChain tools in as they are (needs langchain-core)."""
    # --8<-- [start:import]
    from langchain_core.tools import tool

    from agent_adapter.integrations.langchain import import_langchain

    def shout(text: str) -> str:
        """Upper-case a text."""
        return text.upper()

    legacy_tool = tool(shout)  # what `@tool` did in the old module
    legacy = Registry()
    # policy options (timeout, retries, confirm, ...) apply to imported tools too
    import_langchain(legacy, [legacy_tool], prefix="lc_", timeout=5)
    return Executor(legacy).run("lc_shout", {"text": "faq"})
    # --8<-- [end:import]


def export_back() -> Any:
    """Step 3: LangChain agents keep working; every call now runs through the executor."""
    # --8<-- [start:export]
    from agent_adapter.integrations.langchain import structured_tools

    tools = structured_tools(registry, executor=executor)  # pass to bind_tools / an agent
    call = {"name": "word_count", "args": {"text": "one two"}, "id": "call_1", "type": "tool_call"}
    return tools[0].ainvoke(call)  # a ToolMessage
    # --8<-- [end:export]


async def main() -> None:
    # --8<-- [start:run]
    result = await executor.run_call(ToolCall(name="word_count", arguments={"text": "a b c"}))
    print(result.value, result.status)
    # --8<-- [end:run]
    if importlib.util.find_spec("langchain_core") is None:
        print("langchain-core not installed: import and export steps skipped")
        return
    print(await import_existing())
    print((await export_back()).content)


if __name__ == "__main__":
    asyncio.run(main())
