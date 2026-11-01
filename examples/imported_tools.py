"""Tools from a raw JSON schema, a hand-written argument model, and per-target export."""

import asyncio
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_adapter import Executor, Registry, spec_from_schema
from agent_adapter.integrations.anthropic import anthropic_tools
from agent_adapter.integrations.openai import chat_tools

registry = Registry()

# --8<-- [start:schema]
SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search text."},
        "max-results": {"type": "integer", "default": 5},
    },
    "required": ["query"],
}


async def remote_search(**arguments: Any) -> dict[str, Any]:
    """Stands in for an MCP or HTTP call; gets the schema's own property names."""
    return arguments


search = spec_from_schema(
    "remote_search", "Search the remote index.", SCHEMA, remote_search, effects="read_only"
)
registry.add(search)
# --8<-- [end:schema]


# --8<-- [start:args_model]
class LookupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    article: str = Field(description="Article topic, e.g. returns.")


def legacy_lookup(article: str, verbose: bool = False) -> str:
    """Look up an article in the legacy system."""
    return f"Article {article}"


registry.wrap(legacy_lookup, args_model=LookupArgs, hide="verbose", targets=["anthropic"])
# --8<-- [end:args_model]


async def main() -> None:
    # --8<-- [start:call]
    executor = Executor(registry)
    print(await executor.run("remote_search", {"query": "returns"}))  # unset optionals are not sent
    print(await executor.run("remote_search", {"query": "returns", "max-results": 2}))
    # --8<-- [end:call]
    # --8<-- [start:targets]
    print([t["function"]["name"] for t in chat_tools(registry)])
    print([t["name"] for t in anthropic_tools(registry)])
    print(registry.for_target("mcp").names())
    # --8<-- [end:targets]
    print(json.dumps(search.tool_def().model_dump()["parameters"]["required"]))


if __name__ == "__main__":
    asyncio.run(main())
