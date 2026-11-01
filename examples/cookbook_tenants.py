"""Cookbook: one registry, a tool subset per tenant, one shared executor."""

import asyncio

from agent_adapter import Executor, Registry, ToolCall, ToolContext
from agent_adapter.integrations.openai import chat_tools

catalogue = Registry()


# --8<-- [start:tools]
@catalogue.tool(tags={"basic", "pro"})
def search_docs(query: str, ctx: ToolContext) -> str:
    """Search the tenant's documents."""
    return f"{ctx.extras['tenant']}: results for {query}"


@catalogue.tool(tags={"pro"})
def export_report(month: str) -> str:
    """Export the monthly report."""
    return f"report {month}"


@catalogue.tool(tags={"pro"}, confirm=True)
def delete_workspace(name: str) -> str:
    """Delete a workspace."""
    return f"deleted {name}"


# --8<-- [end:tools]

# --8<-- [start:views]
PLANS = {"acme": "basic", "globex": "pro"}  # from your tenant database
views = {plan: catalogue.subset(tags={plan}) for plan in set(PLANS.values())}  # built once
executor = Executor(catalogue)  # one executor; no approval port: confirm tools are denied


def registry_for(tenant: str) -> Registry:
    return views[PLANS[tenant]]


# --8<-- [end:views]


async def main() -> None:
    # --8<-- [start:run]
    for tenant in ("acme", "globex"):
        tools = registry_for(tenant)
        print(tenant, [t["function"]["name"] for t in chat_tools(tools)])  # what the model sees
    ctx = executor.context(thread_id="acme:chat-1", extras={"tenant": "acme"})
    search = ToolCall(name="search_docs", arguments={"query": "refunds"})
    export = ToolCall(name="export_report", arguments={"month": "2026-09"})
    for call in (search, export):  # the model may still name a tool it was not shown
        result = await executor.run_call(call, ctx, registry=registry_for("acme"))
        print(result.status, result.value or (result.error.kind if result.error else ""))
    # --8<-- [end:run]


if __name__ == "__main__":
    asyncio.run(main())
