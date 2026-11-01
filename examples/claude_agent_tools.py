"""Claude Agent SDK: registry tools on an in-process MCP server, gated by the same bridge."""

import asyncio
from typing import cast

import mcp
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    McpSdkServerConfig,
    ToolPermissionContext,
)
from claude_agent_sdk.types import HookEvent
from mcp.types import CallToolResult, RequestParamsMeta, TextContent

from agent_adapter import ApprovalDecision, ApprovalRequest, CallbackApproval, Executor, Registry
from agent_adapter.integrations.claude_agent import claude_sdk_bridge

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@registry.tool(confirm=True)
def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


def reviewer(request: ApprovalRequest) -> ApprovalDecision:
    """A stand-in for a human: invoices customer cus_1024, whatever the model asked."""
    return ApprovalDecision(approved=True, edited_args={"customer_id": "cus_1024", "amount": 49.5})


# --8<-- [start:wire]
executor = Executor(registry)
port = CallbackApproval(reviewer)


async def chat(prompt: str) -> None:
    """One Claude Code session: a fresh bridge, so its run and tokens are this session's."""
    bridge = claude_sdk_bridge(registry, port, executor=executor)
    options = ClaudeAgentOptions(
        # The bridge types these as plain dicts, so the adapter imports without the SDK.
        mcp_servers={bridge.name: cast(McpSdkServerConfig, bridge.server)},  # mcp__<name>__<tool>
        can_use_tool=bridge.can_use_tool,  # its approvals are what the server's tools accept
        hooks=cast(dict[HookEvent, list[HookMatcher]], bridge.hooks),  # tokens end with the turn
    )
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                print(message)
    finally:
        executor.end_run(bridge.run_id)  # the bridge's run (and its guards) ends here


# --8<-- [end:wire]


def first_text(result: CallToolResult) -> str:
    """The text of a tool result's first content block."""
    block = result.content[0]
    if not isinstance(block, TextContent):
        raise TypeError(f"expected a text block, got {block.type}")
    return block.text


async def main() -> None:
    """Without the Claude Code CLI: ask permission as the CLI would, then call the server."""
    bridge = claude_sdk_bridge(registry, port, executor=executor)
    tool_input = {"customer_id": "cus_2048", "amount": 49.5}
    permission = await bridge.can_use_tool(
        "mcp__agent_adapter__send_invoice", tool_input, ToolPermissionContext(tool_use_id="tu_1")
    )
    print(type(permission).__name__, permission.updated_input)
    async with mcp.Client(bridge.server["instance"]) as client:
        rate = await client.call_tool("shipping_rate", {"region": "us"})
        sent = await client.call_tool(  # Claude Code sends the tool-use id with the call
            "send_invoice",
            permission.updated_input or tool_input,
            # An open TypedDict (PEP 728 extra_items), which mypy does not model yet.
            meta=cast(RequestParamsMeta, {"claudecode/toolUseId": "tu_1"}),
        )
    print(first_text(rate), first_text(sent))
    executor.end_run(bridge.run_id)


if __name__ == "__main__":
    asyncio.run(main())
