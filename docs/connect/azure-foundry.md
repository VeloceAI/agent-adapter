---
title: "Azure AI Foundry agents: Python tools over MCP"
description: "Let Azure AI Foundry Agent Service call your Python tools through its mcp tool, with auth, approvals and cost tracking on every call."
---

# Connect: Azure AI Foundry Agent Service

:material-check-circle:{ .status-ok } Recipe · through our public [MCP server](../cross-language/mcp.md) · their docs checked 2026-10-30

A Foundry agent's `mcp` tool is run by the Foundry service: Azure calls our server, so it
must be reachable from the internet, with auth. The token lives in a Foundry project
connection, not in your code.

## Our side

```python
--8<-- "examples/connect_server.py:public"
--8<-- "examples/connect_server.py:hosted"
```

Serve `public_app(hosted, host, agent_token, reviewer_token)`.

## Their side

1. In the Foundry project, create a **project connection** with key-based auth: credential
   name `Authorization`, credential value `Bearer <agent token>`. If its endpoint differs
   from `server_url`, the connection's endpoint wins.
2. Add the tool to the agent (`azure-ai-projects`):

```python
tool = MCPTool(
    server_label="shop_tools",
    server_url="https://tools.example.com/mcp",
    allowed_tools=["shipping_rate"],
    require_approval="always",
    project_connection_id="shop-tools-connection",
)
agent = project.agents.create_version(
    agent_name="ShopAgent",
    definition=PromptAgentDefinition(model=model, instructions="Answer customer-support questions.", tools=[tool]),
)
```

| Key | Meaning |
|---|---|
| `server_label`, `server_url` | a unique name per agent, and our `/mcp` URL |
| `allowed_tools` | a non-empty list limits the tools; empty or absent means **all** tools |
| `require_approval` | `always` (default), `never`, `{"never": [...]}`, `{"always": [...]}` |
| `project_connection_id` | the connection that stores the token |
| `mcp_approval_request` → `mcp_approval_response` | Foundry's approval round trip, in the same conversation |

## Checklist

| Check | Why |
|---|---|
| serve `hosted`, and set a non-empty `allowed_tools` | `[]` means every tool; Foundry does not send our resume keys back |
| Foundry's approval is a gate only | `approve: true` there never approves our `confirm` tools ([framework approvals](../guides/security.md#framework-approvals)) |
| the token is in the project connection | not in code; anyone with project access can read it, so give it no `APPROVER_SCOPE` and rotate it on its own |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

For an agent that runs in your own process (MAF .NET with a local MCP client), see
[MAF .NET](maf-dotnet.md).

## Sources

- Foundry, connect to MCP servers: <https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/model-context-protocol>
- Foundry, MCP server authentication: <https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/mcp-authentication>
- MAF, hosted MCP tools: <https://learn.microsoft.com/en-us/agent-framework/agents/tools/hosted-mcp-tools>
