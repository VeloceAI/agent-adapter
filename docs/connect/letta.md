---
title: "Letta: register Python tools over MCP"
description: "Register your MCP URL with Letta once; its server then forwards each tool call to your Python tools, with approvals and cost tracking."
---

# Connect: Letta

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-10-30

Letta is a server platform. You register our MCP URL once; Letta's server then forwards
each tool call to us. Letta's own sandbox cannot run our executor, so do not paste our
tools in as Letta "server tools".

## Our side

Letta Cloud, or a Letta server on another host, reaches us over the network, so serve the
public app with the `hosted` registry:

```python
--8<-- "examples/connect_server.py:public"
--8<-- "examples/connect_server.py:hosted"
```

Serve `public_app(hosted, host, agent_token, reviewer_token)`.

## Their side

Python SDK (`letta-client`):

```python
server = client.mcp_servers.create(
    server_name="shop_tools",
    config={
        "mcp_server_type": "streamable_http",
        "server_url": "https://tools.example.com/mcp",
        "auth_header": "Authorization",
        "auth_token": "Bearer <AA_AGENT_TOKEN>",
    },
)
tools = client.mcp_servers.list_tools(server.id)
agent = client.agents.create(tool_ids=[tool.id for tool in tools], ...)
```

TypeScript SDK: `client.mcpServers.create({ server_name, config: { … } })` with the same
config keys, then `client.mcpServers.listTools(server.id)`.

| Key | Meaning |
|---|---|
| `mcp_server_type: "streamable_http"` | our transport |
| `server_url` | our `/mcp` endpoint |
| `auth_header`, `auth_token` | the header name and its full value (`Bearer …`) |
| `tool_ids` on `agents.create` | which of the listed tools the agent may call |

Letta adds an `x-agent-id` header to each call. It is not auth: our principal comes from
the bearer token only.

## Checklist

| Check | Why |
|---|---|
| serve `hosted`, or leave confirm tools out of `tool_ids` | Letta does not send our resume keys back |
| one token per Letta deployment | one principal per tenant |
| a self-hosted Letta in Docker still uses `public_app` | a container is not loopback: `local_app` refuses it |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- Letta, connecting remote MCP servers: <https://docs.letta.com/guides/mcp/sse>
- Letta, MCP setup: <https://docs.letta.com/guides/mcp/setup>
- Letta, tool execution: <https://docs.letta.com/guides/agents/tool-execution-overview>
