---
title: "LangGraph.js: use Python tools over MCP"
description: "Turn your Python tools into LangChain.js tools with @langchain/mcp-adapters and use them in createAgent or any LangGraph.js graph."
---

# Connect: LangGraph.js

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-10-30

`@langchain/mcp-adapters` (2.0 or later) turns our MCP tools into LangChain.js tools. Use
them with `createAgent`, or in a LangGraph.js graph like any other LangChain tool.

## Our side

```python
--8<-- "examples/connect_server.py:public"
```

In development on one machine, `local_app()` and `http://127.0.0.1:8000/mcp` are enough.

## Their side

```ts
import { MCPAdapter } from "@langchain/mcp-adapters";
import { createAgent } from "langchain";

const adapter = new MCPAdapter({
  servers: {
    shopTools: {
      url: "https://tools.example.com/mcp", // HTTP is the default for a URL
      headers: { Authorization: `Bearer ${process.env.AA_AGENT_TOKEN}` },
    },
  },
});
try {
  const tools = (await adapter.listTools()).filter((t) => t.name !== "send_invoice");
  const agent = createAgent({ model, tools });
  await agent.invoke({ messages: [{ role: "user", content: "Shipping rate to the US?" }] });
} finally {
  await adapter.close();
}
```

| Key | Meaning |
|---|---|
| `servers.<name>.url` | our `/mcp` endpoint; leave `transport` and `mode` out for a new remote server |
| `servers.<name>.headers` | sent on every request: the bearer token |
| `listTools()` | LangChain.js tools, one per MCP tool |
| `close()` | ends the connections |

Older code uses `MultiServerMCPClient({ mcpServers: { … } })` with `getTools()`; 2.0
documents `MCPAdapter`.

## Approval

A LangGraph.js `interrupt()` before a tool node is a UI gate: it adds a step, it does not
approve our call. Confirm tools reached over MCP park as `approval required: <id>` and the
adapter does not send our resume keys back. Keep them out of `tools` (as above), or call
them through the [TS client](../cross-language/typescript-client.md), where `decide` runs
the call.

## Checklist

| Check | Why |
|---|---|
| confirm tools filtered out, or `hosted` served | the adapter cannot resume a parked call |
| one adapter per principal | the header is the identity |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- LangChain.js MCP: <https://docs.langchain.com/oss/javascript/langchain/mcp>
- MCP auth (headers): <https://docs.langchain.com/oss/javascript/langchain/mcp/auth.md>
- MCP connections: <https://docs.langchain.com/oss/javascript/langchain/mcp/connections.md>
