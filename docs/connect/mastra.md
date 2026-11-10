---
title: "Mastra: use Python tools over MCP"
description: "Load your Python tools into Mastra in one MCP call, or use the TypeScript client to keep human-in-the-loop approvals working end to end."
---

# Connect: Mastra

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) or the [TypeScript client](../cross-language/typescript-client.md) · their docs checked 2026-10-30

Two ways in. MCP loads every tool in one call. The TS client keeps approvals in Python:
`decide` runs the call, so confirm tools work end to end.

## Our side

```python
--8<-- "examples/connect_server.py:public"
```

In development on one machine, `local_app()` (MCP only) is enough for the MCP route.

## Their side: MCP

```ts
import { MCPClient } from "@mastra/mcp";

const mcp = new MCPClient({
  servers: {
    shopTools: {
      url: new URL("https://tools.example.com/mcp"),
      requestInit: { headers: { Authorization: `Bearer ${process.env.AA_AGENT_TOKEN}` } },
      requireToolApproval: ({ toolName }) => toolName === "send_invoice", // UI gate only
    },
  },
  timeout: 60_000,
});
const tools = await mcp.listTools(); // names are namespaced by server
```

| Key | Meaning |
|---|---|
| `servers.<name>.url` | our `/mcp` endpoint, as a `URL` |
| `servers.<name>.requestInit.headers` | carries the bearer token |
| `servers.<name>.requireToolApproval` | `boolean` or a function: Mastra asks a person first |
| `timeout` | per request, in milliseconds (default 60 000) |
| `listTools()` / `listToolsets()` | tools for an agent definition / per call |

## Their side: TS client

```ts
import { ToolClient } from "@agent-adapter/client";
import { createTool } from "@mastra/core/tools";

const client = new ToolClient({ baseUrl: "https://tools.example.com", token: process.env.AA_AGENT_TOKEN });
const defs = await client.toolDefinitions({ runId: "run_42" });
// each def is { name, description, inputSchema, execute }: createTool({ id: def.name, ... })
```

Mastra's `createTool` takes `id`, `description`, `inputSchema` and `execute`. Its
`inputSchema` is a Standard JSON Schema (Zod, Valibot, ArkType), not a plain JSON Schema
object: convert `def.inputSchema` with a library that supports that standard, or use the
MCP route. `execute` returns the result, or `{ pending_approval: approval_id }` while a reviewer
decides with `client.decide(...)`. The package is private for now: use it from this
repository.

## Checklist

| Check | Why |
|---|---|
| `requireToolApproval` is extra | it only adds a gate in Mastra's UI; our executor still decides `confirm` tools |
| over MCP, serve `hosted` or keep confirm tools for the TS client route | Mastra's `MCPClient` does not send our resume keys back |
| a browser app calls the TS client through your backend | the server sends no CORS headers, and the token must stay server-side |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- Mastra `MCPClient` reference: <https://mastra.ai/reference/tools/mcp-client>
- Mastra `createTool` reference: <https://mastra.ai/reference/tools/create-tool>
- Our TS client: `clients/typescript/README.md`
