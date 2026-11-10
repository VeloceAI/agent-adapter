---
title: "Genkit: call Python tools over MCP"
description: "Use the @genkit-ai/mcp plugin to give Genkit's ai.generate your Python tools, with approvals and cost tracking on every call."
---

# Connect: Genkit

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-10-30

Genkit's `@genkit-ai/mcp` plugin runs an MCP host inside your Genkit app. It connects to
our server and offers our tools to `ai.generate`.

## Our side

```python
--8<-- "examples/connect_server.py:public"
--8<-- "examples/connect_server.py:hosted"
```

Serve `public_app(hosted, …)`: Genkit has no approval step to resume ours. In development
on one machine, `local_app()` is enough.

## Their side

`npm i genkit @genkit-ai/mcp`:

```ts
import { genkit } from "genkit";
import { createMcpHost } from "@genkit-ai/mcp";

const ai = genkit({ plugins: [/* your model plugin */] });

const mcpHost = createMcpHost({
  name: "agentAdapter",
  mcpServers: {
    shopTools: {
      url: "https://tools.example.com/mcp",
      requestInit: { headers: { Authorization: `Bearer ${process.env.AA_AGENT_TOKEN}` } },
    },
  },
});

const { text } = await ai.generate({
  prompt: "What is the shipping rate to the US?",
  tools: await mcpHost.getActiveTools(ai),
});
```

| Key | Meaning |
|---|---|
| `createMcpHost({ name, mcpServers })` | one host for many servers |
| `mcpServers.<name>.url` | our `/mcp` endpoint (Streamable HTTP config) |
| `requestInit` | the transport's request options: carries the bearer header |
| `disabled` | stops a server without removing it |
| `getActiveTools(ai)` | the tools of every connected server |

This page covers Genkit JS. Genkit Go has its own MCP plugin, and a Python MCP client is
landing (PR 6365); point either at the same URL with the same header.

## Checklist

| Check | Why |
|---|---|
| serve `hosted` | Genkit does not send our resume keys back; a confirm tool would never run |
| the token stays on the server | `process.env`, never in a browser bundle |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- Genkit MCP plugin: <https://genkit.dev/docs/model-context-protocol/>
- Genkit Python MCP client: <https://github.com/genkit-ai/genkit/pull/6365>
