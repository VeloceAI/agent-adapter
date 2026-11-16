---
title: "OpenAI Agents SDK (JS): use Python tools from TypeScript over MCP"
description: "Give an OpenAI Agents SDK for TypeScript agent your Python tools through our MCP server, with approvals, budgets and traces on every call."
---

# Connect: OpenAI Agents SDK for TypeScript

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-11-16 · tested live

`@openai/agents` runs the agent loop in your Node process. `MCPServerStreamableHttp`
connects it to our `/mcp` endpoint; the agent calls our tools like its own. For the Python
Agents SDK, use the in-process [integration](../integrations/openai-agents.md) instead.

## Our side

```python
--8<-- "examples/connect_server.py:public"
--8<-- "examples/connect_server.py:hosted"
```

Serve `public_app(hosted, …)`, or filter confirm tools out on their side: the SDK does not
send our resume keys back. In development on one machine, `local_app()` is enough.

## Their side

`npm i @openai/agents`:

```ts
import { Agent, MCPServerStreamableHttp, run } from "@openai/agents";

const shopTools = new MCPServerStreamableHttp({
  url: "https://tools.example.com/mcp",
  name: "shop_tools",
  requestInit: { headers: { Authorization: `Bearer ${process.env.AA_AGENT_TOKEN}` } },
  cacheToolsList: true,
});

await shopTools.connect();
try {
  const agent = new Agent({
    name: "Shop assistant",
    instructions: "Answer shipping questions with the tools.",
    mcpServers: [shopTools],
  });
  const result = await run(agent, "What is the shipping rate to the US?");
  console.log(result.finalOutput);
} finally {
  await shopTools.close();
}
```

| Key | Meaning |
|---|---|
| `url` | our `/mcp` endpoint (Streamable HTTP) |
| `requestInit.headers` | carries the bearer token on every request |
| `cacheToolsList` | lists our tools once per connection; leave it off while you change the registry |
| `toolFilter` | `createMCPToolStaticFilter({ allowed: [...] })` keeps tools out of the model's view |
| `mcpServers` on `Agent` | the agent sees every tool the server lists |

`examples/connect/js/check.ts` runs this client, Vercel AI SDK, Mastra and Genkit against
our live server: each lists the tools, calls `shipping_rate`, and sees `send_invoice` park
with `approval required: apr_…` instead of running.

## Approvals

The SDK's human-in-the-loop (`needsApproval`, `result.interruptions`,
`result.state.approve(...)`) pauses its own run. It adds a gate; it never approves our
`confirm` tools. For a side effect, call our REST route from a function tool and let your
reviewer decide with `POST /approvals/{id}/decide` (the [TS client](../cross-language/typescript-client.md)
wraps both).

`hostedMcpTool({ serverLabel, serverUrl, requireApproval })` is the other way in: OpenAI's
servers call ours, as on the [OpenAI hosted MCP](openai-hosted-mcp.md) page.

## Checklist

| Check | Why |
|---|---|
| serve `hosted`, or `toolFilter` the confirm tools out | the SDK does not send our resume keys back; a confirm tool parks on every call |
| `close()` in `finally` | the client keeps a session open on our server |
| the token stays on the server | `process.env`, never in a browser bundle |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- Agents SDK MCP guide: <https://openai.github.io/openai-agents-js/guides/mcp/>
- Agents SDK human in the loop: <https://openai.github.io/openai-agents-js/guides/human-in-the-loop/>
