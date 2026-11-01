# Connect: Vercel AI SDK

:material-check-circle:{ .status-ok } Recipe · through the [TypeScript client](../cross-language/typescript-client.md) or our [MCP server](../cross-language/mcp.md) · their docs checked 2026-10-30

AI SDK 7 (TypeScript). The TS client route keeps approval in Python and runs confirm tools
end to end. The MCP route is one call, for tools that need no approval. There is no AI SDK
for Python.

## Our side

```python
--8<-- "examples/connect_server.py:public"
```

## Their side: TS client (recommended)

```ts
import * as ai from "ai";
import { ToolClient } from "@agent-adapter/client";

const client = new ToolClient({ baseUrl: "https://tools.example.com", token: process.env.AA_AGENT_TOKEN });
const tools = await client.asVercelTools(ai.tool, ai.jsonSchema, { runId });

const result = await ai.generateText({ model, tools, prompt: "File my 2026 return." });
```

`asVercelTools` builds each tool with `description`, `inputSchema: jsonSchema(...)` and
`execute`. A confirm tool's `execute` returns `{ pending_approval: approval_id }`. Your
reviewer screen then calls `client.decide(approval_id, { approved })`, which runs the call
and returns its result. Watch the run with `client.events(runId)`.

## Their side: MCP

```ts
import { createMCPClient } from "@ai-sdk/mcp";

const mcp = await createMCPClient({
  transport: {
    type: "http",
    url: "https://tools.example.com/mcp",
    headers: { Authorization: `Bearer ${process.env.AA_AGENT_TOKEN}` },
  },
});
try {
  const { send_invoice, ...tools } = await mcp.tools(); // confirm tools stay out
  await ai.generateText({ model, tools, prompt: "Shipping rate to the US?" });
} finally {
  await mcp.close();
}
```

| Key | Meaning |
|---|---|
| `transport.type: "http"` | Streamable HTTP |
| `transport.url`, `transport.headers` | our `/mcp` endpoint and the bearer token |
| `transport.authProvider` | OAuth; our `bearer_auth` takes static tokens, so use `headers` |
| `toolApproval` on `generateText` / `streamText` | AI SDK 7's approval policy (`needsApproval` is deprecated) |

## Checklist

| Check | Why |
|---|---|
| `toolApproval` is a UI gate | it adds a step; our executor still decides `confirm` tools |
| confirm tools go through the TS client | over MCP the AI SDK does not send our resume keys back |
| the token stays on the server (route handler, server action) | the server sends no CORS headers, and a browser bundle leaks it |
| MCP tool annotations are hints | the AI SDK does not turn them into a policy; neither do we |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- AI SDK MCP tools: <https://ai-sdk.dev/docs/ai-sdk-core/mcp-tools>
- AI SDK tool approvals: <https://ai-sdk.dev/docs/agents/tool-approvals>
- AI SDK 7 release: <https://vercel.com/changelog/ai-sdk-7>
- Our TS client: `clients/typescript/README.md`
