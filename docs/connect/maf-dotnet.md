---
title: "Microsoft Agent Framework .NET: Python tools over MCP"
description: "Give a .NET Microsoft Agent Framework agent your Python tools as AITools through the official MCP C# SDK, with approvals and cost tracking."
---

# Connect: Microsoft Agent Framework (.NET)

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-10-30

MAF .NET uses the official MCP C# SDK. Your process opens an MCP client to our server,
lists the tools and hands them to the agent as `AITool`s. The calls go from your process
to us; Foundry never sees our URL. (For a Foundry-hosted agent, see
[Azure AI Foundry](azure-foundry.md).)

## Our side

```python
--8<-- "examples/connect_server.py:public"
```

In development on one machine, `local_app()` and `http://127.0.0.1:8000/mcp` are enough.

## Their side

Packages: `ModelContextProtocol` and `Microsoft.Agents.AI.Foundry` (or another model provider).

```csharp
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;

await using var mcpClient = await McpClient.CreateAsync(new HttpClientTransport(new()
{
    Endpoint = new Uri("https://tools.example.com/mcp"),
    TransportMode = HttpTransportMode.StreamableHttp,
    AdditionalHeaders = new Dictionary<string, string>
    {
        ["Authorization"] = $"Bearer {Environment.GetEnvironmentVariable("AA_AGENT_TOKEN")}",
    },
}));
var mcpTools = await mcpClient.ListToolsAsync();
var safeTools = mcpTools.Where(t => t.Name != "send_invoice");  // confirm tools stay out

AIAgent agent = new AIProjectClient(new Uri(endpoint), new DefaultAzureCredential())
    .AsAIAgent(
        model: deploymentName,
        instructions: "You answer customer-support questions.",
        tools: [.. safeTools.Cast<AITool>()]);
```

| Name | Meaning |
|---|---|
| `HttpClientTransportOptions.Endpoint` | our `/mcp` URL (required) |
| `TransportMode = HttpTransportMode.StreamableHttp` | skip the SSE fallback that `AutoDetect` tries |
| `AdditionalHeaders` | sent on every request: the bearer token |
| `ListToolsAsync()` | our tools as `McpClientTool` (an `AIFunction`) |
| `AsAIAgent(model:, instructions:, tools:)` | the agent with those tools (any MAF chat client works the same way) |

## Checklist

| Check | Why |
|---|---|
| leave confirm tools out of `tools`, or serve `hosted` | MAF does not send our resume keys back; a parked call never runs |
| MAF's own approval modes are extra | it adds a gate; our executor still decides `confirm` tools |
| one `McpClient` per principal | the header is the identity: do not share a client between users |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- MAF, using MCP tools: <https://learn.microsoft.com/en-us/agent-framework/agents/tools/local-mcp-tools>
- MCP C# SDK transports: <https://csharp.sdk.modelcontextprotocol.io/concepts/transports/transports.html>
- `HttpClientTransportOptions`: <https://csharp.sdk.modelcontextprotocol.io/api/ModelContextProtocol.Client.HttpClientTransportOptions.html>
