---
title: "LangChain4j: use Python tools from Java over MCP"
description: "Give a LangChain4j AI Service your Python tools through our MCP server, with approvals, budgets and traces on every call."
---

# Connect: LangChain4j

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-11-17 · MCP route tested live

LangChain4j's `langchain4j-mcp` module connects an AI Service to any MCP server. Point its
Streamable HTTP transport at our `/mcp` endpoint and the model sees our tools next to its
own `@Tool` methods.

## Our side

```python
--8<-- "examples/connect_server.py:public"
--8<-- "examples/connect_server.py:hosted"
```

Serve `public_app(hosted, …)`: LangChain4j does not send our resume keys back, so a
confirm tool would park on every call. In development on one machine, `local_app()` is
enough.

## Their side

Add `dev.langchain4j:langchain4j-mcp` (a beta module: use the version that matches your
LangChain4j BOM):

```java
import dev.langchain4j.mcp.McpToolProvider;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.http.StreamableHttpMcpTransport;
import dev.langchain4j.service.AiServices;

McpTransport transport = StreamableHttpMcpTransport.builder()
    .url("https://tools.example.com/mcp")
    .customHeaders(Map.of("Authorization", "Bearer " + System.getenv("AA_AGENT_TOKEN")))
    .build();

McpClient mcp = new DefaultMcpClient.Builder()
    .key("shopTools")
    .transport(transport)
    .build();

McpToolProvider tools = McpToolProvider.builder()
    .mcpClients(mcp)
    .filterToolNames("shipping_rate")
    .build();

Assistant assistant = AiServices.builder(Assistant.class)
    .chatModel(model)
    .toolProvider(tools)
    .build();
```

| Key | Meaning |
|---|---|
| `StreamableHttpMcpTransport.builder().url(...)` | our `/mcp` endpoint |
| `customHeaders(...)` | carries the bearer token on every request, POST and GET |
| `DefaultMcpClient.Builder().key(...)` | names the client; one per server |
| `McpToolProvider.builder().filterToolNames(...)` | the tools the model may see |
| `AiServices.builder(...).toolProvider(...)` | offers the tools to your AI Service |

Close the client (`mcp.close()`) when the app stops. A failed call comes back as a tool
error the model can read; our `ToolError` messages reach it as text.

## Checklist

| Check | Why |
|---|---|
| serve `hosted`, or filter to tools without `confirm` | LangChain4j does not send our resume keys back |
| keep the default `followRedirects(false)` | a redirect could send the bearer header to another host |
| the token comes from the environment or a secret store | never in source or a config file in git |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- LangChain4j MCP tutorial: <https://docs.langchain4j.dev/tutorials/mcp>
- `StreamableHttpMcpTransport` source: <https://github.com/langchain4j/langchain4j/tree/main/langchain4j-mcp>
