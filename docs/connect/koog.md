---
title: "Koog: use Python tools from Kotlin agents over MCP"
description: "Load your Python tools into a JetBrains Koog agent's ToolRegistry through our MCP server, with approvals and cost tracking on every call."
---

# Connect: Koog

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-11-17 · MCP route tested live

Koog (JetBrains, Kotlin) turns an MCP server into a `ToolRegistry` with
`McpToolRegistryProvider`. Its `streamableHttp { }` helper sets no request headers; to send
our bearer token, hand it the official Kotlin SDK's transport, which is what our tested
[Kotlin client](https://github.com/VeloceAI/agent-adapter/tree/main/examples/clients/kotlin)
uses.

## Our side

```python
--8<-- "examples/connect_server.py:public"
--8<-- "examples/connect_server.py:hosted"
```

Serve `public_app(hosted, …)`: Koog does not send our resume keys back. In development on
one machine, `local_app()` is enough.

## Their side

Add Koog (`ai.koog:koog-agents`), its MCP module (`ai.koog:agents-mcp`, a beta module:
`1.3.0-beta` for Koog 1.3.0) and a Ktor engine such as `io.ktor:ktor-client-cio`:

```kotlin
import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.mcp.McpToolRegistryProvider
import ai.koog.agents.mcp.metadata.McpServerInfo
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.sse.SSE
import io.ktor.client.request.header
import io.modelcontextprotocol.kotlin.sdk.client.StreamableHttpClientTransport

val http = HttpClient(CIO) { install(SSE) }
val transport = StreamableHttpClientTransport(http, "https://tools.example.com/mcp") {
    header("Authorization", "Bearer ${System.getenv("AA_AGENT_TOKEN")}")
}

val toolRegistry = McpToolRegistryProvider.fromTransport(
    transport = transport,
    serverInfo = McpServerInfo(url = "https://tools.example.com/mcp"),
    name = "shop-agent",
    version = "1.0.0",
)

val agent = AIAgent(
    promptExecutor = executor,
    llmModel = OpenAIModels.Chat.GPT4o,
    toolRegistry = toolRegistry,
)
```

| Key | Meaning |
|---|---|
| `StreamableHttpClientTransport(http, url) { ... }` | the Kotlin SDK's Streamable HTTP transport; the block runs on every request |
| `header("Authorization", ...)` | the bearer token |
| `McpToolRegistryProvider.fromTransport(...)` | connects, lists our tools and wraps each as a Koog tool; `serverInfo` is required |
| `McpToolRegistryProvider.fromClient(...)` | the same, from an MCP `Client` you already connected |
| `AIAgent(toolRegistry = ...)` | offers the tools to the agent |

`defaultSseTransport(url)` speaks the older SSE transport, which our server does not
serve: use the Streamable HTTP transport above.

## Checklist

| Check | Why |
|---|---|
| serve `hosted` | Koog does not send our resume keys back; a confirm tool would never run |
| one `HttpClient` per app, closed on shutdown | the transport keeps a connection open for server messages |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- Koog MCP guide: <https://docs.koog.ai/model-context-protocol/>
- Kotlin MCP SDK client transport: <https://github.com/modelcontextprotocol/kotlin-sdk>
