---
title: "Koog: use Python tools from Kotlin agents over MCP"
description: "Load your Python tools into a JetBrains Koog agent's ToolRegistry through our MCP server, with approvals and cost tracking on every call."
---

# Connect: Koog

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-11-16

Koog (JetBrains, Kotlin) turns an MCP server into a `ToolRegistry` with
`McpToolRegistryProvider`. Its built-in helpers cover stdio and SSE; for our Streamable
HTTP endpoint, hand it the official Kotlin SDK's transport, which is what our tested
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

Add Koog with its MCP module (`ai.koog:koog-agents`; MCP ships as a beta module) and a
Ktor engine such as `io.ktor:ktor-client-cio`:

```kotlin
import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.mcp.McpToolRegistryProvider
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
| `McpToolRegistryProvider.fromTransport(...)` | connects, lists our tools and wraps each as a Koog tool |
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
