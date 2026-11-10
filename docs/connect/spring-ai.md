---
title: "Spring AI: call Python tools from Java over MCP"
description: "Spring AI's MCP client Boot starter turns your Python tools into ToolCallbacks, so a Java ChatClient calls them like local tools."
---

# Connect: Spring AI

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-10-30

Spring AI's MCP client Boot starter connects at start-up and turns each of our tools into a
`ToolCallback`. Your `ChatClient` then calls them like local tools.

## Our side

```python
--8<-- "examples/connect_server.py:public"
```

## Their side

Dependency: `org.springframework.ai:spring-ai-starter-mcp-client` (or
`spring-ai-starter-mcp-client-webflux`).

`application.yml`:

```yaml
spring:
  ai:
    mcp:
      client:
        type: SYNC              # all clients sync or all async
        request-timeout: 30s    # default 20s
        streamable-http:
          connections:
            shop-tools:
              url: https://tools.example.com   # base URL
              endpoint: /mcp                   # default /mcp
```

The bearer token, on every request:

```java
@Bean
McpSyncHttpClientRequestCustomizer agentAdapterToken(@Value("${aa.agent-token}") String token) {
    return (builder, method, endpoint, body, context) ->
            builder.header("Authorization", "Bearer " + token);
}

@Bean
ChatClient chatClient(ChatClient.Builder builder, SyncMcpToolCallbackProvider mcpTools) {
    return builder.defaultTools(mcpTools).build();  // MCP tools are not added automatically
}
```

| Key | Meaning |
|---|---|
| `streamable-http.connections.<name>.url` | our server's base URL |
| `…connections.<name>.endpoint` | the path of our endpoint (`/mcp`) |
| `request-timeout` | per request (default `20s`) |
| `type` | `SYNC` or `ASYNC` for every client |
| `McpSyncHttpClientRequestCustomizer` | edits each outgoing request: adds the header |
| `McpToolFilter` bean | one bean; return `false` to drop a tool |

## Checklist

| Check | Why |
|---|---|
| the auto-configured client is one session for the whole app | every user of the app shares the token's principal and its runs; for per-user principals build clients yourself |
| drop confirm tools with an `McpToolFilter`, or serve `hosted` | Spring AI does not send our resume keys back |
| the token comes from config or a vault, not source | `${aa.agent-token}` |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- MCP client Boot starter: <https://docs.spring.io/spring-ai/reference/api/mcp/mcp-client-boot-starter-docs.html>
- MCP security (request customizers): <https://docs.spring.io/spring-ai/reference/api/mcp/mcp-security.html>
- Tool calling (`defaultTools` with MCP): <https://docs.spring.io/spring-ai/reference/api/tools.html>
- MCP Java SDK 2.0 migration (`httpRequestCustomizer`): <https://github.com/modelcontextprotocol/java-sdk/blob/main/MIGRATION-2.0.md>
