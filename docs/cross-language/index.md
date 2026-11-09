# Cross-language

Your tools stay Python functions in one `Registry`. Other languages reach them over HTTP;
the executor's policies (hooks, approval, limits, cache, ledger, events) run on every call.

```mermaid
flowchart LR
  REG[("Registry")] --> MCP["/mcp<br/>stateless Streamable HTTP"]
  REG --> HTTP["HTTP tool server<br/>/tools · /approvals · /runs/{id}/events"]
  HTTP --> OAS["/openapi.json"]
  MCP --> ANY["MCP clients in every language<br/>and agent frameworks' MCP adapters"]
  HTTP --> TS["TypeScript client"]
  OAS --> GEN["generated clients<br/>(openapi-generator, Kiota, oapi-codegen)"]
```

## Pick a surface

| Surface | Use it when | Page |
|---|---|---|
| MCP | the caller is an agent framework or has an MCP SDK (most do) | [MCP](mcp.md) |
| HTTP tool server | approval must be decided in Python (with edits), you need run events as SSE, cost per run, or provider-shaped catalogues | [HTTP server](http-server.md) |
| OpenAPI | the caller generates a typed client from a spec | [OpenAPI](openapi.md) |
| TypeScript client | a TS / JS app calls the HTTP server | [TypeScript client](typescript-client.md) |

## Every backend language

MCP has official SDKs in ten languages, so each one connects natively. Any other language
calls the HTTP routes, or a client generated from `/openapi.json`.

| Language | Native route | Plain HTTP |
|---|---|---|
| :brand-typescript: TypeScript | MCP TypeScript SDK, or our [TypeScript client](typescript-client.md) | `fetch` |
| :brand-python: Python | in process, no server needed | `httpx` |
| :brand-csharp: C# / .NET | MCP C# SDK, Microsoft Agent Framework ([recipe](../connect/maf-dotnet.md)) | `HttpClient` |
| :brand-java: Java | MCP Java SDK, Spring AI ([recipe](../connect/spring-ai.md)) | `java.net.http` |
| :brand-kotlin: Kotlin | MCP Kotlin SDK | Ktor client |
| :brand-go: Go | MCP Go SDK, ADK Go ([recipe](../connect/adk-go.md)) | `net/http` |
| :brand-rust: Rust | MCP Rust SDK (`rmcp`) | `reqwest` |
| :brand-ruby: Ruby | MCP Ruby SDK | `Net::HTTP` |
| :brand-php: PHP | MCP PHP SDK | Guzzle |
| :brand-swift: Swift | MCP Swift SDK | `URLSession` |
| :brand-openapi: Anything else | | any HTTP client, or a generated one |

Every route runs the same executor, so approvals, budgets, cost and traces apply the same way.

One app serves both: `create_app(registry)` mounts MCP at `/mcp` next to the HTTP routes,
behind the same authenticator. That default needs both the `server` and `mcp` extras.

## Install

| Extra | Gives |
|---|---|
| `agent-adapter[mcp]` | `streamable_http_app`, `mcp_server`, the importer |
| `agent-adapter[server]` | `create_app`, `serve` (Starlette + uvicorn); includes `mcp` for the default `/mcp` mount |
| `agent-adapter[server-wsgi]` | `create_wsgi_app` (adds `a2wsgi`) |

Install with `pip install "agent-adapter[server]"` (the extras you need).

`ToolService` itself needs no extra: it is the server's behaviour with no web framework, so
you can mount it in any framework you already run.
