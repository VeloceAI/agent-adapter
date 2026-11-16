---
title: "Rig: use Python tools from Rust agents over MCP"
description: "Give a Rig agent your Python tools through our MCP server with the rmcp SDK, with approvals, budgets and traces on every call."
---

# Connect: Rig

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-11-16

Rig (Rust) reads MCP tools through the official Rust SDK, `rmcp`, behind its `rmcp`
feature. The transport below is the one our tested
[Rust client](https://github.com/VeloceAI/agent-adapter/tree/main/examples/clients/rust)
uses against our server.

## Our side

```python
--8<-- "examples/connect_server.py:public"
--8<-- "examples/connect_server.py:hosted"
```

Serve `public_app(hosted, …)`: Rig does not send our resume keys back. In development on
one machine, `local_app()` is enough.

## Their side

`Cargo.toml`: `rig` with `features = ["rmcp"]`, and `rmcp` with
`features = ["client", "transport-streamable-http-client-reqwest"]`, on the same `rmcp`
version Rig pins.

```rust
use rig::tool::rmcp::{client_config, McpConnection};
use rmcp::transport::{
    streamable_http_client::StreamableHttpClientTransportConfig, StreamableHttpClientTransport,
};
use rmcp::ClientServiceExt; // rmcp 3.9, the version Rig 0.46 pins

let token = std::env::var("AA_AGENT_TOKEN")?;
let config = StreamableHttpClientTransportConfig::with_uri("https://tools.example.com/mcp")
    .auth_header(token); // the token alone: rmcp adds "Bearer "
let transport = StreamableHttpClientTransport::from_config(config);

let client = client_config("shop-agent", "1.0.0").serve(transport).await?;
let connection = McpConnection::new(client);
let tools = connection.tools().await?; // each converts into a rig DynamicTool
```

| Key | Meaning |
|---|---|
| `StreamableHttpClientTransportConfig::with_uri(...)` | our `/mcp` endpoint |
| `auth_header(token)` | sends `Authorization: Bearer <token>` on every request |
| `client_config(name, version).serve(transport)` | Rig's client info; starts the MCP session |
| `McpConnection::tools()` | our tools, ready for the agent's tool set |
| `McpClientHandler` with a `ToolServerHandle` | for agents: keeps the tools in sync when the list changes |

Keep the `McpConnection` alive as long as the agent uses its tools: dropping it closes the
session. Rig's MCP crate moved in recent releases; check the paths against the `rig-rmcp`
README for your version.

## Checklist

| Check | Why |
|---|---|
| serve `hosted` | Rig does not send our resume keys back; a confirm tool would never run |
| one `rmcp` version in the tree | two versions give two incompatible `Tool` types |
| the token comes from the environment or a secret store | never in source or `Cargo.toml` |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- Rig MCP crate: <https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-rmcp>
- rmcp transport config: <https://docs.rs/rmcp/latest/rmcp/transport/streamable_http_client/struct.StreamableHttpClientTransportConfig.html>
