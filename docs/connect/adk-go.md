---
title: "ADK Go: call Python tools over MCP"
description: "Connect a Go ADK agent to your Python tools through mcptoolset and the official MCP Go SDK, with approvals and cost tracking on the server."
---

# Connect: ADK Go

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-10-30

ADK Go's `mcptoolset` turns an MCP server into a `tool.Toolset` for an `llmagent`. It uses
the official MCP Go SDK for the connection.

## Our side

```python
--8<-- "examples/connect_server.py:public"
```

In development on one machine, `local_app()` and `http://127.0.0.1:8000/mcp` are enough.

## Their side

Module `google.golang.org/adk/v2` (v2.6.0) and `github.com/modelcontextprotocol/go-sdk`:

```go
import (
    "net/http"
    "os"

    "github.com/modelcontextprotocol/go-sdk/mcp"
    "google.golang.org/adk/v2/agent/llmagent"
    "google.golang.org/adk/v2/tool"
    "google.golang.org/adk/v2/tool/mcptoolset"
)

type bearer struct{ token string }

func (b bearer) RoundTrip(r *http.Request) (*http.Response, error) {
    r = r.Clone(r.Context())
    r.Header.Set("Authorization", "Bearer "+b.token)
    return http.DefaultTransport.RoundTrip(r)
}

shopTools, err := mcptoolset.New(mcptoolset.Config{
    Transport: &mcp.StreamableClientTransport{
        Endpoint:   "https://tools.example.com/mcp",
        HTTPClient: &http.Client{Transport: bearer{os.Getenv("AA_AGENT_TOKEN")}},
    },
    RequireConfirmationProvider: func(name string, input any) bool {
        return name == "send_invoice" // ADK asks a person first: an extra gate
    },
})
// ...
agent, err := llmagent.New(llmagent.Config{ /* Name, Model, Instruction */ Toolsets: []tool.Toolset{shopTools}})
```

| Field | Meaning |
|---|---|
| `mcptoolset.Config.Transport` | how to reach us: a `*mcp.StreamableClientTransport` |
| `StreamableClientTransport.Endpoint` / `HTTPClient` | our `/mcp` URL / the client that adds the header |
| `Config.Endpoint`, `Config.Auth` | v2 shortcut: a URL and an `auth.CredentialProvider` instead of `Transport` |
| `RequireConfirmation`, `RequireConfirmationProvider` | ADK's own confirmation, for all tools or per tool |
| `llmagent.Config.Toolsets` | where the toolset goes |

## Checklist

| Check | Why |
|---|---|
| ADK's confirmation is extra | it adds a gate; our executor still decides `confirm` tools |
| filter confirm tools out (`tool.FilterToolset`), or serve `hosted` | ADK does not send our resume keys back |
| the header goes only to our host | a custom `RoundTripper` on its own `http.Client`, never on `http.DefaultClient` |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- `mcptoolset` (v2): <https://pkg.go.dev/google.golang.org/adk/v2/tool/mcptoolset>
- ADK MCP tools guide: <https://adk.dev/tools-custom/mcp-tools/>
- MCP Go SDK: <https://pkg.go.dev/github.com/modelcontextprotocol/go-sdk/mcp>
