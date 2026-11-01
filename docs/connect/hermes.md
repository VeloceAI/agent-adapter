# Connect: Hermes Agent

:material-check-circle:{ .status-ok } Recipe · through our [MCP server](../cross-language/mcp.md) · their docs checked 2026-10-30

Hermes Agent (Nous Research) has a built-in MCP client. It loads our tools from one entry
in its config file. Data enters at Hermes' tool call and leaves as the MCP result.

## Our side

On the same machine, in development:

```python
--8<-- "examples/connect_server.py:local"
```

Anywhere else, the public app:

```python
--8<-- "examples/connect_server.py:public"
```

## Their side

`~/.hermes/config.yaml`:

```yaml
mcp_servers:
  shop_tools:
    url: "https://tools.example.com/mcp"   # local: http://127.0.0.1:8000/mcp
    headers:
      Authorization: "Bearer ${AA_AGENT_TOKEN}"   # the agent token, never the reviewer's
    timeout: 30            # seconds per tool call
    connect_timeout: 30    # seconds for the first handshake
    tools:
      exclude: [send_invoice]   # confirm tools: Hermes cannot resume them
```

| Key | Meaning |
|---|---|
| `mcp_servers.<name>.url` | our `/mcp` endpoint |
| `headers` | sent on every request; carries the bearer token |
| `timeout`, `connect_timeout` | call and handshake limits, in seconds |
| `tools.include` / `tools.exclude` | allow or block tools by name (exclude takes globs) |
| `auth: oauth` | Hermes' OAuth 2.1 flow; our `bearer_auth` takes static tokens, so use `headers` |

## Checklist

| Check | Why |
|---|---|
| the token in `headers` maps to its own `Principal` | Hermes' runs and approvals stay apart from other hosts' |
| exclude confirm tools, or serve `hosted` | Hermes does not send our resume keys back; a parked call never runs |
| Hermes' own "Allow Once / Deny" prompt is extra | it adds a gate; our executor still decides `confirm` tools |
| everything on the [public-host checklist](index.md#public-host-checklist-every-page) | |

## Sources

- Hermes Agent MCP guide: <https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp>
- Hermes Agent repository: <https://github.com/NousResearch/hermes-agent>
