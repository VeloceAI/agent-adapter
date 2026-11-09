# Clients in every language

One Agent Adapter server, one small client per language. Each client calls the same Python
tool twice: over MCP with that language's official SDK, then over plain REST.

```bash
python scripts/check_clients.py            # every language (needs Docker)
python scripts/check_clients.py go rust    # just these
```

The server (`server/server.py`) is the released `agent-adapter[server]` package with one
tool, `get_weather`, behind a bearer token. Every client prints:

```text
MCP: Sunny in Paris, 21 C
REST: Sunny in Paris, 21 C
```

To call your own server, set `AA_URL` (such as `https://tools.yourcompany.com`) and
`AA_TOKEN`, then run the client you need from its folder.
