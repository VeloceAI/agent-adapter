---
title: "C++, Scala, Zig and Lua: call Python tools over REST"
description: "Call your Python tools from C++, Scala, Zig or Lua over plain REST, or with a client generated from our OpenAPI document."
---

# C++, Scala, Zig and Lua

:material-check-circle:{ .status-ok } Available · the [HTTP server](http-server.md) and its [OpenAPI](openapi.md) document

These four languages have no official MCP SDK, so they reach your tools over plain REST.
Approvals, budgets, cost and traces still run on every call: they live in the executor,
not the client. Each call is one JSON POST, so any HTTP client works; a generated client
adds types.

## The whole contract

```bash
curl -s https://tools.example.com/tools/get_weather/invoke \
  -H "Authorization: Bearer $AA_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: order-1042-weather" \
  -d '{"arguments": {"city": "Paris"}, "run_id": "run_42"}'
```

| Answer | Body | What to do |
|---|---|---|
| 200 | `{"status": "ok", "result": ..., "run_id", "call_id"}` | use `result` |
| 202 | `{"status": "pending_approval", "approval_id": "apr_…"}` | a person decides with `POST /approvals/{id}/decide`; `decide` runs the call and returns the 200 body |
| 4xx / 5xx | RFC 9457 `application/problem+json` with `type`, `detail` | `422`: send fixed arguments; `401`: check the token |

`Idempotency-Key` makes a retry safe: the same key replays the first answer and never runs
the tool twice. `GET /tools` lists every tool with its input schema.

## Generate a typed client

Generators want OpenAPI 3.0: download `GET /openapi-3.0.json` (with your token) and run
[OpenAPI Generator](https://openapi-generator.tech/docs/generators/):

```bash
docker run --rm -v "$PWD:/local" openapitools/openapi-generator-cli generate \
  -i /local/openapi-3.0.json -g cpp-restsdk -o /local/client-cpp
```

| Language | Generator (`-g`) | Plain HTTP instead |
|---|---|---|
| :brand-cplusplus: C++ | `cpp-restsdk`, `cpp-oatpp-client`, `cpp-qt-client`, `cpp-boost-beast-client` | libcurl, cpr |
| :brand-scala: Scala | `scala-sttp4` (beta), `scala-sttp`, `scala-pekko` | sttp, `java.net.http` |
| :brand-lua: Lua | `lua` (beta) | lua-resty-http (OpenResty), LuaSocket with LuaSec |
| :brand-zig: Zig | none yet | `std.http.Client` from the standard library |

Each tool becomes one `invoke_<tool>` operation whose request body is typed from the
tool's Python signature, so a renamed argument shows up as a compile error, not a 422.
Regenerate when the registry changes: the document is built from it on every request.

## Rules a client must follow

| Rule | Why |
|---|---|
| send `Authorization: Bearer <token>` on every call | the server maps the token to a principal; runs and approvals belong to it |
| treat 202 as "waiting", never as success | the tool has not run yet |
| reuse one `Idempotency-Key` per logical call | a network retry must not send a second invoice |
| keep `run_id` the same across one agent run | budgets and the run's cost and events are per run |
| never put the token in a URL | logs and proxies keep URLs |

The approver token is a different one: give it to your reviewer screen, never to the
agent, so an agent cannot approve its own calls.

## When MCP arrives

If an MCP SDK lands for your language, switch the agent to our `/mcp` endpoint on the same
server; the token and the tools stay the same ([MCP](mcp.md)). Elixir and Dart already
have community SDKs: see their clients in
[`examples/clients`](https://github.com/VeloceAI/agent-adapter/tree/main/examples/clients).
