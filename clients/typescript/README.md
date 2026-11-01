# @agent-adapter/client

Zero-dependency TypeScript client for the agent-adapter HTTP tool server: call Python tools
from Node, Deno, Bun or the browser, with server-side approval, run events and cost.

- Node ≥ 20 (or any runtime with `fetch`, `ReadableStream`, `TextDecoder`), ESM and CJS.
- Every route of the server: catalogue, invoke, approvals, events (SSE), cost, health.
- Typed errors for every RFC 9457 problem the server returns.
- Adapters for the Vercel AI SDK and other tool shapes.

The package is `private` for now: where it is published is still to be decided. Use it from
this repository (a workspace or `file:` dependency) until then.

In a browser the server must allow CORS for your origin, including the `Authorization`,
`Idempotency-Key` and `Last-Event-ID` request headers in its preflight answer.

```ts
import { ToolClient, ToolArgsInvalidError } from "@agent-adapter/client";

const client = new ToolClient({ baseUrl: "https://tools.internal/api", token: process.env.TOOLS_TOKEN });
const tools = await client.listTools({ shape: "openai" });

const call = await client.invoke<{ sum: number }>("add", { a: 2, b: 3 }, { runId: "run_42" });
if (call.status === "pending_approval") {
  await client.decide(call.approval_id, { approved: true, reason: "checked" });
}

for await (const event of client.events("run_42", { closeOn: ["tool.finished"] })) {
  console.log(event.event, event.data);
}
```

## Options

| Option            | Default                                                   | Meaning                                                               |
| ----------------- | --------------------------------------------------------- | --------------------------------------------------------------------- |
| `baseUrl`         | required                                                  | server root, including any prefix                                     |
| `token`           | none                                                      | bearer token, or a (async) function called before every attempt       |
| `timeoutMs`       | 30 000                                                    | per attempt; for `events` only until the headers arrive               |
| `headers`         | `{}`                                                      | extra headers on every request                                        |
| `retry`           | `{ maxRetries: 2, baseDelayMs: 250, maxDelayMs: 10_000 }` | `false` disables retries                                              |
| `maxEventSize`    | 1 MiB                                                     | longest SSE line or event, in characters                              |
| `idleTimeoutMs`   | 45 000                                                    | an event stream silent this long (keep-alives count) reconnects       |
| `maxResponseSize` | 16 MiB                                                    | largest JSON or error body read, in bytes                             |
| `fetch`           | global `fetch`                                            | custom fetch (proxies, tests); always called with `redirect: "error"` |

Redirects are never followed: a 3xx is a `ToolTransportError`, so the bearer token never goes to another URL.

`invoke` and `decide` wait for the tool to finish: raise `timeoutMs` for tools that run longer than
30 s, and watch long-running tools with `events()`.

## Retries and idempotency

`invoke` and `decide` always send an `Idempotency-Key` (yours, or a random UUID), the same on
every retry of that call; the server replays the first response for a repeated key.

| Request            | Retried on                                                                       |
| ------------------ | -------------------------------------------------------------------------------- |
| `GET`              | network errors, timeouts, 408, 429, 502, 503, 504                                |
| `invoke`, `decide` | connection never established (`ECONNREFUSED`, DNS), 429 / 503 with `Retry-After` |

A POST is never retried after a timeout, a reset connection, 502 or 504: the tool may have run.
`Retry-After` is honoured up to `maxDelayMs`; a longer one is not waited for (the error carries
`retryAfterMs`).

## Errors

| Class                                                                                                                                                  | When                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| `ToolServerError` (and subclasses: `ToolNotFoundError`, `ToolArgsInvalidError`, `ApprovalDeniedError`, `BudgetExceededError`, `ToolTimeoutError`, ...) | non-2xx response; `.problem`, `.status`, `.kind`             |
| `ToolTransportError`                                                                                                                                   | no response: `reason` is `"network"` or `"timeout"`          |
| `ToolProtocolError`, `EventStreamError`                                                                                                                | the response breaks the protocol (bad JSON, oversized event) |

Aborting your `signal` rejects with the signal's reason. Tokens and idempotency keys are never
logged or put in error messages.

## Development

```sh
npm ci
npm test            # node --test, fetch mocked, offline
npm run typecheck   # tsc
npm run build       # tsdown -> dist/ (ESM + CJS + .d.ts)
npm run lint:package  # publint + attw
AA_BASE_URL=http://127.0.0.1:8000 AA_TOKEN=... npm test   # also runs test/live.test.ts
```

Source uses only erasable TypeScript (no enums, namespaces or parameter properties), so Node
≥ 22.18 runs the tests straight from `.ts`.
