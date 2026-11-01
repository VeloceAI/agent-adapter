# Cookbook

Short recipes for tasks people ask about. Each one is a problem, one tested example, a
diagram and the pitfalls. The how-to guides explain a feature; a recipe solves one task.

```mermaid
flowchart LR
  T["your task"] --> R{"which recipe?"}
  R --> A["a tool fails sometimes"] --> R1["Retry a flaky tool"]
  R --> B["spend control"] --> R2["Cap spend per user"]
  R --> C["secrets and PII"] --> R3["Redact arguments and results"]
  R --> D["several sources"] --> R4["Parallel retrieval with a deadline"]
  R --> E["a person must say yes"] --> R5["Approval over Slack or email"]
  R --> F["repeat model calls"] --> R6["Cache model answers across workers"]
  R --> G["tests"] --> R7["Test tools with fakes"]
  R --> H["SaaS plans"] --> R8["A registry per tenant"]
```

| Recipe | Problem it solves | Main pieces |
|---|---|---|
| [Retry a flaky tool](retry-flaky-tool.md) | a network call fails now and then | `retries=`, `timeout=`, `Backoff`, a retryable `ToolError` |
| [Cap spend per user](spend-per-user.md) | one user must not burn the month's budget | `BudgetHook(scope="global")`, one `Ledger` per user and day |
| [Redact arguments and results](redact.md) | keys, emails and card numbers must not reach logs or the model | `LoggingHook(redact=)`, `redact()`, an `after_tool` hook |
| [Parallel retrieval with a deadline](parallel-deadline.md) | answer within N seconds, whatever is slow | `run_many`, `as_completed`, `asyncio.timeout` |
| [Approval over Slack or email](approval-notify.md) | a reviewer decides in a chat app or mail | `DeferredApproval`, the `tool.pending` event, a signed callback |
| [Cache model answers across workers](cache-llm.md) | the same question is paid for again and again | `executor.model(port, cache_ttl=)`, `RedisCache` |
| [Test tools with fakes](test-with-fakes.md) | a tool calls a real service | clients in `ctx.extras`, a fake client, injected `sleep` |
| [A registry per tenant](tenant-registry.md) | each plan sees a different tool set | `Registry.subset(tags=)`, `run_call(..., registry=)` |

Every recipe's code is a file under `examples/cookbook_*.py`.
`tests/docs/test_cookbook.py` runs each one offline and checks what it prints.
