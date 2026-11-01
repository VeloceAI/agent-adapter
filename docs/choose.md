# Which feature do I need?

Answer the questions top to bottom. Each leaf is a page. Most apps need the first three
rows only: a registry, an executor, and the surface that calls it.

```mermaid
flowchart TD
  A{"Do you own the function's code?"} -- yes --> T1["@registry.tool"]
  A -- "no: a library, a class, a service" --> T2["registry.wrap(fn) · import_openapi · import_mcp"]
  T1 --> B{"Who calls the tools?"}
  T2 --> B
  B -- "your code" --> C1["Executor.run / run_call"]
  B -- "a model API" --> C2["renderers + run_tool_loop"]
  B -- "an agent framework" --> C3["integrations.&lt;framework&gt;"]
  B -- "another language or process" --> C4["HTTP server · MCP"]
  C1 --> D{"Can a call do harm?"}
  C2 --> D
  C3 --> D
  C4 --> D
  D -- "yes, decide now" --> D1["confirm= + CallbackApproval"]
  D -- "yes, a person decides later" --> D2["confirm= + DeferredApproval + a store"]
  D -- no --> E{"Does it cost money?"}
  D1 --> E
  D2 --> E
  E -- yes --> E1["Ledger + Prices + BudgetHook"]
  E -- no --> F{"Same call again and again?"}
  E1 --> F
  F -- yes --> F1["cache_ttl + cache_scope · executor.model(cache_ttl=)"]
  F -- no --> G{"Several steps?"}
  F1 --> G
  G -- "independent calls" --> G1["run_many · as_completed"]
  G -- "steps that depend on each other" --> G2["run_plan"]
  G -- "branches, loops, pauses" --> G3["Flow + FlowRunner (or compile to LangGraph / ADK)"]
  G -- "one call at a time" --> H["done"]
```

## Where each leaf is explained

| Leaf | Page |
|---|---|
| `@registry.tool` | [Quickstart](get-started/quickstart.md), [Tools](concepts/tools.md) |
| `registry.wrap`, importers | [Wrap existing code](get-started/wrap-existing-code.md), [OpenAPI](integrations/openapi.md), [MCP](integrations/mcp.md) |
| `Executor.run`, `run_call` | [Executor](concepts/executor.md) |
| renderers, `run_tool_loop` | [OpenAI](integrations/openai.md), [Anthropic](integrations/anthropic.md), [Tool loop](concepts/tool-loop.md) |
| a framework | [Integrations](integrations/index.md), [support matrix](support-matrix.md) |
| another language | [HTTP server](cross-language/http-server.md), [MCP](cross-language/mcp.md) |
| a non-Python framework or a hosted agent | [Connect anywhere](connect/index.md) |
| `CallbackApproval`, `DeferredApproval` | [Human in the loop](how-to/human-in-the-loop.md), [Approval over Slack or email](cookbook/approval-notify.md) |
| ledger and budgets | [Budgets](how-to/budgets.md), [Cap spend per user](cookbook/spend-per-user.md) |
| caching | [Caching](how-to/caching.md) |
| batches and plans | [Parallel retrieval](how-to/parallel-retrieval.md) |
| flows | [First flow](get-started/first-flow.md), [Flows](concepts/flows.md) |
| sync code (Flask, Celery, scripts) | [Sync code and backends](how-to/sync-backends.md) |
| a queue, a Lambda or a pipeline | [Queues](backends/queues.md), [Serverless](backends/serverless.md), [Pipelines](backends/pipelines.md) |
| see what ran: flow graph, timeline, approvals (any framework) | [See your runs](how-to/see-your-runs.md) |

## Always on

Whatever you pick, every call through the executor gets these:

| You get | Without asking |
|---|---|
| argument validation | before the tool runs; errors safe to send back to the model |
| typed errors with stable `kind`s | [Errors](concepts/errors.md) |
| events for tracing | [Observability](guides/observability.md) |
| a denied call when a `confirm` tool has no approval port | [Security](guides/security.md) |
