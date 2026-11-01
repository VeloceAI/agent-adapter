---
search:
  boost: 2
---

# Glossary

One line per term, in alphabetical order. Each links the page that explains it.

| Term | Meaning |
|---|---|
| **Approval** | A reviewer's yes or no on one tool call before it runs. See [Approval](approval.md). |
| **Approval id** | The id of a pending approval request; you resume with it. See [Human in the loop](../how-to/human-in-the-loop.md). |
| **Approval port** | The object that decides approvals: `AutoApprove`, `AutoDeny`, `CallbackApproval`, `DeferredApproval`, or your own `ApprovalPort`. See [Approval](approval.md). |
| **Approval store** | Where pending approvals and decisions live: memory, SQLite or Redis. See [Approval stores](approval-stores.md). |
| **Bridge** | Integration code that maps a framework's own approval or tool protocol onto the executor. See [Integrations](../integrations/index.md). |
| **Cache scope** | How widely a cached tool result is shared: `run`, `thread` or `global`. See [Cache](cache.md). |
| **Call id** | The id of one tool call (`ToolContext.call_id`). See [Context](context.md). |
| **Confirm rule** | A tool's `confirm` setting: `True`, `False` or a function that decides per call whether approval is needed. See [Approval](approval.md). |
| **Context** | The `ToolContext` passed to a tool: run, thread and call ids, state, ledger, cache and event sink. See [Context, events and ports](context.md). |
| **Decision** | An `ApprovalDecision`: approved or not, who decided, and why. `Decision` is an alias. See [Approval](approval.md). |
| **Event** | A structured record of something that happened (call started, call failed, ...). See [Tracing](tracing.md). |
| **Event sink** | Where events go: logs, a callback, OpenTelemetry. See [Tracing](tracing.md). |
| **Executor** | Runs tool calls: validation, hooks, approval, retries, timeouts, cache and limits. See [Executor](executor.md). |
| **Flow** | A graph of nodes the `FlowRunner` runs, or compiles for a framework. See [Flows](flows.md). |
| **Flow pause** | A flow stops at a node that waits for approval and resumes later from its checkpoint. See [Flow pause and resume](flow-resume.md). |
| **Hook** | Code that runs before, after or on error of a tool, model or node call. See [Hooks](hooks.md). |
| **Importer** | A function that turns foreign tools into `ToolSpec`s: `import_mcp`, `import_openapi`, `import_langchain`. See [Integrations](../integrations/index.md). |
| **Integration** | A module in `agent_adapter.integrations` that exports tools to one model API or framework. See [Integrations](../integrations/index.md). |
| **Kind (error)** | The stable id of an error, such as `tool-not-found`. See the [error catalogue](../reference/errors.md). |
| **Ledger** | Records the cost of every model and tool call as `LedgerEntry` rows. See [Cost](cost.md). |
| **One-use token** | An approval the adapter issues after its own port approved, bound to one run and call and spent once. A framework's own approval never counts alone. See [Approval](approval.md). |
| **Port** | A protocol you implement to plug in a backend: `LLMPort`, `CachePort`, `LedgerPort`, `EventSink`, `ApprovalPort`, `EmbeddingPort`. See [Context, events and ports](context.md). |
| **Portable schema** | A JSON schema every supported provider accepts; `SchemaNotPortable` says what breaks it. See [Schemas](schemas.md). |
| **Price** | The per-token cost of a model, from `Prices`. See [Cost](cost.md). |
| **Principal** | An authenticated HTTP caller; its id scopes the runs, calls and approvals it can see. See [HTTP server](../cross-language/http-server.md). |
| **Problem** | An RFC 9457 error body the HTTP server returns, with the error's kind in `type`. See [HTTP server](../cross-language/http-server.md). |
| **Registry** | Holds the tools you define and exports them to every target. See [Registry](registry.md). |
| **Run id** | The id shared by every call in one agent run. Run limits count per run. See [Context](context.md). |
| **ShortCircuit** | Raised by a hook to return a result without calling the tool or model. See [Hooks](hooks.md). |
| **Spec** | A `ToolSpec`: a tool's name, description, argument model and settings. See [Tools](tools.md). |
| **Strict mode** | OpenAI schema mode: every property is required, so an optional argument becomes nullable and `null` means "not given". See [Strict mode and nulls](strict-mode.md). |
| **Target** | A model API or framework a tool is exported to; `targets` limits a tool to some of them. See [Registry](registry.md). |
| **Thread id** | The id of a conversation that spans several runs. See [Context](context.md). |
| **Tool** | A Python function registered with `@tool` or `wrap`, callable by any model or framework. See [Tools](tools.md). |
| **Tool loop** | Model call, tool calls, results back to the model, repeat: `run_tool_loop`. See [Tool loop](tool-loop.md). |
