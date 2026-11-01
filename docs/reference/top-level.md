# Top level

The everyday names, imported from `agent_adapter`. Each loads on first use, so `import agent_adapter` stays cheap. Concepts: [Tools](../concepts/tools.md), [Registry](../concepts/registry.md), [Executor](../concepts/executor.md).

Four names have no docstring of their own, so here is what they are:

| Name | What it is |
|---|---|
| `default_registry` | The process-wide `Registry`, created on first access, then reused |
| `tool`, `wrap` | `default_registry.tool` and `default_registry.wrap`: register a function on it |
| `Decision` | An alias of `ApprovalDecision` |
| `__version__` | The installed version string |

::: agent_adapter
    options:
      members: [AdapterError, ApprovalDecision, ApprovalDenied, ApprovalPending, ApprovalRequest, AutoApprove, AutoDeny, BudgetHook, CallableModel, CallbackApproval, Decision, DeferredApproval, Event, Executor, FallbackModel, Flow, FlowRunner, GuardrailHook, Hook, LLMRequest, LLMResponse, Ledger, LedgerEntry, LoggingHook, MemoryApprovalStore, MemoryCache, Message, MissingDependency, Prices, RateLimitHook, RedisApprovalStore, Registry, RunLimits, SQLiteApprovalStore, Step, ToolCall, ToolContext, ToolDef, ToolError, ToolResult, ToolSpec, Usage, __version__, acall, attach, build_spec, call, configure, default_registry, run_sync, run_tool_loop, set_default_executor, spec_from_schema, tool, wrap]
