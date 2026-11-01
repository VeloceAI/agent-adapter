# Hooks

Hooks around tool, model and node calls, and the built-in hooks. Concepts: [Hooks](../concepts/hooks.md).

## `agent_adapter.hooks`

::: agent_adapter.hooks
    options:
      members: [DEFAULT_REDACT, LLMInvocation, NodeInvocation, Rate, ShortCircuit, ToolInvocation, hook_name, redact, run_after, run_before, run_on_error]

Also exported here, documented under the canonical path: [`BudgetHook`](top-level.md#agent_adapter.BudgetHook), [`GuardrailHook`](top-level.md#agent_adapter.GuardrailHook), [`Hook`](top-level.md#agent_adapter.Hook), [`LoggingHook`](top-level.md#agent_adapter.LoggingHook), [`RateLimitHook`](top-level.md#agent_adapter.RateLimitHook).

## `agent_adapter.hooks.builtin`

::: agent_adapter.hooks.builtin
    options:
      members: [BudgetScope, GuardrailCheck, OnExceed, REDACTED, Stage]

Also exported here, documented under the canonical path: [`BudgetHook`](top-level.md#agent_adapter.BudgetHook), [`DEFAULT_REDACT`](#agent_adapter.hooks.DEFAULT_REDACT), [`GuardrailHook`](top-level.md#agent_adapter.GuardrailHook), [`LoggingHook`](top-level.md#agent_adapter.LoggingHook), [`Rate`](#agent_adapter.hooks.Rate), [`RateLimitHook`](top-level.md#agent_adapter.RateLimitHook), [`redact`](#agent_adapter.hooks.redact).
