# Hooks

Hooks around tool, model and node calls, the built-in hooks and content guards. Concepts: [Hooks](../concepts/hooks.md), [Guardrails](../concepts/guardrails.md), [Team budgets](../how-to/team-budgets.md).

## `agent_adapter.hooks`

::: agent_adapter.hooks
    options:
      members: [CallableGuard, ContentGuard, DEFAULT_REDACT, GuardVerdict, LLMInvocation, NemoGuard, NodeInvocation, PIIGuard, PresidioGuard, Rate, ShortCircuit, ToolInvocation, hook_name, redact, run_after, run_before, run_on_error]

Also exported here, documented under the canonical path: [`BudgetHook`](top-level.md#agent_adapter.BudgetHook), [`GuardrailHook`](top-level.md#agent_adapter.GuardrailHook), [`Hook`](top-level.md#agent_adapter.Hook), [`LoggingHook`](top-level.md#agent_adapter.LoggingHook), [`RateLimitHook`](top-level.md#agent_adapter.RateLimitHook).

## `agent_adapter.hooks.builtin`

::: agent_adapter.hooks.builtin
    options:
      members: [BudgetScope, GuardrailCheck, OnExceed, REDACTED, Stage]

Also exported here, documented under the canonical path: [`BudgetHook`](top-level.md#agent_adapter.BudgetHook), [`DEFAULT_REDACT`](#agent_adapter.hooks.DEFAULT_REDACT), [`GuardrailHook`](top-level.md#agent_adapter.GuardrailHook), [`LoggingHook`](top-level.md#agent_adapter.LoggingHook), [`Rate`](#agent_adapter.hooks.Rate), [`RateLimitHook`](top-level.md#agent_adapter.RateLimitHook), [`redact`](#agent_adapter.hooks.redact).

## `agent_adapter.hooks.builtin.budgets`

::: agent_adapter.hooks.builtin.budgets
    options:
      members: [BudgetStatus, BudgetStore, BudgetUsage, MemoryBudgetStore, PERIODS, Per, RedisBudgetStore, Reserved, SQLAlchemyBudgetStore, SQLiteBudgetStore, Window, principal_of, window_at]

## `agent_adapter.hooks.builtin.guards`

::: agent_adapter.hooks.builtin.guards
    options:
      members: [Action, GuardFn, NO_REPLACEMENT, PIIMatch, PII_KINDS, TextGuard, as_verdict, find_pii, iban_ok, luhn_ok, redact_pii]

Also exported here, documented under the canonical path: [`CallableGuard`](#agent_adapter.hooks.CallableGuard), [`ContentGuard`](#agent_adapter.hooks.ContentGuard), [`GuardVerdict`](#agent_adapter.hooks.GuardVerdict), [`NemoGuard`](#agent_adapter.hooks.NemoGuard), [`PIIGuard`](#agent_adapter.hooks.PIIGuard), [`PresidioGuard`](#agent_adapter.hooks.PresidioGuard).
