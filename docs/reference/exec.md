# Executor

Running tool calls: retries, plans and the tool loop. Concepts: [Executor](../concepts/executor.md), [Executor policies](../concepts/executor-policies.md).

`agent_adapter.exec.integration` is a seam for flow compilers and framework bridges; most code never needs it. Checkpoint stores for crash-safe runs: [Durable runs](../concepts/durable-runs.md).

## `agent_adapter.exec`

::: agent_adapter.exec
    options:
      members: [AgentDepthExceeded, Backoff, HandoffModel, LoopAgent, OnError, SyncTwins, ToolLoopResult, agent_depth, agent_tool, default_executor, handoff, is_control_flow, recover_tool_loop, register_control_flow, reset_default_executor, runner_for]

Also exported here, documented under the canonical path: [`Executor`](top-level.md#agent_adapter.Executor), [`RunLimits`](top-level.md#agent_adapter.RunLimits), [`Step`](top-level.md#agent_adapter.Step), [`acall`](top-level.md#agent_adapter.acall), [`call`](top-level.md#agent_adapter.call), [`configure`](top-level.md#agent_adapter.configure), [`run_tool_loop`](top-level.md#agent_adapter.run_tool_loop), [`set_default_executor`](top-level.md#agent_adapter.set_default_executor).

## `agent_adapter.exec.checkpoints`

::: agent_adapter.exec.checkpoints
    options:
      members: [CallRecord, CallState, Checkpoint, CheckpointKind, CheckpointStatus, CheckpointStore, MemoryCheckpointStore, RedisCheckpointStore, SQLAlchemyCheckpointStore, SQLiteCheckpointStore, StoredCheckpoint, idempotency_key]

## `agent_adapter.exec.control`

`agent_adapter.exec.control` re-exports, documented under the canonical path: [`is_control_flow`](#agent_adapter.exec.is_control_flow), [`register_control_flow`](#agent_adapter.exec.register_control_flow).

## `agent_adapter.exec.integration`

::: agent_adapter.exec.integration
    options:
      members: [AnswerEntry, Journal, JournalData, NotSealable, Pending, RAN_UNKEYED, approval_store, check_plain, decode, encode, gather_unwrapped, make_answer, make_journal, same_decision, settle_own, unsealable_calls, visit_scope]
