# WP15 — LangGraph: TDD plan

Contract: §14.4 `langgraph` row, §13.5, §7, §8, §14.1–§14.3. Research: R2 §1, §4.

## Design decisions

| Topic | Decision |
|---|---|
| Package | `integrations/langgraph/`: `__init__` (exports, `INTEGRATION`), `approval` (`LangGraphApproval`, resume binding), `tools` (`tool_node`, `tool_task`, `fan_out`), `compile` (`compile_flow`, `run_compiled`); private `_channels` (merge-rule channels, `reducer`), `_node` (compiled node), `_memo` (in-process task memo), `_context` (run ids from `RunnableConfig`) |
| Resume values | every interrupt value carries `binding` = SHA-256 of run id, tool, kind, `request_digest` and (exact) call id; a resume value must be `{"binding", "decision"}` with the same binding (`response_schema` + validator: a mismatch is LangGraph's invalid resume, nothing is saved, answer again) |
| Approval port | `LangGraphApproval.request` calls `interrupt()`; inside a compiled node it defers (raises `ApprovalPending`) so the node's single interrupt loop binds the answer (exact call id only for flow-assigned ids) |
| Compiled node | one LangGraph node per flow node running `flow._nodes.run_node` (hooks, timeout, retries, `Call`, `Map` via `run_many`, `Join`), so behaviour equals the local runner; routes computed in the node (`Command(goto=…)`) on input state + own writes |
| Merge rules | every state field is a custom `BaseChannel` applying the `MergeRule` to all writes of a superstep; writes carry `(node, phase, seq)` tags so order is node-name order, then item order, independent of LangGraph's task order |
| Reserved keys | `aa_visits` (completed visits), `aa_loop_counts` (route loops) as dict channels |
| Gates | `interrupt_before` gated nodes (a `StateGraph` subclass adds them in `compile()`); `run_compiled` asks the port at the static pause; a denial ends the run without running the superstep |
| Re-runs | in-process memo per task (run id + checkpoint ns): surfaced requests, journal, Map items; a re-run replays resume values positionally, replays finished calls, never asks the port twice. Cross-process: deterministic call ids keep ledger and guards single; function-node own calls re-run (documented) |
| `max_steps` | `recursion_limit` (`run_compiled` defaults it to 100); `GraphRecursionError` → `FlowLoopExceeded` |

## Tasks (test first, each)

1. `approval.py`: binding, payload, resume validation (good, forged binding, other run, other args, garbage), outside-graph use → `ToolUsageError`. Tests: `tests/integrations/langgraph/test_approval.py`.
2. `_channels.py` + `reducer`: each merge rule through a real graph, replace conflict, equal writes, keyed collision by node name, tagged order. Tests: `test_channels.py`.
3. `tools.py`: `tool_node` (messages, errors, unknown tool, confirm calls sequential and one interrupt at a time, memo), `tool_task`, `fan_out`. Tests: `test_tools.py`, `test_tool_node_approval.py`.
4. Executor pass-through: `GraphInterrupt` unwrapped through `run`, `run_spec`, `run_call`, `run_many`, `as_completed`, `run_plan`, `parallel`; resume without double ledger entries or a second `once_per_run` count. Tests: `test_executor_passthrough.py`.
5. `compile.py` + `_node.py`: compile errors, gates, edits, denial, tool pauses, forged resume, re-runs, memo, recursion limit. Tests: `test_compile.py`, `test_compile_resume.py`.
6. Parity: every case of `tests/flow/parity.py` through `compile_flow` / `run_compiled`, equal to the local runner; a cross-process re-run shows the documented difference. Tests: `test_parity.py`.
7. Contract kit C1–C8: `tests/integrations/test_langgraph_contract.py` (FakeRunner and Executor).
8. Docs `docs/integrations/langgraph.md`, example `examples/langgraph_flow.py` + test; `SHARED-EDITS.md`.

## Exit gate

Kit C1–C8; `GraphInterrupt` unwrapped; resume without double ledger / `once_per_run`;
`tool_node` interrupts one at a time; reducers per merge rule; every parity case equal to the
local runner. Gates: pytest (0 warnings), ruff, ruff format, mypy strict, mkdocs strict,
`nox -s contract-langgraph`, coverage of the new module ≥ 90 %.
