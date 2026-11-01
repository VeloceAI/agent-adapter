# Flows

Flow graphs, nodes and the runner. Concepts: [Flows](../concepts/flows.md), [Flow pause and resume](../concepts/flow-resume.md).

`agent_adapter.flow.visit` is a seam for flow compilers; most code never needs it.

## `agent_adapter.flow`

::: agent_adapter.flow
    options:
      members: [Call, FlowCheckpoint, FlowGraph, FlowResult, Join, Map, Merge, MergeRule, NodeFn, NodeSpec, Route, RouteSpec, apply_updates, keyed, merge_rules, node_model, node_prompt]

Also exported here, documented under the canonical path: [`Flow`](top-level.md#agent_adapter.Flow), [`FlowRunner`](top-level.md#agent_adapter.FlowRunner).

## `agent_adapter.flow.runner`

::: agent_adapter.flow.runner
    options:
      members: [FlowStatus]

Also exported here, documented under the canonical path: [`FlowCheckpoint`](#agent_adapter.flow.FlowCheckpoint), [`FlowResult`](#agent_adapter.flow.FlowResult), [`FlowRunner`](top-level.md#agent_adapter.FlowRunner).

## `agent_adapter.flow.visit`

::: agent_adapter.flow.visit
    options:
      members: [ItemOutcome, NOT_SEALABLE, NO_PORT, NodeEnv, NodeOutcome, SPENT, SUPERSEDED, TAMPERED, allowed_classes, apply_edits, ask_node_gate, call_route, describe_error, frozen, graph_digest, mac, merge_outcomes, next_frontier, node_call_id, node_gate_request, not_sealable, pick_targets, resume_entry, route_state, route_targets, run_node, runner_key, spend_once, superseded, withdraw]
