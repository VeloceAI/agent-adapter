"""Semi-public flow seam for flow compilers (LangGraph, ADK); not in `flow.__all__`.

The compilers run node visits, gates, routes, resumes and seals exactly like the local
runner. They import those internals from here, never from the private modules, so a
refactor there keeps this surface stable.
"""

from agent_adapter.flow._digest import graph_digest
from agent_adapter.flow._gates import NO_PORT, _ask, _request
from agent_adapter.flow._invoke import describe_error
from agent_adapter.flow._map import ItemOutcome
from agent_adapter.flow._nodes import NodeEnv, NodeOutcome, node_call_id, run_node
from agent_adapter.flow._pause import _superseded
from agent_adapter.flow._result import frozen
from agent_adapter.flow._resume import SPENT, resume_entry, spend_once
from agent_adapter.flow._seal import (
    NOT_SEALABLE,
    TAMPERED,
    allowed_classes,
    mac,
    not_sealable,
    runner_key,
)
from agent_adapter.flow._step import (
    apply_edits,
    call_route,
    merge_outcomes,
    next_frontier,
    pick_targets,
    route_state,
    route_targets,
)
from agent_adapter.flow._withdraw import SUPERSEDED, withdraw

node_gate_request = _request
"""The `ApprovalRequest` a gated node visit asks for (call id = the visit's id)."""

ask_node_gate = _ask
"""Ask the port for a gate: its decision, the pending request, or a denial without a port."""

superseded = _superseded
"""Pending records a resume never reuses: not the checkpoint's, a call id the flow did not set."""

__all__ = [
    "NOT_SEALABLE",
    "NO_PORT",
    "SPENT",
    "SUPERSEDED",
    "TAMPERED",
    "ItemOutcome",
    "NodeEnv",
    "NodeOutcome",
    "allowed_classes",
    "apply_edits",
    "ask_node_gate",
    "call_route",
    "describe_error",
    "frozen",
    "graph_digest",
    "mac",
    "merge_outcomes",
    "next_frontier",
    "node_call_id",
    "node_gate_request",
    "not_sealable",
    "pick_targets",
    "resume_entry",
    "route_state",
    "route_targets",
    "run_node",
    "runner_key",
    "spend_once",
    "superseded",
    "withdraw",
]
