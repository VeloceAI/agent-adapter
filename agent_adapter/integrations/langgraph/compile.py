"""Compile a `Flow` to a LangGraph `StateGraph`; `run_compiled` runs it to a `FlowResult`."""

from __future__ import annotations

import functools
import logging
from typing import Annotated, Any

from agent_adapter.core.errors import FlowCompileError, ToolUsageError
from agent_adapter.exec.defaults import default_executor
from agent_adapter.exec.executor import Executor
from agent_adapter.flow.graph import LOOP_FIELD, VISITS_FIELD, FlowGraph
from agent_adapter.flow.spec import Flow
from agent_adapter.flow.visit import NodeEnv, allowed_classes, graph_digest, runner_key
from agent_adapter.integrations.langgraph._channels import COUNTS, merge_channel
from agent_adapter.integrations.langgraph._integration import INTEGRATION, TARGET
from agent_adapter.integrations.langgraph._node import CompiledFlow, FlowNode
from agent_adapter.integrations.langgraph._run import run_compiled
from agent_adapter.integrations.langgraph.approval import LangGraphApproval

logger = logging.getLogger(__name__)

_RESERVED_NODES = ("__start__", "__end__")


def compile_flow(
    flow: Flow[Any], *, executor: Executor | None = None, checkpoint_key: bytes | None = None
) -> Any:
    """`flow` as a LangGraph `StateGraph`; compile it with a checkpointer to pause and resume.

    Each flow node is one LangGraph node running the local runner's node visit. Merge rules
    are channels seeing every write of a superstep; `pause` gates go to `interrupt_before`
    (added by the returned graph's `compile()`); tool approvals inside a node pause the task.
    `executor=None` uses the default executor, read now, with `LangGraphApproval()`.
    `checkpoint_key` seals a paused node's progress (default: the `LangGraphApproval` port's
    key, else a per-process key): give every process that resumes the same key.
    `node.finished` names the changed state fields; their values (`trace_state`) need FlowRunner.

    Raises:
        FlowDefinitionError: the flow is invalid.
        FlowCompileError: a node name LangGraph reserves (`__start__`, `__end__`), or a
            `SubFlow` node (run those flows with `FlowRunner`).
        ToolUsageError: `flow` is not a Flow, `executor` is not an Executor, or a bad key.
        MissingDependency: langgraph is not installed.
    """
    INTEGRATION.require()
    if not isinstance(flow, Flow):
        raise ToolUsageError(f"compile_flow needs a Flow, got {type(flow).__name__}")
    graph = flow.validate()
    runner = executor if executor is not None else _default_executor()
    if not isinstance(runner, Executor):
        raise ToolUsageError(
            f"compile_flow needs an Executor (Map nodes use run_many), got {type(runner).__name__}"
        )
    problems = [
        f"node name {name!r} is reserved by LangGraph"
        for name in _RESERVED_NODES
        if name in graph.nodes
    ]
    problems.extend(
        f"node {name!r} is a SubFlow, which the LangGraph compiler does not support yet; "
        "run this flow with FlowRunner"
        for name, spec in graph.nodes.items()
        if spec.kind == "subflow"
    )
    if problems:
        raise FlowCompileError(TARGET, problems, flow=graph.name)
    port = runner.approval
    mac_key = port.key if isinstance(port, LangGraphApproval) else runner_key(checkpoint_key)
    seal_key = runner_key(checkpoint_key) if checkpoint_key is not None else mac_key
    return _build(graph, runner, seal_key, mac_key)


def _default_executor() -> Executor:
    return default_executor().replace(approval=LangGraphApproval())


def _build(graph: FlowGraph, executor: Executor, seal_key: bytes, mac_key: bytes) -> Any:
    from langgraph.graph import END, START

    env = NodeEnv(graph.name, executor, executor.hooks, allowed=allowed_classes(graph.state_model))
    gated = tuple(name for name, spec in graph.nodes.items() if spec.pause is not False)
    info = CompiledFlow(graph, executor, env, gated, graph_digest(graph), seal_key, mac_key)
    builder = _graph_class()(_schema(graph))
    builder.aa_flow = info
    for name, spec in graph.nodes.items():
        route = graph.routes.get(name)
        builder.add_node(
            name,
            FlowNode(info, spec),
            defer=spec.kind == "join",
            destinations=tuple(dict.fromkeys(route.targets.values())) if route else None,
        )
    builder.add_edge(START, graph.start)
    for source, targets in graph.edges.items():
        for target in targets:
            builder.add_edge(source, target)
    for end in graph.ends:
        builder.add_edge(end, END)
    logger.debug("flow %s compiled to LangGraph (%d nodes)", graph.name, len(graph.nodes))
    return builder


def _schema(graph: FlowGraph) -> Any:
    """A TypedDict state schema: one merge-rule channel per field, plus the reserved counters."""
    from typing import TypedDict

    fields: dict[str, Any] = {
        name: Annotated[Any, merge_channel(rule)] for name, rule in graph.rules.items()
    }
    fields[VISITS_FIELD] = Annotated[dict[str, int], merge_channel(COUNTS)]
    fields[LOOP_FIELD] = Annotated[dict[str, int], merge_channel(COUNTS)]
    typed: Any = TypedDict  # a schema built at run time
    return typed(f"{graph.name}_state", fields, total=False)


@functools.cache
def _graph_class() -> type[Any]:
    from langgraph.graph import StateGraph

    base: Any = StateGraph  # Any: the type check must not depend on the extra

    class FlowStateGraph(base):  # type: ignore[misc]
        """A compiled flow's `StateGraph`; `compile()` adds the gated nodes to interrupt_before."""

        aa_flow: CompiledFlow

        def compile(
            self, checkpointer: Any = None, *, interrupt_before: Any = None, **kwargs: Any
        ) -> Any:
            before = interrupt_before
            if before != "*":
                before = list(dict.fromkeys([*(before or ()), *self.aa_flow.gated])) or None
            return super().compile(checkpointer, interrupt_before=before, **kwargs)

    return FlowStateGraph


__all__ = ["compile_flow", "run_compiled"]
