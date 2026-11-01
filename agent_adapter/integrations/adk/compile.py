"""Compile a Flow to an ADK `Workflow`: one driver node runs the supersteps."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.errors import FlowCompileError, ToolUsageError
from agent_adapter.core.ports import LLMPort
from agent_adapter.exec.defaults import runner_for
from agent_adapter.exec.executor import Executor
from agent_adapter.flow.spec import Flow
from agent_adapter.flow.visit import NodeEnv, allowed_classes
from agent_adapter.integrations.adk._steps import Plan
from agent_adapter.integrations.adk._support import DRIVER, TARGET, require_adk
from agent_adapter.integrations.adk._workflow import node_classes


def compile_flow(
    flow: Flow[Any],
    *,
    executor: Executor | None = None,
    name: str | None = None,
    max_steps: int = 100,
    model: LLMPort | None = None,
) -> Any:
    """`flow` as an ADK `Workflow` (`START -> aa_driver`); run it with `run_workflow`.

    `node.finished` names the changed state fields; their values (`trace_state`) need FlowRunner.

    Args:
        executor: Runs every node and tool call; default `default_executor()`, read now.
        name: The Workflow's name (an identifier); default the flow's name.
        max_steps: Supersteps before `FlowLoopExceeded` (ADK has no recursion limit).
        model: Exposed to nodes through `node_model(ctx)`, as on `FlowRunner`.

    Raises:
        FlowDefinitionError: the flow is invalid.
        FlowCompileError: `executor` is not an `Executor`, or `name` is not an identifier.
        ToolUsageError: `max_steps` is not an int >= 1.
    """
    require_adk()
    graph = flow.validate()
    runner = runner_for(executor)
    workflow_name = graph.name if name is None else name
    problems = _problems(runner, workflow_name)
    if problems or not isinstance(runner, Executor):
        raise FlowCompileError(TARGET, problems, flow=graph.name)
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ToolUsageError(f"max_steps must be an int >= 1, got {max_steps!r}")
    env = NodeEnv(
        graph.name,
        runner,
        runner.hooks,
        model,
        None,
        None,
        allowed_classes(graph.state_model),
    )
    driver_class, _ = node_classes()
    driver = driver_class(name=DRIVER)
    driver._plan = Plan(graph, env, max_steps)
    from google.adk.workflow import START, Workflow

    return Workflow(name=workflow_name, edges=[(START, driver)])


def _problems(runner: Any, workflow_name: Any) -> list[str]:
    problems = []
    if not isinstance(runner, Executor):
        problems.append(f"flows run on an agent_adapter Executor, got {type(runner).__name__}")
    if not isinstance(workflow_name, str) or not workflow_name.isidentifier():
        problems.append(f"the Workflow name must be an identifier, got {workflow_name!r}")
    return problems


def _flow_plan(workflow: Any) -> Plan:
    """The plan of a Workflow `compile_flow` made.

    Raises:
        ToolUsageError: `workflow` was not made by `compile_flow`.
    """
    driver_class, _ = node_classes()
    for node in getattr(getattr(workflow, "graph", None), "nodes", ()):
        if isinstance(node, driver_class):
            plan: Plan = node._plan
            return plan
    raise ToolUsageError(f"{workflow!r} is not a Workflow made by compile_flow")


__all__ = ["compile_flow"]
