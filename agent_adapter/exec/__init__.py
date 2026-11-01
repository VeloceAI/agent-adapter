"""Executor: runs tools with guards, hooks, approval, cache, retries, batches, plans and loops."""

from agent_adapter.exec._sync_twins import SyncTwins
from agent_adapter.exec.control import is_control_flow, register_control_flow
from agent_adapter.exec.defaults import (
    acall,
    call,
    configure,
    default_executor,
    reset_default_executor,
    runner_for,
    set_default_executor,
)
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.guards import RunLimits
from agent_adapter.exec.parallel import OnError
from agent_adapter.exec.plan import Step
from agent_adapter.exec.retry import Backoff
from agent_adapter.exec.tool_loop import ToolLoopResult, run_tool_loop

__all__ = [
    "Backoff",
    "Executor",
    "OnError",
    "RunLimits",
    "Step",
    "SyncTwins",
    "ToolLoopResult",
    "acall",
    "call",
    "configure",
    "default_executor",
    "is_control_flow",
    "register_control_flow",
    "reset_default_executor",
    "run_tool_loop",
    "runner_for",
    "set_default_executor",
]
