"""Run and call identity from a LangGraph `RunnableConfig`."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.integrations.langgraph._integration import INTEGRATION
from agent_adapter.runtime.current import current_context

NS_KEY = "langgraph_checkpoint_ns"  # RunnableConfig metadata: the running task's namespace


def task_namespace(config: Any) -> str:
    """The checkpoint namespace of the task a `RunnableConfig` belongs to ("" outside one)."""
    if not isinstance(config, dict):
        return ""
    metadata = config.get("metadata") or {}
    configurable = config.get("configurable") or {}
    return str(metadata.get(NS_KEY) or configurable.get("checkpoint_ns") or "")


def run_ids(config: Any) -> tuple[str, str]:
    """`(run_id, thread_id)`: `configurable["run_id"]`, else `["thread_id"]` (also the thread)."""
    configurable = (config or {}).get("configurable") or {} if isinstance(config, Mapping) else {}
    thread_id = str(configurable.get("thread_id") or "")
    return str(configurable.get("run_id") or thread_id), thread_id


def call_context(config: Any, native: Mapping[str, Any]) -> ToolContext:
    """The context of one native call: the current adapter run, else the config's run."""
    bound = INTEGRATION.bind_context(native)
    outer = current_context()
    if outer is not None:
        return outer.child(call_id=bound.call_id, extras={**outer.extras, **bound.extras})
    run_id, thread_id = run_ids(config)
    return dataclasses.replace(bound, run_id=run_id, thread_id=thread_id)
