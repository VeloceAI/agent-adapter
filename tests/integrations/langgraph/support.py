"""Helpers for the LangGraph tests: tiny graphs, configs and resume values."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations.langgraph import LangGraphApproval

_threads = itertools.count()


def config(thread: str | None = None, **extra: Any) -> dict[str, Any]:
    """A config with a fresh (or the given) thread id."""
    return {"configurable": {"thread_id": thread or f"t{next(_threads)}"}, **extra}


def one_node(node: Callable[..., Any], state: type = MessagesState) -> Any:
    """START -> node -> END, compiled with an in-memory checkpointer."""
    graph = StateGraph(state)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    return graph.compile(checkpointer=InMemorySaver())


def bound(
    interrupt: Any, decision: ApprovalDecision | dict[str, Any], key: bytes | None = None
) -> dict[str, Any]:
    """The signed resume value answering `interrupt` (an `Interrupt` or its value)."""
    if not isinstance(decision, ApprovalDecision):
        decision = ApprovalDecision.model_validate(decision)
    return LangGraphApproval(key).resume_value(interrupt, decision)


def resume(
    interrupt: Any, decision: ApprovalDecision | dict[str, Any], key: bytes | None = None
) -> Command:
    """`Command(resume={interrupt.id: bound(...)})`."""
    return Command(resume={interrupt.id: bound(interrupt, decision, key)})
