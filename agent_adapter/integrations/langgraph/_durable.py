"""A compiled node's progress, sealed into its interrupt value and read back on a re-run.

When a visit pauses, its interrupt value carries `aa_state`: the approvals it surfaced (and
how each was answered), the replay journal and the finished `Map` items, signed with the
flow's key and bound to the run, the task and the graph. A re-run (in any process with the
same key) reads the task's interrupt write back from the checkpointer and replays finished
calls as the local runner does; a state that does not verify fails closed.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.core.errors import FlowError, ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec.integration import JournalData, NotSealable, check_plain, decode, encode
from agent_adapter.flow.visit import ItemOutcome, mac

logger = logging.getLogger(__name__)

FORMAT = "agent_adapter.langgraph.task/1"
STATE_KEY = "aa_state"  # in the interrupt value
CHECKPOINTER_KEY = "__pregel_checkpointer"  # LangGraph 1.x: the run's saver, in `configurable`
INTERRUPT_CHANNEL = "__interrupt__"  # LangGraph 1.x: the channel of saved interrupts
FAIL_CLOSED = (
    "the paused node's sealed state does not verify: resume in the pausing process or give "
    "the call a stable call_id (or compile every process with the same checkpoint_key)"
)


@dataclass
class Surfaced:
    """An approval a task surfaced, in order, and how it was answered."""

    request: ApprovalRequest
    exact: bool  # the call id is the flow's (it repeats on a re-run)
    via: str = ""  # "" waiting, "interrupt" a resume value, "port" decided at once, "store"
    decision: ApprovalDecision | None = None  # kept for "port" and "store"


@dataclass
class TaskState:
    """An interrupted task's progress: its approvals, finished calls and Map items."""

    surfaced: list[Surfaced] = field(default_factory=list)
    journal: JournalData = field(default_factory=dict)
    items: dict[int, ItemOutcome] = field(default_factory=dict)


@dataclass(frozen=True)
class Seal:
    """What a sealed state is bound to."""

    key: bytes
    graph: str  # graph digest
    run_id: str
    task: str  # checkpoint namespace
    allowed: Mapping[str, type]


def seal(seal_with: Seal, state: TaskState) -> dict[str, Any]:
    """The sealed form of `state` for the interrupt value.

    Raises:
        NotSealable: a journal entry or Map item has no canonical form.
    """
    check_plain(state.journal, "journal")
    data = {
        "format": FORMAT,
        "graph": seal_with.graph,
        "run": seal_with.run_id,
        "task": seal_with.task,
        "surfaced": [_surfaced(s) for s in state.surfaced],
        "journal": state.journal,
        "items": encode(state.items, None, "items"),
    }
    return {"payload": data, "mac": mac(seal_with.key, data)}


def _surfaced(s: Surfaced) -> dict[str, Any]:
    decision = s.decision.model_dump(mode="json") if s.decision is not None else None
    return {
        "request": s.request.model_dump(mode="json"),
        "exact": s.exact,
        "via": s.via,
        "decision": decision,
    }


def open_sealed(seal_with: Seal, sealed: Any, flow: str) -> TaskState:
    """The state in `sealed`, after its MAC and bindings are checked.

    Raises:
        FlowError: not this task's sealed state, or a wrong MAC (fails closed).
    """
    bound: tuple[Any, ...] = ()
    try:
        data, given = sealed["payload"], sealed["mac"]
        bound = (data["format"], data["graph"], data["run"], data["task"])
        ok = isinstance(given, str) and hmac.compare_digest(given, mac(seal_with.key, data))
    except (KeyError, TypeError, ValueError):
        ok = False
    expected = (FORMAT, seal_with.graph, seal_with.run_id, seal_with.task)
    if not ok or bound != expected:
        logger.warning("flow %s: sealed task state refused (task %s)", flow, seal_with.task)
        raise FlowError(f"flow {flow!r}: {FAIL_CLOSED}", flow=flow)
    try:
        return TaskState(
            surfaced=[_open_surfaced(s) for s in data["surfaced"]],
            journal=dict(data["journal"]),
            items=dict(decode(data["items"], None, seal_with.allowed)),
        )
    except (KeyError, TypeError, ValueError, NotSealable) as exc:
        raise FlowError(
            f"flow {flow!r}: sealed task state cannot be loaded: {exc}", flow=flow
        ) from exc


def _open_surfaced(data: Mapping[str, Any]) -> Surfaced:
    decision = data["decision"]
    return Surfaced(
        request=ApprovalRequest.model_validate(data["request"]),
        exact=bool(data["exact"]),
        via=str(data["via"]),
        decision=ApprovalDecision.model_validate(decision) if decision is not None else None,
    )


async def load(config: Any, seal_with: Seal, flow: str) -> TaskState | None:
    """This task's sealed state from its last interrupt write; None when it has none.

    Raises:
        FlowError: a sealed state is there but does not verify.
        ToolUsageError: an unsupported LangGraph version (no saver key in the config).
    """
    configurable = (config or {}).get("configurable") or {}
    task = seal_with.task
    if ":" not in task:
        return None  # not a LangGraph task (called directly)
    if CHECKPOINTER_KEY not in configurable:  # present (None) when the graph has no saver
        raise ToolUsageError("unsupported LangGraph version: no checkpointer in config")
    saver = configurable[CHECKPOINTER_KEY]
    if saver is None:
        return None
    parent = task.rpartition("|")[0]
    task_id = task.rpartition(":")[2]
    where: dict[str, Any] = {"thread_id": configurable.get("thread_id"), "checkpoint_ns": parent}
    checkpoint_id = (configurable.get("checkpoint_map") or {}).get(parent)
    if checkpoint_id:
        where["checkpoint_id"] = checkpoint_id
    found = await saver.aget_tuple({"configurable": where})
    for writer, channel, value in (found.pending_writes or ()) if found is not None else ():
        if writer != task_id or channel != INTERRUPT_CHANNEL:
            continue
        for interrupt in value if isinstance(value, list | tuple) else (value,):
            payload = getattr(interrupt, "value", None)
            if isinstance(payload, Mapping) and STATE_KEY in payload:
                return open_sealed(seal_with, payload[STATE_KEY], flow)
    return None
