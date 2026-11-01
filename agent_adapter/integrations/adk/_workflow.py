"""The ADK nodes of a compiled flow and the driver's entry: initial state in, result out."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import FlowError
from agent_adapter.core.types import to_jsonable
from agent_adapter.flow.graph import FlowGraph
from agent_adapter.flow.state import validated_state
from agent_adapter.integrations.adk._driver import Cursor, Pause, Turn
from agent_adapter.integrations.adk._steps import Plan
from agent_adapter.integrations.adk._support import INTERRUPT_PREFIX
from agent_adapter.integrations.adk._tokens import INTERRUPTS
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)

MAX_CANCELLED = 10_000


class _Cancelled:
    """Run ids whose driver was cancelled, until `run_workflow` reads them (bounded)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, None] = OrderedDict()

    def add(self, run_id: str) -> None:
        with self._lock:
            self._runs[run_id] = None
            while len(self._runs) > MAX_CANCELLED:
                self._runs.popitem(last=False)

    def take(self, run_id: str) -> bool:
        with self._lock:
            return self._runs.pop(run_id, 0) is None


CANCELLED = _Cancelled()


@functools.cache
def node_classes() -> tuple[Any, Any]:
    """The driver and step node classes, built on first use (`google.adk` imported lazily)."""
    from google.adk import workflow
    from google.adk.events import Event
    from pydantic import PrivateAttr

    base: Any = workflow.BaseNode

    class FlowDriver(base):  # type: ignore[misc]
        """Runs a flow's supersteps; re-run on every resume, it replays finished children."""

        rerun_on_resume: bool = True
        _plan: Any = PrivateAttr(default=None)

        async def _run_impl(self, *, ctx: Any, node_input: Any) -> AsyncIterator[Any]:
            async for item in drive(self._plan, ctx, node_input):
                yield item

    class FlowStep(base):  # type: ignore[misc]
        """One step of the driver (a gate, a node visit, an answer); its output is JSON."""

        _work: Any = PrivateAttr(default=None)

        async def _run_impl(self, *, ctx: Any, node_input: Any) -> AsyncIterator[Any]:
            yield Event(output=await self._work(ctx))

    return FlowDriver, FlowStep


async def drive(plan: Plan, adk: Any, node_input: Any) -> AsyncIterator[Any]:
    """The driver's body: run until done, denied or paused, then yield the result event."""
    from google.adk.events import Event, RequestInput
    from google.adk.tools.tool_confirmation import ToolConfirmation

    turn = Turn(plan, adk, node_classes()[1])
    ctx = turn.ctx
    if not any(key.startswith(INTERRUPT_PREFIX) for key in turn.inputs):
        emit(ctx.sink, make_event(EventType.RUN_STARTED, ctx))
    emit(ctx.sink, make_event(EventType.FLOW_STARTED, ctx, flow=plan.flow))
    try:
        state = _initial(plan.graph, node_input)
    except FlowError:
        _finished(turn, "failed", 0, ok=False)
        raise
    cursor = Cursor(state=state, frontier=(plan.graph.start,))
    try:
        end = await turn.run(cursor)
    except asyncio.CancelledError:
        # ADK's Runner ends the turn quietly on a cancelled root; run_workflow reports it
        CANCELLED.add(ctx.run_id)
        _finished(turn, "failed", cursor.steps, ok=False)
        raise
    except Exception as exc:
        if is_control_flow(exc):
            raise
        _finished(turn, "failed", cursor.steps, ok=False)
        raise
    if isinstance(end, Pause):
        _finished(turn, "paused", cursor.steps)
        pending = end.pending
        INTERRUPTS.remember(end.interrupt_id, pending)
        payload = {
            "tool_name": pending.tool,
            "args": pending.arguments,
            "aa": {
                "request": pending.model_dump(mode="json"),
                "state": to_jsonable(cursor.state),
                "steps": cursor.steps,
            },
        }
        yield RequestInput(
            interrupt_id=end.interrupt_id,
            message=pending.hint or f"Please approve or reject {pending.tool}.",
            payload=payload,
            response_schema=ToolConfirmation,
        )
        return
    _finished(turn, end.status, cursor.steps, ok=end.status == "done")
    yield Event(
        output={
            "status": end.status,
            "state": to_jsonable(end.state),
            "steps": cursor.steps,
            "visits": dict(end.visits),
            "denied_at": end.denied_at,
        }
    )


def _finished(turn: Turn, status: str, steps: int, *, ok: bool | None = None) -> None:
    ctx = turn.ctx
    flow = turn.plan.flow
    emit(ctx.sink, make_event(EventType.FLOW_FINISHED, ctx, flow=flow, status=status, steps=steps))
    if ok is not None:
        emit(ctx.sink, make_event(EventType.RUN_FINISHED, ctx, ok=ok))
        turn.plan.env.executor.end_run(ctx.run_id)
    logger.debug("flow %s %s after %d steps", flow, status, steps)


def _initial(graph: FlowGraph, node_input: Any) -> BaseModel:
    """The initial state from the user's message (a JSON object) or a mapping or model.

    Raises:
        FlowError: the input is not a JSON object of the state model's fields.
    """
    model = graph.state_model
    if isinstance(node_input, model):
        return node_input
    data = node_input
    parts = getattr(node_input, "parts", None)
    if parts is not None:
        text = "".join(part.text for part in parts if getattr(part, "text", None))
        try:
            data = json.loads(text) if text.strip() else {}
        except ValueError as exc:
            raise FlowError(
                f"flow {graph.name!r}: the message must be the initial state as a JSON object",
                flow=graph.name,
            ) from exc
    if not isinstance(data, Mapping):
        raise FlowError(
            f"flow {graph.name!r}: the initial state must be a JSON object, "
            f"got {type(data).__name__}",
            flow=graph.name,
        )
    return validated_state(model, data, what=f"initial state of flow {graph.name!r}")
