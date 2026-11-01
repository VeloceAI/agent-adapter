"""Run a compiled flow on ADK's `Runner` and read the outcome back as a `FlowResult`."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import weakref
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.errors import FlowError, ToolUsageError
from agent_adapter.core.types import ApprovalRequest, to_jsonable
from agent_adapter.flow import FlowResult
from agent_adapter.flow.state import validated_state
from agent_adapter.flow.visit import frozen
from agent_adapter.integrations.adk._steps import Plan
from agent_adapter.integrations.adk._support import (
    DRIVER,
    INTERRUPT_PREFIX,
    REQUEST_INPUT,
    genai_types,
    require_adk,
)
from agent_adapter.integrations.adk._workflow import CANCELLED
from agent_adapter.integrations.adk.compile import _flow_plan

logger = logging.getLogger(__name__)

_VISIT_RUN = re.compile(r"v(\d+)(?:r\d+)?")  # a node visit attempt's run id
_SESSIONS: list[Any] = []  # the process-wide default session service, made on first use
_SESSIONS_LOCK = threading.Lock()
# One asyncio.Lock per (event loop, app, user, session) in this process; dropped when unused.
_RESUMES: weakref.WeakValueDictionary[tuple[int, str, str, str], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)


def _session_lock(app: str, user_id: str, session_id: str) -> asyncio.Lock:
    """The lock that serialises runs of one session in this process (and event loop)."""
    key = (id(asyncio.get_running_loop()), app, user_id, session_id)
    with _SESSIONS_LOCK:
        lock = _RESUMES.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _RESUMES[key] = lock
        return lock


def _default_sessions() -> Any:
    with _SESSIONS_LOCK:
        if not _SESSIONS:
            from google.adk.sessions import InMemorySessionService

            _SESSIONS.append(InMemorySessionService())
        return _SESSIONS[0]


async def run_workflow(
    workflow: Any,
    state: Any,
    *,
    session_service: Any = None,
    user_id: str = "user",
) -> FlowResult[Any]:
    """Run a `compile_flow` Workflow from `state`, or continue it with a `resume_message`.

    Pass the initial state (a model or mapping) to start a run in a new session, or the
    `types.Content` from `resume_message(...)` to resume the paused run in its session.
    `checkpoint` is always None: ADK's session service holds the run. `visits` counts the
    finished node visits of this run, de-duplicated by node path. Runs of one session are
    serialised in this process; across processes the host must serialise them (a losing
    concurrent resume raises `ApprovalAlreadyDecided`, it never runs a call twice).

    Args:
        session_service: Where sessions live; default one in-memory service per process.

    Raises:
        ToolUsageError: `workflow` is not compiled by `compile_flow`, or a resume message
            names no flow interrupt.
        FlowError: as the local runner (the run raises what a node or merge raised).
        ApprovalAlreadyDecided: the answered approval was already used (a concurrent or
            repeated resume in another process).
    """
    require_adk()
    from google.adk.runners import Runner

    plan = _flow_plan(workflow)
    sessions = session_service if session_service is not None else _default_sessions()
    app = workflow.name
    runner = Runner(app_name=app, node=workflow, session_service=sessions)
    types = genai_types()
    if isinstance(state, types.Content):
        session_id, message = _resumed_session(state), state
    else:
        session = await sessions.create_session(app_name=app, user_id=user_id)
        session_id, message = session.id, _first_message(plan, state)
    async with _session_lock(app, user_id, session_id):  # a second resume replays the first
        run = runner.run_async(user_id=user_id, session_id=session_id, new_message=message)
        events = [event async for event in run]
    invocation = next((e.invocation_id for e in events if e.invocation_id), "")
    session = await sessions.get_session(app_name=app, user_id=user_id, session_id=session_id)
    history = [
        e for e in getattr(session, "events", None) or events if e.invocation_id == invocation
    ]
    return _result(plan, events, history, cancelled=CANCELLED.take(invocation))


def _first_message(plan: Plan, state: Any) -> Any:
    model = plan.graph.state_model
    if not isinstance(state, model):
        if not isinstance(state, Mapping):
            raise ToolUsageError(
                f"state must be a {model.__name__}, a mapping or a resume message, "
                f"got {type(state).__name__}"
            )
        state = validated_state(model, state, what=f"initial state of flow {plan.graph.name!r}")
    types = genai_types()
    text = json.dumps(to_jsonable(state), ensure_ascii=False)
    return types.Content(role="user", parts=[types.Part(text=text)])


def _resumed_session(message: Any) -> str:
    """The session id inside the flow interrupt a resume message answers."""
    for part in message.parts or ():
        response = getattr(part, "function_response", None)
        interrupt = getattr(response, "id", None) or ""
        if interrupt.startswith(INTERRUPT_PREFIX) and ":" in interrupt[len(INTERRUPT_PREFIX) :]:
            return interrupt[len(INTERRUPT_PREFIX) :].rsplit(":", 1)[0]
    raise ToolUsageError("the message answers no interrupt of a compiled flow")


def _result(
    plan: Plan, events: Sequence[Any], history: Sequence[Any], *, cancelled: bool = False
) -> FlowResult[Any]:
    """The FlowResult the run's events describe: paused, done or denied."""
    visits = frozen(_visits(plan, history))
    for event in events:
        for call in event.get_function_calls():
            if call.name == REQUEST_INPUT and (call.id or "").startswith(INTERRUPT_PREFIX):
                data = (call.args or {}).get("payload", {}).get("aa", {})
                pending = ApprovalRequest.model_validate(data["request"])
                return FlowResult(
                    state=_state(plan, data["state"]),
                    status="paused",
                    steps=int(data["steps"]),
                    visits=visits,
                    pending=pending,
                )
    for event in reversed(events):
        output = event.output
        path = getattr(getattr(event, "node_info", None), "path", "") or ""
        if path.endswith(f"/{DRIVER}@1") and isinstance(output, Mapping) and "status" in output:
            return FlowResult(
                state=_state(plan, output["state"]),
                status=output["status"],
                steps=int(output["steps"]),
                visits=visits,
                denied_at=output.get("denied_at"),
            )
    flow = plan.graph.name
    if cancelled:
        logger.warning("flow %s: cancelled inside ADK", flow)
        raise FlowError(
            f"flow {flow!r}: cancelled inside ADK (code in the flow raised CancelledError, "
            "which ADK's Runner turns into a quiet end of the turn)",
            flow=flow,
        )
    raise FlowError(f"flow {flow!r}: the run ended without a result", flow=flow)


def _state(plan: Plan, data: Any) -> BaseModel:
    return validated_state(plan.graph.state_model, data, what=f"state of flow {plan.graph.name!r}")


def _visits(plan: Plan, history: Sequence[Any]) -> dict[str, int]:
    """Finished visits per node, from the node paths of this run (each path counted once)."""
    seen: set[tuple[str, str]] = set()
    for event in history:
        output = event.output
        path = getattr(getattr(event, "node_info", None), "path", "") or ""
        parent, _, leaf = path.rpartition("/")
        name, _, run_id = leaf.partition("@")
        found = _VISIT_RUN.fullmatch(run_id)
        if not (parent.endswith(f"/{DRIVER}@1") or parent == f"{DRIVER}@1"):
            continue
        if (
            found
            and name in plan.graph.nodes
            and isinstance(output, Mapping)
            and "updates" in output
        ):
            seen.add((name, found.group(1)))
    counts = Counter(name for name, _ in seen)
    return {name: counts[name] for name in plan.graph.nodes if name in counts}
