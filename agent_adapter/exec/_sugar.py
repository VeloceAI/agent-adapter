"""Shortcuts behind the Executor: run by function, the shared sync loop, one-call resume."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from agent_adapter.approval.deferred import server_owned
from agent_adapter.approval.store import ApprovalRecord, ApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalAlreadyDecided, ToolUsageError
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import SPEC_ATTR, Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ToolCall
from agent_adapter.exec._approve import approval_store, same_decision
from agent_adapter.runtime._fork import ForkSafeLock
from agent_adapter.runtime.loop_runner import LoopRunner

logger = logging.getLogger(__name__)

T = TypeVar("T")

# What `Executor.run` accepts: a tool name, a registered spec, or the registered function.
ToolRef = str | ToolSpec | Callable[..., Any]

USED_ONCE = (
    "a decision runs its call once; to run the tool again, make a new call "
    "(it asks for a new approval)"
)

_runner: LoopRunner | None = None
_runner_lock = ForkSafeLock()


def shared_runner() -> LoopRunner:
    """The process-wide loop every sync twin runs on, so loop-bound state stays warm."""
    global _runner
    with _runner_lock:
        if _runner is None or _runner.closed:
            _runner = LoopRunner(name="agent_adapter-sync")
            logger.debug("started the shared loop for sync calls")
        return _runner


def blocking(work: Callable[[], Awaitable[T]], timeout: float | None) -> T:
    """Run `work` on the shared loop and block until it is done.

    Raises:
        TimeoutError: past `timeout`.
        ToolUsageError: called from a tool already running on that loop (await instead).
    """
    return shared_runner().run(work, timeout=timeout)


def check_options(registry: object, approval: object) -> None:
    """Fail fast on an Executor given the wrong kind of registry or approval.

    Raises:
        ToolUsageError: `registry` is not a Registry, or `approval` is not an ApprovalPort.
    """
    if registry is not None and not isinstance(registry, Registry):
        raise ToolUsageError(
            f"Executor's registry must be a Registry, got {type(registry).__name__}: register "
            "functions with @tool (the default registry, used by Executor()) or "
            "Registry().tool, then pass that Registry"
        )
    if approval is not None and not isinstance(approval, ApprovalPort):
        raise ToolUsageError(
            f"approval must be an ApprovalPort (an object with async request()), got "
            f"{type(approval).__name__}: use AutoApprove() in tests, DeferredApproval(store) "
            "for reviews (DeferredApproval.sqlite(path) for a file), or CallbackApproval(fn)"
        )


def check_args(args: object) -> dict[str, Any]:
    """`args` as a dict.

    Raises:
        ToolUsageError: `args` is not a mapping.
    """
    if args is None:
        return {}
    if not isinstance(args, Mapping):
        raise ToolUsageError(
            "args must be a mapping of argument names to values, e.g. {'a': 1}, "
            f"got {type(args).__name__}"
        )
    return dict(args)


def tool_name(tool: object, registry: Registry | None) -> str:
    """The registered name of `tool`: a name, a spec of `registry`, or its function.

    Raises:
        ToolUsageError: a spec or function `registry` does not hold, or not a tool at all.
    """
    if isinstance(tool, str):
        return tool
    if isinstance(tool, ToolSpec):
        if registry is not None and _holds(registry, tool):
            return tool.name
        raise ToolUsageError(
            f"tool spec {tool.name!r} is not in this executor's registry: run it with "
            "executor.run_spec(spec, args), or register it with registry.add(spec)"
        )
    if callable(tool):
        return _function_name(tool, registry)
    raise ToolUsageError(
        "run needs a tool name (str) or a function registered with @tool, "
        f"got {type(tool).__name__}"
    )


def _holds(registry: Registry, spec: ToolSpec) -> bool:
    return spec.name in registry and registry.get(spec.name) is spec


def _function_name(fn: Callable[..., Any], registry: Registry | None) -> str:
    marked = getattr(fn, SPEC_ATTR, None)
    if registry is not None and isinstance(marked, ToolSpec) and _holds(registry, marked):
        return marked.name
    found = [] if registry is None else [spec.name for spec in registry if spec.func == fn]
    if len(found) == 1:
        return found[0]
    label = getattr(fn, "__qualname__", None) or repr(fn)
    if found:
        raise ToolUsageError(
            f"{label!r} is registered as several tools {found}: pass the tool name instead"
        )
    if isinstance(marked, ToolSpec):
        raise ToolUsageError(
            f"tool {marked.name!r} is registered in another Registry, not this executor's: "
            "pass that registry to Executor(registry)"
        )
    raise ToolUsageError(
        f"{label!r} is not a tool: decorate it with @tool (or registry.tool), or register "
        "it with registry.wrap(fn), then run it by function or by name"
    )


async def resume_target(
    port: ApprovalPort | None,
    approval_id: str,
    decision: bool | ApprovalDecision | None,
    *,
    decided_by: str,
    reason: str,
    ctx: ToolContext | None = None,
) -> tuple[ToolCall, str]:
    """Decide a parked tool call in the port's store; return it, rebuilt, and its run id.

    The rebuilt call has the record's ids, tool and original arguments, so the port answers
    it with this record's decision, once. A pending record needs an explicit `decision`.

    Raises:
        ToolUsageError: no store, a flow node, a record the HTTP server parked, `ctx` of
            another run, a pending record without a decision, or no `decided_by`.
        ApprovalNotFound: no such record.
        ApprovalAlreadyDecided: already consumed, or decided otherwise than `decision`.
    """
    store = _store_of(port)
    record = await store.get(approval_id)
    check_resumable(record)
    if record.consumed:
        raise used_once(approval_id)
    _same_run(ctx, record.request.run_id)  # before anything is decided
    if record.status == "pending":
        if decision is None:
            raise ToolUsageError(
                f"approval {approval_id!r} is still pending: pass the decision, e.g. "
                "executor.resume(approval_id, True, decided_by='reviewer')"
            )
        await _decide(store, approval_id, _signed(_as_decision(decision, reason), decided_by))
    elif decision is not None and not same_decision(
        record.decision, _as_decision(decision, reason)
    ):
        raise ApprovalAlreadyDecided(approval_id, status=record.status)
    request = record.request
    logger.info(
        "resuming approval %s for tool %r (run %s, call %s)",
        approval_id,
        request.tool,
        request.run_id,
        request.call_id,
    )
    call = ToolCall(name=request.tool, arguments=dict(request.arguments), id=request.call_id)
    return call, request.run_id


def check_resumable(record: ApprovalRecord) -> None:
    """Refuse a record no executor or queue may resume: a flow node, or a server record.

    Raises:
        ToolUsageError: `record` pauses a flow node, or the HTTP tool server parked it.
    """
    request = record.request
    if request.kind != "tool":
        raise ToolUsageError(
            f"approval {record.approval_id!r} pauses flow node {request.tool!r}, not a tool "
            "call: resume it with FlowRunner.resume"
        )
    if server_owned(record):
        logger.warning(
            "refused to resume approval %s for tool %r (run %s, call %s): the HTTP tool "
            "server parked it",
            record.approval_id,
            request.tool,
            request.run_id,
            request.call_id,
        )
        raise ToolUsageError(
            f"approval {record.approval_id!r} was parked by the HTTP tool server, which "
            f"re-authorizes the invoker: decide it there (POST /approvals/{record.approval_id}"
            "/decide)"
        )


def used_once(approval_id: str) -> ApprovalAlreadyDecided:
    """The error for a resume of a record whose decision was already used."""
    error = ApprovalAlreadyDecided(approval_id, status="consumed")
    error.add_note(USED_ONCE)
    return error


async def _decide(store: ApprovalStore, approval_id: str, wanted: ApprovalDecision) -> None:
    """Record `wanted`; a concurrent decide that agrees with it is not an error."""
    try:
        await store.decide(approval_id, wanted)
    except ApprovalAlreadyDecided:
        current = await store.get(approval_id)
        if current.consumed:
            raise used_once(approval_id) from None
        if not same_decision(current.decision, wanted):
            raise
        logger.info("approval %s: a concurrent decide agreed with this one", approval_id)


def resume_context(
    ctx: ToolContext | None, run_id: str, make: Callable[[], ToolContext]
) -> ToolContext:
    """`ctx` when it is for `run_id`, else a new context from `make`.

    Raises:
        ToolUsageError: `ctx` is for another run.
    """
    if ctx is None:
        return make()
    _same_run(ctx, run_id)
    return ctx


def _same_run(ctx: ToolContext | None, run_id: str) -> None:
    if ctx is not None and ctx.run_id != run_id:
        raise ToolUsageError(
            f"this approval belongs to run {run_id!r}, but ctx is for run {ctx.run_id!r}: "
            "pass ctx=None or a context of the same run"
        )


def _store_of(port: ApprovalPort | None) -> ApprovalStore:
    store = approval_store(port)
    if store is None:
        raise ToolUsageError(
            f"resume needs an approval port with a store, got {type(port).__name__}: "
            "Executor(approval=DeferredApproval(store))"
        )
    return store


def _as_decision(decision: bool | ApprovalDecision, reason: str) -> ApprovalDecision:
    if isinstance(decision, ApprovalDecision):
        return decision
    if isinstance(decision, bool):
        return ApprovalDecision(approved=decision, reason=reason)
    raise ToolUsageError(
        f"decision must be True, False or an ApprovalDecision, got {type(decision).__name__}"
    )


def _signed(decision: ApprovalDecision, decided_by: str) -> ApprovalDecision:
    """`decision` naming who decided and when, for the audit trail.

    Raises:
        ToolUsageError: neither `decision` nor `decided_by` names the reviewer.
    """
    who = decision.decided_by or decided_by
    if not who:
        raise ToolUsageError(
            "resume records who decided: pass decided_by='reviewer' (or an "
            "ApprovalDecision with decided_by)"
        )
    return decision.model_copy(
        update={"decided_by": who, "decided_at": decision.decided_at or time.time()}
    )
