"""OpenAI Agents SDK: registry tools as `FunctionTool`s, and a bridge for its approval gate.

The SDK's own approval (`RunState.approve`, `always_approve`, a stored run state) is never
taken as an approval. `resolve_interruptions` asks the port and issues a one-use token per
approved call; without a matching token the executor asks its own port when the call runs.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from agent_adapter.approval.batch import ask_each
from agent_adapter.approval.builtin import evaluate_confirm
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalDenied,
    IntegrationError,
    ToolArgsInvalid,
    ToolUsageError,
)
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec, strict_downgraded
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest, to_jsonable
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_approval import Grant, issue_tokens, validated_call
from agent_adapter.integrations._native_gate import (
    NO_CALL_ID,
    TokenTable,
    check_framework,
    denial,
    dump_arguments,
)
from agent_adapter.integrations.base import (
    MODEL_VISIBLE_ERRORS,
    IntegrationBase,
    ToolRunner,
    model_visible_error,
)
from agent_adapter.runtime.ids import new_id
from agent_adapter.schema.transforms import to_openai_parameters

if TYPE_CHECKING:
    from agents import FunctionTool, RunContextWrapper
    from agents.run_state import RunState

logger = logging.getLogger(__name__)

TARGET = "openai_agents"
_SUPPORTED = ((0, 24), (1, 0))  # the `openai-agents` extra's range
_NO_TRACE = "no-op"  # the Agents SDK's trace id when tracing is disabled
_SPEC_ATTR = "_agent_adapter_spec"  # set on exported tools, so a resolve finds their spec
# One-use approvals `resolve_interruptions` issues, bound to run state, call id, tool and args.
_TOKENS = TokenTable()

_NeedsApproval = Callable[["RunContextWrapper[Any]", dict[str, Any], str], Awaitable[bool]]
_Plan = tuple[ToolSpec | None, ApprovalRequest | ApprovalDecision]


def function_tool(
    spec: ToolSpec, *, executor: ToolRunner | None = None, strict: bool = True
) -> FunctionTool:
    """A `FunctionTool` that runs `spec` through the executor; `confirm` maps to `needs_approval`.

    Raises:
        MissingDependency: `openai-agents` is not installed.
        ToolUsageError: `spec.targets` excludes `openai_agents`.
        IntegrationError: the installed SDK is outside the extra's range, lacks the run-state
            scope the approval tokens are bound to, or cannot switch off its failure formatter.
    """
    INTEGRATION.require()
    check_target(spec, TARGET)
    _check_sdk()
    from agents import FunctionTool

    strict = strict and not strict_downgraded(spec)
    runner = lazy_runner(executor)

    async def on_invoke_tool(native: Any, arguments: str) -> Any:
        return await _invoke(spec, runner, native, arguments)

    tool = FunctionTool(
        name=spec.name,
        description=spec.description,
        params_json_schema=to_openai_parameters(spec.json_schema(), strict=strict),
        on_invoke_tool=on_invoke_tool,
        strict_json_schema=strict,
        needs_approval=_needs_approval(spec),
    )
    setattr(tool, _SPEC_ATTR, spec)
    return _without_failure_formatter(tool)


def _check_sdk() -> None:
    import agents
    from agents import agent_tool_state
    from agents.run_state import RunState

    check_framework(
        TARGET,
        str(getattr(agents, "__version__", "0")),
        _SUPPORTED,
        {
            "agent_tool_state.get_agent_tool_state_scope": hasattr(
                agent_tool_state, "get_agent_tool_state_scope"
            ),
            "RunState._agent_tool_state_scope_id": "_agent_tool_state_scope_id"
            in getattr(RunState, "__dataclass_fields__", {}),
        },
    )


def _without_failure_formatter(tool: FunctionTool) -> FunctionTool:
    """Stop the SDK turning a tool-raised CancelledError into model text."""
    try:
        from agents.tool import set_function_tool_failure_error_function
    except ImportError as exc:
        raise IntegrationError(
            "this openai-agents version has no set_function_tool_failure_error_function; "
            "control flow could not propagate from tools",
            integration=TARGET,
        ) from exc
    set_function_tool_failure_error_function(tool, None)
    return tool


def function_tools(
    registry: Registry, *, executor: ToolRunner | None = None, strict: bool = True
) -> list[FunctionTool]:
    """`function_tool` for every registry tool exported to `openai_agents`."""
    return [
        function_tool(spec, executor=executor, strict=strict)
        for spec in registry.for_target(TARGET)
    ]


async def resolve_interruptions(
    result: Any, port: ApprovalPort, *, run_id: str = ""
) -> RunState[Any]:
    """Ask `port` about each pending approval and record its decision on the run state.

    `result` is a `RunResult` (its `to_state()` is used) or a `RunState` (updated in place).
    Pass the returned state to `Runner.run(agent, state)` to continue. Each approved call of
    an exported tool gets a one-use token bound to this state, its call id, the tool and its
    validated arguments; edits travel in the token. Pass the same `run_id` on every pass, so
    an out-of-band store sees one run.

    Raises:
        ApprovalPending: a decision is still open; nothing was applied or consumed.
        ToolUsageError: a decision edits the arguments of a tool this bridge did not export
            (the edit could not reach it); raised before any decision is used.
    """
    state: RunState[Any] = result.to_state() if hasattr(result, "to_state") else result
    items = state.get_interruptions()
    planned = [_plan(item, run_id) for item in items]
    asks = [ask for _, ask in planned if isinstance(ask, ApprovalRequest)]
    foreign = {a.approval_id for s, a in planned if s is None and isinstance(a, ApprovalRequest)}

    def vet(request: ApprovalRequest, decision: ApprovalDecision) -> None:
        edits = decision.approved and decision.edited_args is not None
        if edits and request.approval_id in foreign:
            raise ToolUsageError(
                f"the decision for {request.tool!r} edits its arguments, but that tool was "
                "not exported by this bridge, so the edit could not reach it"
            )

    decisions = iter(await ask_each(port, asks, vet=vet))
    scope = _scope(state)
    grants: list[Grant] = []
    for item, (spec, ask) in zip(items, planned, strict=True):
        if isinstance(ask, ApprovalDecision):
            _record(state, item, ask)
            continue
        decision = next(decisions)
        _record(state, item, decision)
        if spec is not None and decision.approved:
            grants.append(Grant(ask.tool, ask.arguments, decision, ask.call_id, scope))
    if grants and not scope:
        logger.warning(
            "%s: %d call(s) approved, but the run state has no scope to bind them to; "
            "the executor asks again when they run",
            TARGET,
            len(grants),
        )
    elif grants:
        issue_tokens(_TOKENS, grants, integration=TARGET)
    return state


def _plan(item: Any, run_id: str) -> _Plan:
    """The request to ask (validated arguments for our tools), or a denial decided here."""
    tool, call_id = str(item.tool_name or ""), str(item.call_id or "")
    arguments = _parse_arguments(item.arguments)
    spec = _spec_of(item)
    if spec is None:
        return None, _approval_request(tool, arguments, call_id, run_id)
    if not call_id:
        logger.warning("%s: %s has no call id; denied", TARGET, tool)
        return spec, denial(NO_CALL_ID, by=TARGET)
    dumped = dump_arguments(spec, arguments)
    if dumped is None:
        return spec, denial("invalid arguments", by=TARGET)
    return spec, _approval_request(tool, dumped, call_id, run_id)


def _spec_of(item: Any) -> ToolSpec | None:
    """The spec of the exported tool an interruption belongs to; None for other tools."""
    for tool in getattr(getattr(item, "agent", None), "tools", None) or ():
        spec = getattr(tool, _SPEC_ATTR, None)
        if isinstance(spec, ToolSpec) and getattr(tool, "name", None) == item.tool_name:
            return spec
    return None


def _record(state: Any, item: Any, decision: ApprovalDecision) -> None:
    """Apply one decision to the run state (the SDK's gate only; tokens are issued apart)."""
    if not decision.approved:
        tool, call_id = str(item.tool_name or ""), str(item.call_id or "")
        denied = ApprovalDenied(tool, call_id=call_id, reason=decision.reason)
        state.reject(item, rejection_message=json.dumps(model_visible_error(denied)))
        return
    state.approve(item)


def _approval_request(
    tool: str, arguments: dict[str, Any], call_id: str, run_id: str
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=new_id("apr_"),
        run_id=run_id,
        call_id=call_id,
        tool=tool,
        arguments=arguments,
        metadata={"integration": TARGET},
        created_at=time.time(),
    )


def _parse_arguments(arguments: str | None) -> dict[str, Any]:
    """The model's JSON arguments as a dict; anything else becomes `{}` (logged)."""
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        logger.warning("tool call arguments are not valid JSON; asking with no arguments")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _needs_approval(spec: ToolSpec) -> bool | _NeedsApproval:
    """`confirm` as the SDK's `needs_approval`; invalid arguments are never sent for approval."""
    rule = spec.confirm
    if rule is False:
        return False

    async def needs_approval(native: Any, arguments: dict[str, Any], call_id: str) -> bool:
        try:
            call = validated_call(spec, arguments, call_id)
        except ToolArgsInvalid:
            return False  # the call fails validation in the executor; nothing to approve
        if rule is True:
            return True
        ctx = ToolContext(run_id=_trace_id(), call_id=call_id, extras={TARGET: native})
        return await evaluate_confirm(rule, call, ctx)

    return needs_approval


async def _invoke(spec: ToolSpec, runner: ToolRunner, native: Any, arguments: str) -> Any:
    """Run one SDK tool call; model-visible errors come back as a JSON error string."""
    ctx = INTEGRATION.bind_context(native)
    try:
        args = _decode(spec, arguments)
        ctx.approval = _take(spec, args, ctx.call_id, native)
        value = await runner.run_spec(spec, args, ctx, call_id=ctx.call_id or None)
    except MODEL_VISIBLE_ERRORS as exc:
        logger.info("tool %s returned an error to the model: %s", spec.name, exc.kind)
        return json.dumps(model_visible_error(exc))
    return value if isinstance(value, str) else json.dumps(to_jsonable(value))


def _take(
    spec: ToolSpec, args: dict[str, Any], call_id: str, native: Any
) -> ApprovalDecision | None:
    """The decision of the token issued for exactly this call; None lets the executor decide."""
    if spec.confirm is False:
        return None
    scope, dumped = _scope(native), dump_arguments(spec, args)
    if not scope or not call_id or dumped is None:
        return None
    token = _TOKENS.take(spec.name, dumped, call_id=call_id, run_id=scope)
    return token.decision if token is not None else None


def _scope(holder: Any) -> str:
    """The SDK's id of one resumable run state, shared by the tool contexts that resume it.

    `to_state()` and `RunState.from_json` each make a new one, so a token never outlives the
    state object it was issued for. Works on a `RunState` and on a tool context.
    """
    from agents.agent_tool_state import get_agent_tool_state_scope

    return get_agent_tool_state_scope(holder) or ""


def _decode(spec: ToolSpec, arguments: str) -> dict[str, Any]:
    """The SDK's JSON argument string as a dict; ToolArgsInvalid when it is not an object."""
    try:
        parsed = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as exc:
        raise ToolArgsInvalid(
            spec.name, [{"type": "json_invalid", "loc": (), "msg": f"invalid JSON: {exc.msg}"}]
        ) from exc
    if not isinstance(parsed, dict):
        raise ToolArgsInvalid(
            spec.name, [{"type": "dict_type", "loc": (), "msg": "arguments must be an object"}]
        )
    return parsed


def _trace_id() -> str:
    """The current Agents SDK trace id (kept across resumes), or "" when tracing is off."""
    from agents.tracing import get_current_trace

    trace = get_current_trace()
    trace_id = getattr(trace, "trace_id", "") if trace is not None else ""
    return trace_id if isinstance(trace_id, str) and trace_id != _NO_TRACE else ""


class OpenAIAgentsIntegration(IntegrationBase):
    """Exports `FunctionTool`s; the SDK's `needs_approval` gate is the approval bridge."""

    name = TARGET
    requires = ("agents",)
    extra = "openai-agents"
    call_id_fields = ("tool_call_id",)  # agents.tool_context.ToolContext.tool_call_id

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> FunctionTool:
        """`function_tool(spec, executor=executor)`."""
        return function_tool(spec, executor=executor)

    def bind_context(self, native: Any) -> ToolContext:
        """The call id from the SDK's tool context; the run id from its trace, when traced.

        Never an approval: the SDK's gate only adds a pause in front of the executor's.
        """
        ctx = super().bind_context(native)
        ctx.run_id = _trace_id()
        return ctx


INTEGRATION = OpenAIAgentsIntegration()


__all__ = [
    "INTEGRATION",
    "OpenAIAgentsIntegration",
    "function_tool",
    "function_tools",
    "resolve_interruptions",
]
