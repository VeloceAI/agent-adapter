"""ADK tools: one `BaseTool` per ToolSpec; confirmations resolved from the approval port.

`check_require_confirmation` applies the spec's `confirm` rule, so ADK pauses the call and
asks for a ToolConfirmation. ADK's `confirmed` alone never approves: the call gets
`ctx.approval` only from a token `resolve_confirmations` issued after the port approved
(see `_tokens`); without one the executor asks its own port when the call runs.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from agent_adapter.approval.batch import ask_each
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest, to_jsonable
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_gate import (
    denial,
    dump_arguments,
    needs_approval,
    plan_call,
    run_bound,
)
from agent_adapter.integrations.adk._support import (
    INTEGRATION,
    REQUEST_CONFIRMATION,
    TARGET,
    TOKEN_KEY,
    genai_types,
    require_adk,
)
from agent_adapter.integrations.adk._tokens import TOKENS, function_response, token_decision
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.schema.transforms import to_gemini_parameters

logger = logging.getLogger(__name__)

NO_EDITS = "argument edits are not supported for tools outside the registry"


def adk_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """An ADK `BaseTool` running `spec` through the executor; `confirm` gates via ADK.

    Raises:
        MissingDependency: `google-adk` is not installed.
        IntegrationError: the installed ADK is outside the extra's range.
        ToolUsageError: `spec.targets` excludes `adk`.
    """
    require_adk()
    check_target(spec, TARGET)
    tool: Any = _tool_class()(spec, lazy_runner(executor))
    return tool


def adk_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """An ADK `BaseTool` for every registry tool exported to `adk`."""
    return INTEGRATION.export_tools(registry, executor=executor)


def bind_context(tool_context: Any) -> ToolContext:
    """Our context for an ADK `ToolContext` (run = invocation id, call = function call id)."""
    return INTEGRATION.bind_context(tool_context)


@functools.cache
def _tool_class() -> type:
    """The `BaseTool` subclass, built on first use so `google.adk` is imported lazily."""
    from google.adk.tools.base_tool import BaseTool

    types = genai_types()
    base: Any = BaseTool

    class AdapterTool(base):  # type: ignore[misc]
        """One ToolSpec as an ADK tool; every call goes through the runner."""

        def __init__(self, spec: ToolSpec, runner: ToolRunner) -> None:
            super().__init__(name=spec.name, description=spec.description)
            self.spec = spec
            self._runner = runner

        def _get_declaration(self) -> Any:
            return types.FunctionDeclaration(
                name=self.name,
                description=self.description,
                parameters_json_schema=to_gemini_parameters(self.spec.json_schema()),
            )

        async def check_require_confirmation(self, args: dict[str, Any], tool_context: Any) -> bool:
            """Whether the `confirm` rule wants approval; invalid args never pause."""
            if self.spec.confirm is False:
                return False
            dumped = dump_arguments(self.spec, args)
            if dumped is None:
                return False
            ctx = bind_context(tool_context)
            return await needs_approval(self.spec, dumped, run_id=ctx.run_id, call_id=ctx.call_id)

        async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
            ctx = bind_context(tool_context)
            confirmation = getattr(tool_context, "tool_confirmation", None)
            if confirmation is not None:
                ctx.approval = _approval(self.spec, args, ctx, confirmation)
            return to_jsonable(await run_bound(self._runner, self.spec, args, ctx))

    return AdapterTool


def _approval(
    spec: ToolSpec, args: Mapping[str, Any], ctx: ToolContext, confirmation: Any
) -> ApprovalDecision | None:
    """The decision of the adapter token in `confirmation` for exactly this call, else None."""
    dumped = dump_arguments(spec, args)
    if dumped is None:
        return None
    request = ApprovalRequest(
        approval_id="", run_id=ctx.run_id, call_id=ctx.call_id, tool=spec.name, arguments=dumped
    )
    return token_decision(confirmation, request)


async def resolve_confirmations(
    events: Iterable[Any], port: ApprovalPort, registry: Registry
) -> Any:
    """The user `Content` answering every open tool confirmation in `events`, from `port`.

    Registry tools are asked with validated arguments (a `confirm` rule that says no is
    confirmed unasked; invalid arguments are refused); an approval carries a one-use token
    bound to the run (the events' `invocation_id`), the call id, the tool and the
    arguments, and may edit them. Other tools are asked with their raw arguments and cannot
    be edited. Returns None when nothing is open.

    Raises:
        ApprovalPending: a decision is still open (every call was asked first; nothing set).
        ToolUsageError: the open confirmations come from several invocations (ADK takes the
            answers of one invocation per message), or carry no invocation id.
    """
    found = list(events)
    answered = {response.id for event in found for response in event.get_function_responses()}
    open_calls = [
        (str(getattr(event, "invocation_id", "") or ""), call)
        for event in found
        for call in event.get_function_calls()
        if call.name == REQUEST_CONFIRMATION and call.id and call.id not in answered
    ]
    if not open_calls:
        return None
    run_id = _one_invocation({run for run, _ in open_calls})
    calls: list[tuple[str, ApprovalDecision | ApprovalRequest]] = []
    for _, call in open_calls:
        name, arguments, call_id = _original(call)
        calls.append((call.id, await plan_call(registry, name, arguments, call_id, run_id, TARGET)))
    asks = [plan for _, plan in calls if isinstance(plan, ApprovalRequest)]
    decided = iter(await ask_each(port, asks))
    parts = []
    for confirmation_id, plan in calls:
        decision = next(decided) if isinstance(plan, ApprovalRequest) else plan
        response = _response(plan, decision, registry)
        content = function_response(confirmation_id, REQUEST_CONFIRMATION, response)
        parts.extend(content.parts)
    return genai_types().Content(role="user", parts=parts)


def _one_invocation(runs: set[str]) -> str:
    """The single invocation id the open confirmations belong to.

    Raises:
        ToolUsageError: none, or more than one.
    """
    if "" in runs:
        raise ToolUsageError("open ADK confirmations must carry an invocation id (the run id)")
    if len(runs) > 1:
        raise ToolUsageError(
            f"open ADK confirmations from {len(runs)} invocations: resolve one invocation at "
            "a time (ADK takes the answers of one invocation per message)"
        )
    return next(iter(runs))


def _original(call: Any) -> tuple[str, dict[str, Any], str]:
    original = (call.args or {}).get("originalFunctionCall")
    if not isinstance(original, Mapping):
        return "", {}, ""
    arguments = original.get("args")
    return (
        str(original.get("name") or ""),
        dict(arguments) if isinstance(arguments, Mapping) else {},
        str(original.get("id") or ""),
    )


def _response(
    plan: ApprovalDecision | ApprovalRequest, decision: ApprovalDecision, registry: Registry
) -> dict[str, Any]:
    """The ToolConfirmation for one call: a token for an approved registry call."""
    if not isinstance(plan, ApprovalRequest):
        return {"confirmed": decision.approved}  # decided here: no port answer to carry
    if decision.approved and plan.tool not in registry:
        if decision.edited_args is None:
            return {"confirmed": True}  # ADK runs a foreign tool on its own confirmation
        logger.warning("approval of %s edits its arguments; denied", plan.tool)
        decision = denial(NO_EDITS, by=TARGET)
    if not decision.approved:
        return {"confirmed": False}
    secret = TOKENS.issue(plan, decision)
    if secret is None:
        logger.warning("%s (call %s) approved without a token; the port decides", *_ids(plan))
        return {"confirmed": True}
    return {"confirmed": True, "payload": {TOKEN_KEY: secret}}


def _ids(request: ApprovalRequest) -> tuple[str, str]:
    return request.tool, request.call_id


__all__ = ["adk_tool", "adk_tools", "bind_context", "resolve_confirmations"]
