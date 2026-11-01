"""Pydantic AI: registry tools as `Tool`s, and a bridge for its deferred-approval flow.

Pydantic AI's own approval (`tool_call_approved`, any `DeferredToolResults`) is never taken
as an approval. `resolve_deferred` asks the port and issues a one-use token per approved
call one of these tools paused on, bound to that call's `ToolCallPart` object. Without a
matching token the executor asks its own port when the call runs.
"""

from __future__ import annotations

import json
import logging
import time
import weakref
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agent_adapter.approval.batch import ask_each
from agent_adapter.approval.builtin import evaluate_confirm
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalDenied, ToolArgsInvalid
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_approval import Grant, issue_tokens, validated_call
from agent_adapter.integrations._native_gate import TokenTable, dump_arguments
from agent_adapter.integrations.base import (
    MODEL_VISIBLE_ERRORS,
    IntegrationBase,
    ToolRunner,
    model_visible_error,
)
from agent_adapter.integrations.pydantic_ai_compress import (
    pydantic_ai_history_processor,  # registers the format
)
from agent_adapter.runtime.ids import new_id

if TYPE_CHECKING:
    from pydantic_ai import DeferredToolRequests, DeferredToolResults, Tool

logger = logging.getLogger(__name__)

TARGET = "pydantic_ai"
# One-use approvals `resolve_deferred` issues, bound to the paused call, call id, tool and args.
_TOKENS = TokenTable()


class _PausedCalls:
    """The `ToolCallPart`s this bridge's tools paused on, by object identity, while alive.

    Parts are unhashable dataclasses, so entries are keyed by `id()` and checked with a
    weak reference; an entry goes when its part is collected.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[weakref.ref[Any], ToolSpec, str]] = {}

    def add(self, part: Any, spec: ToolSpec) -> None:
        """Remember that `spec` paused on `part`, under a fresh key."""
        if self.find(part) is not None:
            return
        key = id(part)
        try:
            ref = weakref.ref(part, lambda _: self._entries.pop(key, None))
        except TypeError:
            logger.warning("%s: cannot track the paused call of %s", TARGET, spec.name)
            return
        self._entries[key] = (ref, spec, new_id("tcp_"))

    def find(self, part: Any) -> tuple[ToolSpec, str] | None:
        """The spec and key `part` was paused under; None for any other object."""
        entry = self._entries.get(id(part))
        if entry is None or entry[0]() is not part:
            return None
        return entry[1], entry[2]


_PAUSED = _PausedCalls()


def pydantic_ai_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Tool[Any]:
    """A `Tool` that runs `spec` through the executor.

    With `confirm` set, the tool validates the arguments first and raises
    `ApprovalRequired` when the rule (or `True`) asks for approval, so the agent needs
    `DeferredToolRequests` in its `output_type`. Invalid arguments raise `ModelRetry`.

    Raises:
        MissingDependency: `pydantic-ai` is not installed.
        ToolUsageError: `spec.targets` excludes `pydantic_ai`.
    """
    INTEGRATION.require()
    check_target(spec, TARGET)
    from pydantic_ai import Tool

    runner = lazy_runner(executor)

    async def call(native: Any, /, **arguments: Any) -> Any:
        return await _invoke(spec, runner, native, arguments)

    tool: Tool[Any] = Tool.from_schema(
        call,
        name=spec.name,
        description=spec.description,
        json_schema=spec.json_schema(),
        takes_ctx=True,
    )
    return tool


def pydantic_ai_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Tool[Any]]:
    """`pydantic_ai_tool` for every registry tool exported to `pydantic_ai`."""
    return [pydantic_ai_tool(spec, executor=executor) for spec in registry.for_target(TARGET)]


async def resolve_deferred(
    requests: DeferredToolRequests, port: ApprovalPort, *, run_id: str = ""
) -> DeferredToolResults:
    """Ask `port` about each call in `requests.approvals`; the results resume the agent.

    Approvals become `ToolApproved(override_args=edited_args)`, denials `ToolDenied` with a
    model-visible error. Each approved call that one of this bridge's tools paused on gets a
    one-use token bound to that call's `ToolCallPart` object (the resumed history must hold
    the same object), its call id, the tool and the validated arguments it runs with.
    Resume once, in this process, within a minute. Calls deferred for external execution
    (`requests.calls`) are left to the host. Pass the previous run's `run_id` so out-of-band
    stores see one run.

    Raises:
        ApprovalPending: a decision is still open; nothing was consumed. Pass the same
            `run_id` on every pass.
    """
    from pydantic_ai import DeferredToolResults, ToolApproved, ToolDenied

    paused = [_PAUSED.find(part) for part in requests.approvals]
    asks = [
        _approval_request(part, requests.metadata, run_id, ours)
        for part, ours in zip(requests.approvals, paused, strict=True)
    ]
    approvals: dict[str, Any] = {}
    grants: list[Grant] = []
    for request, ours, decision in zip(asks, paused, await ask_each(port, asks), strict=True):
        if not decision.approved:
            denied = ApprovalDenied(request.tool, call_id=request.call_id, reason=decision.reason)
            approvals[request.call_id] = ToolDenied(json.dumps(model_visible_error(denied)))
            continue
        edited = dict(decision.edited_args) if decision.edited_args is not None else None
        approvals[request.call_id] = ToolApproved(override_args=edited)
        if ours is not None:
            spec, key = ours
            runs_with = dump_arguments(spec, edited if edited is not None else request.arguments)
            if runs_with is not None:
                grants.append(Grant(spec.name, runs_with, decision, request.call_id, key))
    issue_tokens(_TOKENS, grants, integration=TARGET)
    return DeferredToolResults(approvals=approvals)


def _approval_request(
    part: Any,
    metadata: Mapping[str, Mapping[str, Any]],
    run_id: str,
    ours: tuple[ToolSpec, str] | None,
) -> ApprovalRequest:
    """What the port is asked: validated arguments for our tools, else the model's raw ones."""
    call_id = str(part.tool_call_id)
    raw = part.args_as_dict()
    dumped = dump_arguments(ours[0], raw) if ours is not None else None
    return ApprovalRequest(
        approval_id=new_id("apr_"),
        run_id=run_id,
        call_id=call_id,
        tool=str(part.tool_name),
        arguments=dumped if dumped is not None else raw,
        metadata={"integration": TARGET, "native": dict(metadata.get(call_id, {}))},
        created_at=time.time(),
    )


async def _invoke(
    spec: ToolSpec, runner: ToolRunner, native: Any, arguments: dict[str, Any]
) -> Any:
    """Run one Pydantic AI tool call through `runner`."""
    from pydantic_ai.exceptions import ApprovalRequired, ModelRetry

    ctx = INTEGRATION.bind_context(native)
    try:
        if spec.confirm is not False:
            ctx.approval = _take(spec, arguments, ctx.call_id, native)
            resumed = getattr(native, "tool_call_approved", False) is True
            if ctx.approval is None and not resumed and await _needs_approval(spec, arguments, ctx):
                logger.info("tool %s needs approval (call %s)", spec.name, ctx.call_id)
                _pause(spec, native, ctx.call_id)
                raise ApprovalRequired(metadata={"tool": spec.name})
        return await runner.run_spec(spec, arguments, ctx, call_id=ctx.call_id or None)
    except ToolArgsInvalid as exc:
        raise ModelRetry(str(exc)) from exc
    except MODEL_VISIBLE_ERRORS as exc:
        logger.info("tool %s returned an error to the model: %s", spec.name, exc.kind)
        return model_visible_error(exc)


def _pause(spec: ToolSpec, native: Any, call_id: str) -> None:
    """Remember the `ToolCallPart` this call paused on, so `resolve_deferred` can bind to it."""
    part = _call_part(native, spec.name, call_id)
    if part is None:
        logger.warning(
            "%s: no tool call part for %s (call %s) in the run's messages; "
            "the executor asks again when it resumes",
            TARGET,
            spec.name,
            call_id,
        )
        return
    _PAUSED.add(part, spec)


def _take(
    spec: ToolSpec, arguments: dict[str, Any], call_id: str, native: Any
) -> ApprovalDecision | None:
    """The decision of the token issued for exactly this call; None lets the executor decide."""
    found = _PAUSED.find(_call_part(native, spec.name, call_id))
    dumped = dump_arguments(spec, arguments)
    if found is None or found[0] is not spec or dumped is None or not call_id:
        return None
    token = _TOKENS.take(spec.name, dumped, call_id=call_id, run_id=found[1])
    return token.decision if token is not None else None


def _call_part(native: Any, tool: str, call_id: str) -> Any:
    """The latest `ToolCallPart` for this call in the run's messages; None without one."""
    if not call_id:
        return None
    for message in reversed(getattr(native, "messages", None) or ()):
        for part in getattr(message, "parts", ()):
            if (
                getattr(part, "part_kind", "") == "tool-call"
                and getattr(part, "tool_call_id", None) == call_id
                and getattr(part, "tool_name", None) == tool
            ):
                return part
    return None


async def _needs_approval(spec: ToolSpec, arguments: dict[str, Any], ctx: ToolContext) -> bool:
    """Whether `confirm` wants approval for these arguments; validates them first.

    Raises:
        ToolArgsInvalid: the arguments fail validation (never sent for approval).
    """
    call = validated_call(spec, arguments, ctx.call_id)
    return spec.confirm is True or await evaluate_confirm(spec.confirm, call, ctx)


class PydanticAIIntegration(IntegrationBase):
    """Exports `Tool`s; Pydantic AI's deferred tool requests are the approval bridge."""

    name = TARGET
    requires = ("pydantic_ai",)
    extra = "pydantic-ai"
    call_id_fields = ("tool_call_id",)  # RunContext.tool_call_id

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Tool[Any]:
        """`pydantic_ai_tool(spec, executor=executor)`."""
        return pydantic_ai_tool(spec, executor=executor)

    def bind_context(self, native: Any) -> ToolContext:
        """Call id, run id and thread (`conversation_id`) from the `RunContext`.

        Never an approval: Pydantic AI's deferral only adds a pause in front of the executor's.
        """
        ctx = super().bind_context(native)
        ctx.run_id = _text(getattr(native, "run_id", ""))
        ctx.thread_id = _text(getattr(native, "conversation_id", ""))
        return ctx


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


INTEGRATION = PydanticAIIntegration()


__all__ = [
    "INTEGRATION",
    "PydanticAIIntegration",
    "pydantic_ai_history_processor",
    "pydantic_ai_tool",
    "pydantic_ai_tools",
    "resolve_deferred",
]
