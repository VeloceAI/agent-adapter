"""Haystack: each ToolSpec as a `Tool`; approval through a `before_tool` confirmation hook."""

from __future__ import annotations

import inspect
import json
import logging
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending, IntegrationError
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_gate import (
    TokenTable,
    dump_arguments,
    gate_decision,
    pending_message,
    run_bound,
)
from agent_adapter.integrations.base import IntegrationBase, ToolRunner, model_visible_error
from agent_adapter.runtime.ids import new_id
from agent_adapter.runtime.sync import run_sync

logger = logging.getLogger(__name__)

TARGET = "haystack"
_STATE_PARAM = "_aa_state"  # Haystack injects the Agent's live `State` into this parameter
_SPEC_ATTR = "_aa_spec"  # the ToolSpec behind an exported Tool
_NOT_SERIALISABLE = "the agent_adapter confirmation hook holds a live approval port"


@dataclass
class _Scope:
    """One Agent run (one `State`): its run id, each hook's tokens, call ids already bound."""

    run_id: str = field(default_factory=lambda: new_id("run_"))
    tokens: dict[int, TokenTable] = field(default_factory=dict)  # by id() of the hook
    claimed: set[str] = field(default_factory=set)

    def tokens_of(self, hook: object) -> TokenTable:
        """The token table of one hook on this run (created on first use)."""
        with _LOCK:
            return self.tokens.setdefault(id(hook), TokenTable())


class _NotSerialisable:
    """Refuses Haystack's `to_dict` / `from_dict`: the object holds a live approval port."""

    def to_dict(self) -> dict[str, Any]:
        raise IntegrationError(_NOT_SERIALISABLE, integration=TARGET)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        raise IntegrationError(_NOT_SERIALISABLE, integration=TARGET)


# Kept beside the State, never in it: State data reaches outputs and snapshots.
_SCOPES: weakref.WeakKeyDictionary[Any, _Scope] = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()  # sync tools run on worker threads


class HaystackIntegration(IntegrationBase):
    """Exports Haystack `Tool`s; `confirmation_hook` asks an approval port before calls."""

    name = TARGET
    requires = ("haystack",)
    extra = "haystack"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `Tool` running `spec` through the executor."""
        return haystack_tool(spec, executor=executor)

    def bind_context(self, native: Any) -> ToolContext:
        """Context from the Agent `State`: run id, call id and the hook's token, if any.

        `native` is `(state, spec, arguments)`; the call is matched against the pending tool
        calls of the state's last message (one claim per call id).
        """
        state, spec, arguments = native
        scope = _scope(state) if state is not None else None
        if scope is None:
            return ToolContext(extras={TARGET: state})
        call_id, approval = _claim(state, scope, spec, arguments)
        return ToolContext(
            run_id=scope.run_id, call_id=call_id, extras={TARGET: state}, approval=approval
        )

    def bind_approval(self, port: ApprovalPort) -> Any:
        """A `before_tool` hook asking `port` about this module's tools."""
        return _hook(port, None)


INTEGRATION = HaystackIntegration()


def haystack_tool(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """A Haystack `Tool` for `spec` with both `function` (via `run_sync`) and `async_function`."""
    INTEGRATION.require()
    from haystack.components.agents.state import State
    from haystack.tools.tool import Tool

    runner = lazy_runner(executor)

    async def call(**arguments: Any) -> Any:
        state = arguments.pop(_STATE_PARAM, None)
        ctx = INTEGRATION.bind_context((state, spec, arguments))
        return await run_bound(runner, spec, arguments, ctx)

    def call_sync(**arguments: Any) -> Any:
        return run_sync(lambda: call(**arguments))

    signature = _signature(State)
    call.__signature__ = signature  # type: ignore[attr-defined]
    call_sync.__signature__ = signature  # type: ignore[attr-defined]
    tool = Tool(
        name=spec.name,
        description=spec.description,
        parameters=spec.json_schema(),
        function=call_sync,
        async_function=call,
    )
    setattr(tool, _SPEC_ATTR, spec)
    return tool


def haystack_tools(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """Haystack `Tool`s for every registry tool exported to this target."""
    return INTEGRATION.export_tools(registry, executor=executor)


def confirmation_hook(port: ApprovalPort, registry: Registry) -> Any:
    """A `before_tool` Agent hook: asks `port` about `registry` tools whose rule says so.

    Register it as `Agent(..., hooks={"before_tool": [hook]})`. It runs Haystack's
    `ConfirmationHook` with a port-backed strategy: a denial removes the call and tells the
    model why. An approval becomes a one-use token for that pending call (run, call id,
    tool, arguments), taken when it runs, so the executor does not ask again. Each hook keeps
    its own tokens and drops those of an earlier step. Edits are not supported: an edit is
    a denial.
    """
    return _hook(port, registry)


def _signature(state_type: Any) -> inspect.Signature:
    """`(_aa_state: State | None = None, **arguments)`."""
    return inspect.Signature(
        [
            inspect.Parameter(
                _STATE_PARAM, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=state_type
            ),
            inspect.Parameter("arguments", inspect.Parameter.VAR_KEYWORD),
        ]
    )


def _scope(state: Any) -> _Scope | None:
    """The run scope kept for `state`, created on first use; None when it cannot be kept."""
    with _LOCK:
        try:
            scope = _SCOPES.get(state)
            if scope is None:
                scope = _SCOPES[state] = _Scope()
        except TypeError:  # not weak-referenceable or not hashable
            return None
    return scope


def _pending(state: Any, name: str, arguments: Mapping[str, Any]) -> tuple[list[str], set[str]]:
    """Ids of the last message's pending calls of `name` with these arguments, and all ids."""
    messages = (getattr(state, "data", None) or {}).get("messages") or [None]
    calls = getattr(messages[-1], "tool_calls", None) or []
    every = {call.id for call in calls if call.id}
    matching = [
        call.id
        for call in calls
        if call.id and call.tool_name == name and call.arguments == arguments
    ]
    return matching, every


def _claim(
    state: Any, scope: _Scope, spec: ToolSpec, arguments: Mapping[str, Any]
) -> tuple[str, ApprovalDecision | None]:
    """Bind this invocation to a pending call id, taking the token issued for that call."""
    matching, every = _pending(state, spec.name, arguments)
    dumped = dump_arguments(spec, arguments)
    with _LOCK:
        scope.claimed &= every  # only this step's ids matter: the set stays bounded
        free = [call_id for call_id in matching if call_id not in scope.claimed]
        chosen, approval = (free[0] if free else ""), None
        tables = list(scope.tokens.values())
        for call_id in free:
            if dumped is None:
                break
            token = next(
                (
                    found
                    for table in tables
                    if (
                        found := table.take(spec.name, dumped, call_id=call_id, run_id=scope.run_id)
                    )
                ),
                None,
            )
            if token is not None:  # the first pending call an approval was issued for
                chosen, approval = call_id, token.decision
                break
        if chosen:
            scope.claimed.add(chosen)
    return chosen, approval


def _hook(port: ApprovalPort, registry: Registry | None) -> Any:
    """The confirmation hook behind both factories."""
    from haystack.hooks.human_in_the_loop import ConfirmationHook

    class AdapterConfirmationHook(_NotSerialisable):
        """Runs Haystack's `ConfirmationHook` with a fresh port-backed strategy per step."""

        allowed_hook_points = ("before_tool",)

        def run(self, state: Any) -> None:
            run_sync(lambda: self.run_async(state))

        async def run_async(self, state: Any) -> None:
            # An untrackable state gets a throwaway scope: its calls are asked again (fail closed).
            scope = _scope(state) or _Scope()
            tokens = scope.tokens_of(self)
            tokens.clear()  # this hook's tokens of an earlier step that never ran
            strategy = _PortStrategy(port, _specs(registry, state), scope.run_id, tokens)
            await ConfirmationHook({"*": strategy}).run_async(state)

    return AdapterConfirmationHook()


def _specs(registry: Registry | None, state: Any) -> dict[str, ToolSpec]:
    """Name -> spec: the registry's, or else those behind the Agent's exported tools."""
    if registry is not None:
        return {spec.name: spec for spec in registry}
    tools = state.data.get("tools") or []
    specs = (getattr(tool, _SPEC_ATTR, None) for tool in tools)
    return {spec.name: spec for spec in specs if isinstance(spec, ToolSpec)}


class _PortStrategy(_NotSerialisable):
    """A Haystack `ConfirmationStrategy` that asks an `ApprovalPort` and issues tokens."""

    def __init__(
        self, port: ApprovalPort, specs: Mapping[str, ToolSpec], run_id: str, tokens: TokenTable
    ) -> None:
        self.port = port
        self.specs = specs
        self.run_id = run_id
        self.tokens = tokens  # this hook's table on this run

    def run(self, **kwargs: Any) -> Any:
        return run_sync(lambda: self.run_async(**kwargs))

    async def run_async(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> Any:
        from haystack.hooks.human_in_the_loop import ToolExecutionDecision

        def decide(execute: bool, feedback: str | None = None) -> Any:
            return ToolExecutionDecision(
                tool_name=tool_name,
                execute=execute,
                tool_call_id=tool_call_id,
                feedback=feedback,
                final_tool_params=tool_params if execute else None,
            )

        spec = self.specs.get(tool_name)
        if spec is None:
            return decide(True)
        call_id = tool_call_id or ""
        run_id = self.run_id
        try:
            decision = await gate_decision(
                spec,
                tool_params,
                port=self.port,
                integration=TARGET,
                call_id=call_id,
                run_id=run_id,
            )
        except ApprovalPending as exc:
            logger.info("haystack: %s (call %s) waits for approval", spec.name, call_id)
            return decide(False, pending_message(exc))
        if decision is None:
            return decide(True)
        if not decision.approved:
            denied = ApprovalDenied(spec.name, call_id=call_id, reason=decision.reason)
            return decide(False, json.dumps(model_visible_error(denied)))
        dumped = dump_arguments(spec, tool_params) or {}
        self.tokens.issue(spec.name, dumped, decision, call_id=call_id, run_id=run_id)
        return decide(True)


__all__ = [
    "INTEGRATION",
    "HaystackIntegration",
    "confirmation_hook",
    "haystack_tool",
    "haystack_tools",
]
