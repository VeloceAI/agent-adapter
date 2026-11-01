"""Agno: each ToolSpec as a `Function`; approval through `requires_confirmation`.

Agno's confirmations are never taken as an approval. `resolve_requirements` asks the
approval port and issues a one-use token per approved call; without a matching token the
executor asks its own port when the call runs.
"""

from __future__ import annotations

from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolDefinitionError
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations._default_runner import lazy_runner
from agent_adapter.integrations._native_gate import (
    TokenTable,
    check_framework,
    decide_calls,
    json_text,
    run_bound,
    take_approval,
)
from agent_adapter.integrations.agno_compress import agno_compressed_model  # registers the format
from agent_adapter.integrations.base import IntegrationBase, ToolRunner

TARGET = "agno"
_SUPPORTED = ((3, 1), (4, 0))  # the `agno` extra's range
_CALL_PARAM = "fc"  # Agno passes the `FunctionCall` to an entrypoint parameter of this name
# One-use approvals `resolve_requirements` issues, bound to run, call id, tool and arguments.
_TOKENS = TokenTable()


class AgnoIntegration(IntegrationBase):
    """Exports `Function`s; an Agno agent pauses `confirm` tools until they are resolved."""

    name = TARGET
    requires = ("agno",)
    extra = "agno"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """One `Function` running `spec` through the executor."""
        return agno_function(spec, executor=executor)

    def bind_context(self, native: Any) -> ToolContext:
        """Ids from a `FunctionCall`: call id, and the run and session of an agent run.

        Never an approval: Agno's confirmation only adds a gate in front of the executor's.
        """
        call_id = getattr(native, "call_id", None)
        run = getattr(getattr(native, "function", None), "_run_context", None)
        run_id, session_id = getattr(run, "run_id", None), getattr(run, "session_id", None)
        return ToolContext(
            run_id=run_id if isinstance(run_id, str) else "",
            call_id=call_id if isinstance(call_id, str) else "",
            thread_id=session_id if isinstance(session_id, str) else "",
            extras={TARGET: native},
        )


INTEGRATION = AgnoIntegration()


def agno_function(spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
    """An Agno `Function` for `spec`; any `confirm` rule sets `requires_confirmation`.

    Results reach Agno as JSON text (Agno shows `str(result)` to the model).

    Raises:
        ToolDefinitionError: the tool has an argument named `fc` (Agno reserves it).
        IntegrationError: the installed Agno is outside the extra's range or lacks what the
            bridge reads (`Function._run_context`, `FunctionCall.call_id`).
    """
    INTEGRATION.require()
    _check_agno()
    if _CALL_PARAM in spec.json_schema().get("properties", {}):
        raise ToolDefinitionError(f"tool {spec.name!r}: Agno reserves the argument name 'fc'")
    from agno.tools.function import Function

    runner = lazy_runner(executor)

    async def entrypoint(fc: Any, **arguments: Any) -> Any:
        ctx = INTEGRATION.bind_context(fc)
        ctx.approval = take_approval(_TOKENS, spec, arguments, ctx)
        return json_text(await run_bound(runner, spec, arguments, ctx))

    return Function(
        name=spec.name,
        description=spec.description,
        parameters=spec.json_schema(),
        entrypoint=entrypoint,
        skip_entrypoint_processing=True,
        requires_confirmation=spec.confirm is not False,
    )


def agno_functions(registry: Registry, *, executor: ToolRunner | None = None) -> list[Any]:
    """Agno `Function`s for every registry tool exported to this target."""
    return INTEGRATION.export_tools(registry, executor=executor)


async def resolve_requirements(output: Any, port: ApprovalPort, registry: Registry) -> None:
    """Confirm or reject each confirmation `output` paused on, from `port`.

    Then continue with `agent.acontinue_run(run_response=output)`. The run id is
    `output.run_id`, the one the executor checks when the confirmed call runs. Registry
    tools are asked with validated arguments (confirmed unasked when their `confirm` rule
    says no); other tools with their raw arguments.

    Raises:
        ApprovalPending: a decision is still open (every call was asked first; nothing set).
    """
    run_id = str(getattr(output, "run_id", "") or "")
    pending = [
        r for r in getattr(output, "active_requirements", None) or () if r.needs_confirmation
    ]
    calls = [
        (
            str(r.tool_execution.tool_name or ""),
            dict(r.tool_execution.tool_args or {}),
            str(r.tool_execution.tool_call_id or ""),
        )
        for r in pending
    ]
    decisions = await decide_calls(
        port, calls, registry, run_id=run_id, integration=TARGET, tokens=_TOKENS
    )
    for requirement, decision in zip(pending, decisions, strict=True):
        if decision.approved:
            requirement.confirm()
        else:
            requirement.reject(decision.reason or "denied")


def _check_agno() -> None:
    import agno
    from agno.tools.function import Function, FunctionCall

    check_framework(
        TARGET,
        str(getattr(agno, "__version__", "0")),
        _SUPPORTED,
        {
            "Function._run_context": "_run_context"
            in getattr(Function, "__private_attributes__", {}),
            "FunctionCall.call_id": "call_id" in getattr(FunctionCall, "model_fields", {}),
        },
    )


__all__ = [
    "INTEGRATION",
    "AgnoIntegration",
    "agno_compressed_model",
    "agno_function",
    "agno_functions",
    "resolve_requirements",
]
