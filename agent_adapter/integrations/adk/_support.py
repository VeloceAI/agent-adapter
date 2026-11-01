"""What every ADK module shares: the integration object, the version guard, ADK names."""

from __future__ import annotations

import functools
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations._native_gate import check_framework
from agent_adapter.integrations.base import IntegrationBase, ToolRunner

TARGET = "adk"
SUPPORTED = ((2, 13), (3, 0))  # the `adk` extra's range
REQUEST_INPUT = "adk_request_input"  # a workflow interrupt's function call name
REQUEST_CONFIRMATION = "adk_request_confirmation"  # an LlmAgent tool confirmation's name
INTERRUPT_PREFIX = "aa:"  # interrupts raised by a compiled flow
TOKEN_KEY = "aa_token"  # the one key a confirmation payload may hold
DRIVER = "aa_driver"  # the driver node of a compiled flow


class AdkIntegration(IntegrationBase):
    """Exports `BaseTool`s and compiles flows to ADK `Workflow`s."""

    name = TARGET
    requires = ("google.adk",)
    extra = "adk"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        """`adk_tool(spec, executor=executor)`."""
        from agent_adapter.integrations.adk.tools import adk_tool

        return adk_tool(spec, executor=executor)

    def compile_flow(self, flow: Any, *, executor: Any = None, **options: Any) -> Any:
        """`compile_flow(flow, executor=executor, **options)`."""
        from agent_adapter.integrations.adk.compile import compile_flow

        return compile_flow(flow, executor=executor, **options)

    def bind_context(self, native: Any) -> ToolContext:
        """Run (invocation id), call id, thread (session id) and a state copy of a ToolContext.

        Never an approval: ADK's confirmation only adds a gate in front of the executor's.
        """
        session = getattr(native, "session", None)
        to_dict = getattr(getattr(native, "state", None), "to_dict", None)
        snapshot = to_dict() if callable(to_dict) else {}
        return ToolContext(
            run_id=_text(getattr(native, "invocation_id", "")),
            call_id=_text(getattr(native, "function_call_id", "")),
            thread_id=_text(getattr(session, "id", "")),
            state=dict(snapshot),
            extras={TARGET: native},
        )


INTEGRATION = AdkIntegration()


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def require_adk() -> None:
    """Refuse a missing ADK, a release outside the extra's range, or one lacking what we use.

    Raises:
        MissingDependency: `google-adk` is not installed.
        IntegrationError: the version is outside the range, or a relied-on name is missing.
    """
    INTEGRATION.require()
    _check_release()


@functools.cache
def _check_release() -> None:
    from google.adk import version
    from google.adk.agents.context import Context
    from google.adk.tools.base_tool import BaseTool

    check_framework(
        TARGET,
        str(getattr(version, "__version__", "0")),
        SUPPORTED,
        {
            "BaseTool.check_require_confirmation": hasattr(BaseTool, "check_require_confirmation"),
            "Context.run_node": hasattr(Context, "run_node"),
            "Context.resume_inputs": hasattr(Context, "resume_inputs"),
        },
    )


def genai_types() -> Any:
    """The `google.genai.types` module (installed with ADK)."""
    from google.genai import types

    return types
