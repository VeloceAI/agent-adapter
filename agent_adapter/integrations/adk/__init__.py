"""Google ADK 2.x: tools as `BaseTool`s with a confirmation bridge, flows as a `Workflow`."""

from agent_adapter.integrations.adk._run import run_workflow
from agent_adapter.integrations.adk._support import INTEGRATION, TARGET, AdkIntegration
from agent_adapter.integrations.adk._tokens import resume_message
from agent_adapter.integrations.adk.compile import compile_flow
from agent_adapter.integrations.adk.tools import (
    adk_tool,
    adk_tools,
    bind_context,
    resolve_confirmations,
)

__all__ = [
    "INTEGRATION",
    "TARGET",
    "AdkIntegration",
    "adk_tool",
    "adk_tools",
    "bind_context",
    "compile_flow",
    "resolve_confirmations",
    "resume_message",
    "run_workflow",
]
