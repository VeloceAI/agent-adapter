"""Claude Agent SDK tool handlers: errors become isError results, control flow propagates."""

import pytest

pytest.importorskip("claude_agent_sdk")

from agent_adapter.core import control
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations import claude_agent
from tests.fakes.approval import ScriptedApproval
from tests.integrations._claude_cli import dispatch

pytestmark = pytest.mark.contract


def registry_of(*specs: ToolSpec) -> Registry:
    registry = Registry()
    registry.add_many(specs)
    return registry


def deny_all() -> ScriptedApproval:
    return ScriptedApproval([ApprovalDecision(approved=False, reason="executor")] * 3)


async def test_tool_failures_become_error_results():
    async def boom() -> str:
        """Fail."""
        raise RuntimeError("secret detail")

    bridge = claude_agent.claude_sdk_bridge(registry_of(build_spec(boom)), deny_all())
    result = await dispatch(bridge, "boom", {}, "t1")
    assert result["is_error"] and "secret detail" not in result["content"][0]["text"]


async def test_registered_control_flow_propagates_instead_of_an_error_result(monkeypatch):
    class HostInterrupt(Exception):
        pass

    monkeypatch.setattr(control, "_registered", ())  # keep the global registry clean
    control.register_control_flow(HostInterrupt)

    async def interrupted() -> str:
        """Unwind like a host interrupt."""
        raise HostInterrupt()

    registry = registry_of(build_spec(interrupted))
    with pytest.raises(HostInterrupt):
        await dispatch(
            claude_agent.claude_sdk_bridge(registry, deny_all()), "interrupted", {}, "t1"
        )
    (sdk_tool,) = claude_agent.claude_sdk_tools(registry)
    with pytest.raises(HostInterrupt):
        await sdk_tool.handler({})
