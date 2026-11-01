"""AG-UI integration object: protocol, metadata, tool descriptors and the version guard."""

import importlib.metadata

import pytest

pytest.importorskip("ag_ui")

from agent_adapter.approval import AutoApprove, DeferredApproval, MemoryApprovalStore
from agent_adapter.core._default import default_registry, reset_default_registry
from agent_adapter.core.errors import IntegrationError
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.integrations import discovery
from agent_adapter.integrations.ag_ui import INTEGRATION, AGUIBridge, _support
from agent_adapter.integrations.base import Integration

pytestmark = pytest.mark.contract

registry = Registry()


@registry.tool(targets={"ag_ui"})
def lookup(order_id: str) -> str:
    """Look an order up."""
    return order_id


@registry.tool(targets={"openai"})
def elsewhere() -> str:
    """Not exported to AG-UI."""
    return ""


def test_integration_satisfies_the_protocol():
    assert isinstance(INTEGRATION, Integration)
    assert (INTEGRATION.name, INTEGRATION.requires, INTEGRATION.extra) == (
        "ag_ui",
        ("ag_ui",),
        "ag-ui",
    )
    INTEGRATION.require()
    assert INTEGRATION.bind_approval(object()) is None


def test_export_tools_describes_the_targets_tools():
    (tool,) = INTEGRATION.export_tools(registry)
    assert tool.name == "lookup" and tool.description == "Look an order up."
    assert tool.parameters["properties"]["order_id"]["type"] == "string"


@pytest.fixture
def clean_default_registry():
    restore = reset_default_registry()
    yield default_registry
    restore()


def test_bridge_without_a_registry_uses_the_executors_default(clean_default_registry):
    executor = Executor()
    bridge = AGUIBridge(executor)
    assert executor.registry is clean_default_registry
    assert bridge._resumer.registry is executor.registry


async def test_default_registry_calls_pause_and_resume(clean_default_registry):
    ran: list[str] = []

    @clean_default_registry.tool(confirm=True)
    def wipe(path: str) -> str:
        """Wipe a path."""
        ran.append(path)
        return "wiped"

    async def agent(run):
        if not run.input.resume:
            await run.call("wipe", {"path": "/a"}, call_id="w1")
        return [r.status for r in run.resumed]

    executor = Executor(approval=DeferredApproval(MemoryApprovalStore()))
    bridge = AGUIBridge(executor, port=AutoApprove())
    first = {"threadId": "t", "runId": "r1", "messages": []}
    events = [e async for e in bridge.stream(first, agent)]
    (interrupt,) = events[-1].outcome.interrupts
    assert ran == [] and interrupt.tool_call_id == "w1"
    answer = {"interruptId": interrupt.id, "status": "resolved", "payload": {"approved": True}}
    second = {**first, "runId": "r2", "resume": [answer]}
    events = [e async for e in bridge.stream(second, agent)]
    assert ran == ["/a"] and events[-1].result == ["ok"]


def test_version_guard_refuses_another_major(monkeypatch):
    import ag_ui.core

    _support.core.cache_clear()
    monkeypatch.setattr(ag_ui.core, "PROTOCOL_VERSION", "2.0")
    try:
        with pytest.raises(IntegrationError, match="outside the supported range"):
            _support.core()
    finally:
        _support.core.cache_clear()


def test_discovery_finds_the_entry_point():
    names = {point.name for point in importlib.metadata.entry_points(group=discovery.GROUP)}
    if "ag_ui" not in names:
        pytest.skip("the ag_ui entry point is a requested pyproject.toml edit")
    assert discovery.get_integration("ag_ui") is INTEGRATION
