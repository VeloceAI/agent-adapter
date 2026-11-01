"""`attach`: name and object lookup, the error catalogue, and no approval options (§14.4)."""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, ClassVar

import pytest

import agent_adapter
from agent_adapter import AutoApprove, Executor, MissingDependency, Registry, attach
from agent_adapter.core.tool import ToolSpec
from agent_adapter.errors import IntegrationError, ToolUsageError
from agent_adapter.integrations import _attach
from agent_adapter.integrations.base import IntegrationBase, ToolRunner


class Fake(IntegrationBase):
    """Exports `(name, tool, executor)` tuples so a test can see what attach passed."""

    call_id_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(self, name: str, requires: tuple[str, ...] = (), extra: str = "") -> None:
        self.name = name  # type: ignore[misc]
        self.requires = requires  # type: ignore[misc]
        self.extra = extra  # type: ignore[misc]

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        return (self.name, spec.name, executor)


class Hooked(Fake):
    """An integration with its own `attach`, which receives the options."""

    def attach(self, native: Any, registry: Registry, **opts: Any) -> Any:
        return ("hooked", native, len(registry), opts)


class Plain:
    """Implements the Integration protocol without IntegrationBase (no export_tools)."""

    name = "plain"
    requires: tuple[str, ...] = ()
    extra = ""

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        return f"plain:{spec.name}"

    def bind_context(self, native: Any) -> Any:
        return None

    def bind_approval(self, port: Any) -> Any:
        return None


def _module(monkeypatch: pytest.MonkeyPatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _cls(module: str, name: str = "Agent") -> type:
    return type(name, (), {"__module__": module})


@pytest.fixture
def found(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    for name in ("fakegraph", "fakecore", "fakeagents"):
        _module(monkeypatch, name)
    table: dict[str, Any] = {
        "fake_graph": Fake("fake_graph", ("fakegraph", "fakecore"), "fake-graph"),
        "fake_chain": Fake("fake_chain", ("fakecore",), "fake-chain"),
        "fake_agents": Fake("fake_agents", ("fakeagents",), "fake-agents"),
        "missing": Fake("missing", ("not_installed_anywhere",), "missing-extra"),
    }
    monkeypatch.setattr(_attach, "available_integrations", lambda: dict(table))
    return table


@pytest.fixture
def registry() -> Registry:
    registry = Registry()

    @registry.tool
    def gst_rate(category: str) -> float:
        """GST rate in percent."""
        return 18.0

    return registry


def test_attach_is_the_top_level_name():
    assert agent_adapter.attach is _attach.attach
    assert "attach" in agent_adapter.__all__


@pytest.mark.parametrize("name", ["fake_agents", "Fake-Agents", " fake agents ", "FAKE.AGENTS"])
def test_attach_by_name_ignores_case_dashes_and_spaces(found, registry, name):
    assert attach(name, registry) == [("fake_agents", "gst_rate", None)]


def test_attach_by_extra_name(found, registry):
    found["fake_agents"].extra = "agents-sdk"
    assert attach("agents-sdk", registry) == [("fake_agents", "gst_rate", None)]


def test_attach_passes_the_executor_to_every_tool(found, registry):
    executor = Executor(registry)
    assert attach("fake_agents", registry, executor=executor) == [
        ("fake_agents", "gst_rate", executor)
    ]


def test_an_unknown_name_suggests_and_lists_the_integrations(found, registry):
    with pytest.raises(IntegrationError) as caught:
        attach("fake_agnts", registry)
    message = str(caught.value)
    assert "did you mean 'fake_agents'?" in message
    assert "installed: fake_agents, fake_chain, fake_graph;" in message
    assert 'needs an extra: missing (pip install "agent-adapter[missing-extra]")' in message
    assert caught.value.integration == "fake_agnts"


def test_a_known_name_without_its_extra_names_the_pip_line(found, registry):
    with pytest.raises(MissingDependency, match=r"agent-adapter\[missing-extra\]"):
        attach("missing", registry)


def test_attach_detects_the_framework_from_an_instance_class_or_module(
    found, registry, monkeypatch
):
    agent_cls = _cls("fakeagents.agent")
    expected = [("fake_agents", "gst_rate", None)]
    assert attach(agent_cls(), registry) == expected
    assert attach(agent_cls, registry) == expected
    assert attach(sys.modules["fakeagents"], registry) == expected


def test_a_user_subclass_is_detected_through_its_bases(found, registry):
    mine = type("MyAgent", (_cls("fakeagents.agent"),), {"__module__": "myapp.agents"})
    assert attach(mine(), registry) == [("fake_agents", "gst_rate", None)]


def test_the_integrations_first_import_name_wins_a_shared_module(found, registry):
    # fakecore is fake_chain's own SDK, and only fake_graph's second requirement.
    assert attach(_cls("fakecore.tools")(), registry)[0][0] == "fake_chain"
    assert attach(_cls("fakegraph.graph")(), registry)[0][0] == "fake_graph"


def test_two_integrations_owning_one_module_is_ambiguous(found, registry):
    found["twin"] = Fake("twin", ("fakeagents",), "twin")
    with pytest.raises(IntegrationError, match=r"matches \['fake_agents', 'twin'\]"):
        attach(_cls("fakeagents.agent")(), registry)


def test_an_object_from_no_framework_lists_the_integrations(found, registry):
    with pytest.raises(IntegrationError, match=r"cannot tell which framework a object") as caught:
        attach(object(), registry)
    assert "installed: fake_agents" in str(caught.value)


def test_an_integration_without_requires_matches_its_own_name(monkeypatch, registry):
    _module(monkeypatch, "rawsdk")
    table = {"rawsdk": Fake("rawsdk")}
    monkeypatch.setattr(_attach, "available_integrations", lambda: table)
    assert attach(_cls("rawsdk.client")(), registry) == [("rawsdk", "gst_rate", None)]


@pytest.mark.parametrize(
    "option",
    ["approval", "approve", "requires_approval", "auto_approve", "confirm", "needs_consent"],
)
def test_attach_never_takes_an_approval_option(found, registry, option):
    with pytest.raises(ToolUsageError, match="approval is never passed to attach") as caught:
        attach("fake_agents", registry, **{option: True})
    assert option in str(caught.value)


def test_an_approval_port_as_executor_or_registry_is_refused(found, registry):
    with pytest.raises(ToolUsageError, match="executor must be an Executor"):
        attach("fake_agents", registry, executor=AutoApprove())  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError, match="executor must be an Executor"):
        attach("fake_agents", registry, executor=True)  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError, match="registry must be a Registry"):
        attach("fake_agents", AutoApprove())  # type: ignore[arg-type]


def test_a_registry_first_says_to_pass_the_framework_first(found, registry):
    with pytest.raises(ToolUsageError, match="pass the framework first"):
        attach(registry)


def test_no_registry_uses_the_default_registry(found, registry, monkeypatch):
    from agent_adapter.core import _default

    monkeypatch.setattr(_default, "default_registry", registry)
    assert attach("fake_agents") == [("fake_agents", "gst_rate", None)]


def test_options_go_to_an_integrations_own_attach(monkeypatch, registry):
    table = {"hooked": Hooked("hooked")}
    monkeypatch.setattr(_attach, "available_integrations", lambda: table)
    target = _cls("hooked.app")()
    assert attach(target, registry, executor=None, prefix="x") == (
        "hooked",
        target,
        1,
        {"executor": None, "prefix": "x"},
    )
    assert attach("hooked", registry)[1] is None  # by name: no native object


def test_options_without_an_own_attach_are_refused(found, registry):
    with pytest.raises(ToolUsageError, match=r"takes no attach options \['prefix'\]"):
        attach("fake_agents", registry, prefix="x")


def test_a_protocol_only_integration_exports_tool_by_tool(monkeypatch, registry):
    monkeypatch.setattr(_attach, "available_integrations", lambda: {"plain": Plain()})
    assert attach("plain", registry) == ["plain:gst_rate"]


def test_an_empty_registry_warns_and_returns_nothing(found, caplog):
    with caplog.at_level(logging.WARNING, logger=_attach.__name__):
        assert attach("fake_agents", Registry()) == []
    assert "no tools in the registry are exported to 'fake_agents'" in caplog.text


def test_tools_not_exported_to_the_target_are_left_out(found):
    registry = Registry()

    @registry.tool(targets=["fake_chain"])
    def only_chain() -> str:
        """Only for fake_chain."""
        return "x"

    assert attach("fake_agents", registry) == []
    assert attach("fake_chain", registry) == [("fake_chain", "only_chain", None)]


def test_real_discovery_attaches_to_the_dependency_free_renderers(registry):
    (chat,) = attach("openai", registry)
    assert chat["function"]["name"] == "gst_rate"
    (block,) = attach("anthropic", registry)
    assert block["name"] == "gst_rate"


def test_a_real_openai_agents_agent_gets_function_tools(registry):
    agents = pytest.importorskip("agents")
    (tool,) = attach(agents.Agent(name="helper"), registry)
    assert isinstance(tool, agents.FunctionTool)
    assert tool.name == "gst_rate"


def test_a_real_langgraph_graph_gets_tool_nodes(registry):
    graph = pytest.importorskip("langgraph.graph")
    (node,) = attach(graph.StateGraph(dict), registry)
    assert type(node).__name__ == "ToolNode"
