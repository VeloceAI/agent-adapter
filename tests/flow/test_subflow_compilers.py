"""The LangGraph and ADK compilers refuse SubFlow nodes with a typed error, before building."""

import pytest

from agent_adapter.core.errors import FlowCompileError
from agent_adapter.exec import Executor
from agent_adapter.integrations.adk import compile as adk_compile
from agent_adapter.integrations.langgraph import compile as langgraph_compile
from agent_adapter.integrations.langgraph._integration import INTEGRATION
from tests.flow.helpers import registry
from tests.flow.subflow_helpers import child_flow, parent_flow


def test_langgraph_refuses_a_subflow_node(monkeypatch):
    monkeypatch.setattr(INTEGRATION, "require", lambda: None)  # no langgraph needed to refuse
    with pytest.raises(FlowCompileError) as caught:
        langgraph_compile.compile_flow(parent_flow(child_flow()), executor=Executor(registry))
    assert caught.value.target == "langgraph"
    assert caught.value.problems == [
        "node 'sub' is a SubFlow, which the LangGraph compiler does not support yet; "
        "run this flow with FlowRunner"
    ]


def test_adk_refuses_a_subflow_node(monkeypatch):
    monkeypatch.setattr(adk_compile, "require_adk", lambda: None)  # no ADK needed to refuse
    with pytest.raises(FlowCompileError) as caught:
        adk_compile.compile_flow(parent_flow(child_flow()), executor=Executor(registry))
    assert caught.value.target == "adk"
    assert caught.value.problems == [
        "node 'sub' is a SubFlow, which the ADK compiler does not support yet; "
        "run this flow with FlowRunner"
    ]
