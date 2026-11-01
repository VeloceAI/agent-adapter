"""A tool exported with `executor=None` runs on the default executor set after the export.

Each case reuses a contract harness, so the call goes through the framework's own path.
Cases whose framework is not installed skip.
"""

import importlib

import pytest

from agent_adapter.exec.defaults import default_executor, set_default_executor
from agent_adapter.exec.executor import Executor
from tests.fakes.hooks import RecordingHook
from tests.integrations.kit import Probe, harness_session

pytestmark = pytest.mark.contract

# (contract module suffix, harness class); declaration-only targets (OpenAI, Anthropic) run
# nothing themselves and are left out.
HARNESSES = [
    ("adk", "AdkToolHarness"),
    ("agent_framework", "AgentFrameworkHarness"),
    ("agno", "AgnoHarness"),
    ("autogen", "AutoGenHarness"),
    ("claude_agent", "ClaudeAgentHarness"),
    ("crewai", "CrewAIHarness"),
    ("dspy", "DSPyHarness"),
    ("gemini", "AfcHarness"),
    ("gemini", "FunctionCallHarness"),
    ("haystack", "HaystackHarness"),
    ("langchain", "LangChainHarness"),
    ("langchain", "SyncLangChainHarness"),
    ("langgraph", "LangGraphHarness"),
    ("llamaindex", "LlamaIndexHarness"),
    ("mcp", "MCPHarness"),
    ("openai_agents", "AgentsHarness"),
    ("pydantic_ai", "PydanticAIHarness"),
    ("semantic_kernel", "SemanticKernelHarness"),
    ("smolagents", "SmolagentsHarness"),
    ("strands", "StrandsHarness"),
]


@pytest.mark.parametrize(("module", "harness_name"), HARNESSES, ids=[h for _, h in HARNESSES])
async def test_default_executor_set_after_export_runs_the_call(module, harness_name):
    try:  # importorskip there skips this case when the framework is missing
        contract = importlib.import_module(f"tests.integrations.test_{module}_contract")
    except ImportError as exc:  # installed, but too old for its contract module
        pytest.skip(f"{module}: {type(exc).__name__}: {exc}")
    probe, hook = Probe(), RecordingHook()
    previous = default_executor()
    async with harness_session(getattr(contract, harness_name)) as harness:
        native = harness.export(probe.add, None)
        set_default_executor(Executor(hooks=[hook]))
        try:
            raw = await harness.call(native, {"a": 2, "b": 3}, call_id="call_late_default")
        finally:
            set_default_executor(previous)
    assert harness.value(raw) == 5, f"{harness_name}: {raw!r}"
    assert hook.phases() == ["before_tool", "after_tool"], f"{harness_name}: {hook.phases()}"
