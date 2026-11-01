import importlib

import pytest

import agent_adapter
from agent_adapter import _version

# Top-level name -> the public subpackage that is its home too; both give the same object.
SHARED_HOMES = {
    "agent_adapter.exec": (
        "Executor",
        "RunLimits",
        "Step",
        "run_tool_loop",
        "set_default_executor",
    ),
    "agent_adapter.runtime": ("run_sync",),
    "agent_adapter.approval": (
        "AutoApprove",
        "AutoDeny",
        "CallbackApproval",
        "DeferredApproval",
        "ApprovalRequest",
        "ApprovalDecision",
        "Decision",
        "MemoryApprovalStore",
        "SQLiteApprovalStore",
        "RedisApprovalStore",
    ),
    "agent_adapter.hooks": ("Hook", "LoggingHook", "GuardrailHook", "BudgetHook", "RateLimitHook"),
    "agent_adapter.cost": ("Ledger", "Prices"),
    "agent_adapter.cache": ("MemoryCache",),
    "agent_adapter.flow": ("Flow", "FlowRunner"),
    "agent_adapter.errors": (
        "AdapterError",
        "ToolError",
        "ApprovalDenied",
        "ApprovalPending",
        "MissingDependency",
    ),
}

# WP01 names and where 1.0 code imports each from.
WP01_HOMES = {
    "agent_adapter": (
        "ToolCall",
        "ToolResult",
        "Event",
        "Usage",
        "LedgerEntry",
        "ApprovalRequest",
        "ApprovalDecision",
        "Decision",
        "Message",
        "ToolDef",
        "LLMRequest",
        "LLMResponse",
        "spec_from_schema",
        "run_sync",
        "Hook",
    ),
    "agent_adapter.errors": ("ErrorInfo",),
    "agent_adapter.hooks": ("ShortCircuit",),
}


@pytest.mark.parametrize(
    ("home", "name"), [(home, name) for home, names in SHARED_HOMES.items() for name in names]
)
def test_top_level_name_is_the_same_object_as_in_its_home(home, name):
    module = importlib.import_module(home)
    assert name in agent_adapter.__all__
    assert name in module.__all__
    assert getattr(agent_adapter, name) is getattr(module, name)


@pytest.mark.parametrize(
    ("home", "name"), [(home, name) for home, names in WP01_HOMES.items() for name in names]
)
def test_wp01_names_have_a_public_home(home, name):
    module = importlib.import_module(home)
    assert name in module.__all__
    assert getattr(module, name) is not None


def test_version_comes_from_the_leaf_module():
    assert agent_adapter.__version__ == _version.__version__
