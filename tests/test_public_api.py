"""The 1.0 top level: a curated, lazy front door; every other name lives in its home."""

import importlib
import inspect
import sys

import pytest

import agent_adapter
from agent_adapter import errors
from agent_adapter.core import errors as core_errors
from agent_adapter.core.types import ErrorInfo
from tests.work import ROOT, run_child


def _error_classes() -> set[str]:
    return {
        name
        for name, value in vars(core_errors).items()
        if isinstance(value, type) and issubclass(value, core_errors.AdapterError)
    }


# The top level before 1.0; a name missing from today's __all__ has moved to a home.
PRE_1_0_TOP_LEVEL = frozenset(
    {
        *(_error_classes() - {"ApprovalStoreCorrupt"}),  # added after 1.0 split
        "ApprovalDecision",
        "ApprovalRequest",
        "Decision",
        "ErrorInfo",
        "Event",
        "Executor",
        "Hook",
        "LLMRequest",
        "LLMResponse",
        "LedgerEntry",
        "Message",
        "Registry",
        "RunLimits",
        "ShortCircuit",
        "Step",
        "ToolCall",
        "ToolContext",
        "ToolDef",
        "ToolResult",
        "ToolSpec",
        "Usage",
        "__version__",
        "build_spec",
        "default_registry",
        "run_sync",
        "run_tool_loop",
        "spec_from_schema",
        "tool",
        "wrap",
    }
)

LOADED_BY_IMPORT = """
import json, sys
before = set(sys.modules)
import agent_adapter
print(json.dumps({"loaded": sorted(set(sys.modules) - before)}))
"""

LOADED_BY_AN_ERROR = """
import json, sys
import agent_adapter
agent_adapter.AdapterError, agent_adapter.ToolError, agent_adapter.MissingDependency
pydantic = "pydantic" in sys.modules
print(json.dumps({"pydantic": pydantic, "hooks": agent_adapter.hooks.__name__}))
"""

RACE_FOR_THE_DEFAULT_REGISTRY = """
import json, threading
import agent_adapter
seen, start = [], threading.Barrier(8)
def first_access():
    start.wait()
    seen.append(id(agent_adapter.default_registry))
threads = [threading.Thread(target=first_access) for _ in range(8)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print(json.dumps({"registries": len(set(seen))}))
"""


def _dropped() -> set[str]:
    return set(PRE_1_0_TOP_LEVEL) - set(agent_adapter.__all__)


def test_import_loads_only_the_package_and_its_version():
    loaded = run_child(LOADED_BY_IMPORT)["loaded"]
    assert loaded == ["agent_adapter", "agent_adapter._version"]  # no pydantic, no typing


def test_top_level_errors_skip_pydantic_and_subpackages_load_on_access():
    assert run_child(LOADED_BY_AN_ERROR) == {"pydantic": False, "hooks": "agent_adapter.hooks"}


def test_threads_racing_for_the_default_registry_share_one():
    assert run_child(RACE_FOR_THE_DEFAULT_REGISTRY) == {"registries": 1}


@pytest.mark.parametrize("name", agent_adapter.__all__)
def test_every_top_level_name_resolves_and_is_listed_by_dir(name):
    assert getattr(agent_adapter, name) is not None
    assert name in dir(agent_adapter)


def test_dir_hides_the_loader_internals():
    assert "_LAZY" not in dir(agent_adapter)
    assert "TYPE_CHECKING" not in dir(agent_adapter)


def test_the_lazy_table_covers_exactly_the_top_level():
    assert set(agent_adapter._LAZY) | {"__version__"} == set(agent_adapter.__all__)
    assert set(agent_adapter._MOVED).isdisjoint(agent_adapter.__all__)


def test_tool_wrap_and_default_registry_are_stable_single_objects():
    registry = agent_adapter.default_registry
    assert agent_adapter.default_registry is registry
    assert agent_adapter.tool is agent_adapter.tool
    assert agent_adapter.wrap is agent_adapter.wrap
    assert agent_adapter.tool.__self__ is registry
    assert agent_adapter.wrap.__self__ is registry
    assert isinstance(registry, agent_adapter.Registry)


def test_every_core_error_is_in_the_errors_module():
    assert _error_classes() | {"ErrorInfo"} == set(errors.__all__)
    for name in _error_classes():
        assert getattr(errors, name) is getattr(core_errors, name)
    assert errors.ErrorInfo is ErrorInfo


def test_every_dropped_name_is_importable_from_its_home():
    dropped = _dropped()
    assert dropped == set(agent_adapter._MOVED)
    for name in dropped:
        home = importlib.import_module(agent_adapter._MOVED[name])
        assert name in home.__all__
        assert inspect.isclass(getattr(home, name))


def test_short_circuit_lives_in_hooks():
    from agent_adapter.hooks import ShortCircuit
    from agent_adapter.hooks.base import ShortCircuit as defined

    assert ShortCircuit is defined


# Ports and types that host code implements or annotates against, and their one home.
PORT_HOMES = {
    "agent_adapter.models": ("LLMPort", "PromptCachePlugin", "ModelCaps", "CapabilityLookup"),
    "agent_adapter.cache": ("CachePort",),
    "agent_adapter.cost": ("LedgerPort", "LedgerSink"),
    "agent_adapter.tracing": ("EventSink",),
    "agent_adapter.approval": ("ApprovalPort", "ApprovalStore"),
    "agent_adapter.embeddings": ("EmbeddingPort",),
    "agent_adapter.server": ("is_invoker", "IdempotencyStore", "EventStream"),
}


def _core_protocols() -> set[str]:
    from agent_adapter.core import ports

    return {
        name
        for name, value in vars(ports).items()
        if inspect.isclass(value)
        and getattr(value, "_is_protocol", False)
        and value.__module__ == ports.__name__
    }


@pytest.mark.parametrize(
    ("home", "name"), [(home, name) for home, names in PORT_HOMES.items() for name in names]
)
def test_every_port_is_listed_in_its_home(home, name):
    module = importlib.import_module(home)
    assert name in module.__all__
    assert getattr(module, name) is not None


def test_every_core_port_has_a_public_home():
    homed = {name for names in PORT_HOMES.values() for name in names}
    assert _core_protocols() == {
        "ApprovalPort",
        "CachePort",
        "EventSink",
        "LLMPort",
        "LedgerPort",
        "PromptCachePlugin",
    }
    assert _core_protocols() <= homed


def test_a_homed_port_is_the_core_object():
    from agent_adapter import cache, cost, models, server, tracing
    from agent_adapter.core import ports, types
    from agent_adapter.server import auth

    assert models.LLMPort is ports.LLMPort
    assert models.PromptCachePlugin is ports.PromptCachePlugin
    assert models.ModelCaps is types.ModelCaps
    assert models.CapabilityLookup is ports.CapabilityLookup
    assert cache.CachePort is ports.CachePort
    assert cost.LedgerPort is ports.LedgerPort
    assert tracing.EventSink is ports.EventSink
    assert server.is_invoker is auth.is_invoker


def test_a_moved_attribute_names_its_new_home():
    with pytest.raises(AttributeError) as caught:
        _ = agent_adapter.ToolNotFound
    assert str(caught.value) == "ToolNotFound moved: from agent_adapter.errors import ToolNotFound"
    assert caught.value.name == "ToolNotFound"


def test_a_moved_from_import_names_its_new_home():
    with pytest.raises(ImportError, match=r"^ShortCircuit moved: from agent_adapter\.hooks import"):
        from agent_adapter import ShortCircuit  # noqa: F401


def test_without_cpython_frames_a_moved_name_falls_back_to_attribute_error(monkeypatch):
    def no_frames(depth: int = 0) -> None:
        raise ValueError("call stack is not deep enough")

    monkeypatch.setattr(sys, "_getframe", no_frames)
    assert agent_adapter._in_from_import() is False


def test_hasattr_and_getattr_defaults_still_work_for_moved_names():
    assert not hasattr(agent_adapter, "ToolNotFound")
    assert getattr(agent_adapter, "ErrorInfo", None) is None


def test_an_unknown_attribute_points_at_all():
    with pytest.raises(AttributeError, match=r"has no attribute 'Nope'; see agent_adapter.__all__"):
        _ = agent_adapter.Nope
    with pytest.raises(ImportError, match="cannot import name 'Nope'"):
        from agent_adapter import Nope  # noqa: F401


def _subpackages_on_disk() -> set[str]:
    package = ROOT / "agent_adapter"
    return {p.name for p in package.iterdir() if (p / "__init__.py").is_file()}


def test_the_lazy_subpackage_table_matches_the_folders_on_disk():
    assert _subpackages_on_disk() | {"errors"} == set(agent_adapter._SUBPACKAGES)


SUBPACKAGES_BY_ATTRIBUTE = """
import importlib, json
import agent_adapter
names = sorted(agent_adapter._SUBPACKAGES)
load = importlib.import_module
same = {n: getattr(agent_adapter, n) is load("agent_adapter." + n) for n in names}
print(json.dumps(same))
"""


def test_every_subpackage_loads_on_attribute_access():
    names = sorted(_subpackages_on_disk() | {"errors"})
    assert run_child(SUBPACKAGES_BY_ATTRIBUTE) == dict.fromkeys(names, True)
    assert agent_adapter.hooks is importlib.import_module("agent_adapter.hooks")
    assert agent_adapter.errors is errors


TYPED_SNIPPET = '''
import agent_adapter
from agent_adapter import Registry, ToolSpec, default_registry, tool
from agent_adapter.errors import ToolNotFound

registry: Registry = default_registry
error: type[Exception] = ToolNotFound


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


total: int = add(1, 2)
spec: ToolSpec = default_registry.get("add")
wrong: int = agent_adapter.Registry()
agent_adapter.ToolNotFound
'''


def test_strict_mypy_sees_the_real_types(tmp_path):
    api = pytest.importorskip("mypy.api")
    snippet = tmp_path / "snippet.py"
    snippet.write_text(TYPED_SNIPPET, encoding="utf-8")
    config = tmp_path / "mypy.ini"
    config.write_text(
        "[mypy]\nstrict = True\nplugins = pydantic.mypy\nfollow_imports = silent\n"
        f"python_version = 3.11\nmypy_path = {ROOT.as_posix()}\n",
        encoding="utf-8",
    )
    stdout, _, status = api.run(
        ["--config-file", str(config), "--cache-dir", str(tmp_path / "cache"), str(snippet)]
    )
    found = [line.split(": error: ", 1)[1] for line in stdout.splitlines() if ": error: " in line]
    assert status == 1, stdout
    assert found == [  # only the two deliberate mistakes; everything else is typed
        'Incompatible types in assignment (expression has type "Registry", '
        'variable has type "int")  [assignment]',
        'Module has no attribute "ToolNotFound"  [attr-defined]',
    ]


# --- DX: names added after the 1.0 split, and misses that suggest the fix ---------------------

DX_NAMES = {"CallableModel", "FallbackModel", "acall", "call", "configure"}


def test_the_dx_names_are_top_level_and_are_their_homes_objects():
    from agent_adapter.exec import defaults
    from agent_adapter.models import CallableModel, FallbackModel

    assert set(agent_adapter.__all__) >= DX_NAMES
    assert agent_adapter.CallableModel is CallableModel
    assert agent_adapter.FallbackModel is FallbackModel
    assert agent_adapter.call is defaults.call
    assert agent_adapter.acall is defaults.acall
    assert agent_adapter.configure is defaults.configure


def test_a_typo_suggests_the_close_top_level_name():
    with pytest.raises(AttributeError) as caught:
        _ = agent_adapter.Executer
    assert str(caught.value) == (
        "module 'agent_adapter' has no attribute 'Executer'; did you mean 'Executor'? "
        "see agent_adapter.__all__ for the top-level names"
    )


def test_a_typo_in_a_from_import_suggests_the_close_name():
    with pytest.raises(ImportError) as caught:
        from agent_adapter import Executer  # noqa: F401
    assert str(caught.value) == (
        "cannot import name 'Executer' from 'agent_adapter'; did you mean 'Executor'? "
        "see agent_adapter.__all__ for the top-level names"
    )


@pytest.mark.parametrize(
    ("name", "home"),
    [
        ("tool_node", "agent_adapter.integrations.langgraph"),
        ("function_tools", "agent_adapter.integrations.openai_agents"),
        ("OpenAIModel", "agent_adapter.models"),
    ],
)
def test_a_name_from_a_subpackage_names_its_import(name, home):
    with pytest.raises(AttributeError) as caught:
        getattr(agent_adapter, name)
    assert str(caught.value) == (
        f"module 'agent_adapter' has no attribute {name!r}; it lives in {home}: "
        f"from {home} import {name}"
    )


def test_a_from_import_of_a_subpackage_name_names_its_import():
    with pytest.raises(ImportError) as caught:
        from agent_adapter import tool_node  # noqa: F401
    assert str(caught.value) == (
        "cannot import name 'tool_node' from 'agent_adapter'; it lives in "
        "agent_adapter.integrations.langgraph: "
        "from agent_adapter.integrations.langgraph import tool_node"
    )


def test_a_name_with_several_homes_lists_them():
    with pytest.raises(AttributeError) as caught:
        _ = agent_adapter.compile_flow
    assert "it lives in agent_adapter.integrations.adk, agent_adapter.integrations.langgraph" in (
        str(caught.value)
    )


def test_private_and_dunder_misses_skip_the_hint_scan():
    assert agent_adapter._hint("__wrapped__") == ""
    assert agent_adapter._hint("_private") == ""
    assert not hasattr(agent_adapter, "__wrapped__")


def test_the_hint_index_ignores_unreadable_files(monkeypatch, tmp_path):
    (tmp_path / "broken.py").write_text("__all__ = [", encoding="utf-8")
    (tmp_path / "computed.py").write_text("__all__ = list('ab')", encoding="utf-8")
    assert agent_adapter._exports(tmp_path / "broken.py") == ()
    assert agent_adapter._exports(tmp_path / "computed.py") == ()
    assert agent_adapter._exports(tmp_path / "missing.py") == ()


DX_IMPORT_STAYS_CHEAP = """
import json, sys
import agent_adapter
try:
    agent_adapter.Executer
except AttributeError:
    pass
print(json.dumps({"pydantic": "pydantic" in sys.modules}))
"""


def test_a_miss_does_not_import_the_package():
    assert run_child(DX_IMPORT_STAYS_CHEAP) == {"pydantic": False}
