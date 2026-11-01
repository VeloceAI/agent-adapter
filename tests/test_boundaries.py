"""Import rules R1-R10 (contract §2.3), checked on the AST so nothing needs to be imported."""

import ast

import pytest

from tests.boundary_rules import (
    FORBIDDEN_ROOTS,
    PACKAGE,
    RULES,
    SUBPACKAGES,
    VERSION,
    all_imports,
    base_third_party,
    dag_offenders,
    dynamic_imports,
    forbidden_roots,
    module_key,
    non_odm_third_party,
    odm_offenders,
    odm_roots,
    private_internals,
    relative_imports,
    roots,
    sources,
    target_key,
    top_level_imports,
)

# --- the rules over the real package ---------------------------------------------------------


def test_version_module_is_a_leaf():
    tree = ast.parse((PACKAGE / f"{VERSION}.py").read_text(encoding="utf-8"))
    assert all_imports(tree) == []


def test_package_layout_is_known():
    rels = [rel for rel, _ in sources()]
    assert "core/errors.py" in rels and "integrations/openai.py" in rels
    unknown = {rel.split("/")[0] for rel in rels if "/" in rel} - SUBPACKAGES
    assert unknown == set()


def test_package_uses_absolute_imports_only():
    offenders = [f"{rel}: {t}" for rel, src in sources() for t in relative_imports(src)]
    assert offenders == []


@pytest.mark.parametrize("rule", list(RULES))
def test_package_follows_rule(rule):
    detector = RULES[rule]
    offenders = [f"{rel}: {item}" for rel, src in sources() for item in detector(rel, src)]
    assert offenders == []


# --- detector self-tests ---------------------------------------------------------------------


def test_relative_import_detector_flags_every_level():
    source = "from . import x\nfrom ..core import y\nfrom .errors import Z\nimport os\n"
    assert relative_imports(source) == [".", "..core", ".errors"]


def test_module_and_target_keys():
    assert module_key("core/errors.py") == "base"
    assert module_key("core/tool.py") == "core"
    assert module_key("core/__init__.py") == "core"
    assert module_key("cache/providers/openai.py") == "cache"
    assert module_key("__init__.py") is None
    assert target_key("agent_adapter.core.context") == "base"
    assert target_key("agent_adapter.core") == "core"
    assert target_key("agent_adapter.core.registry") == "core"
    assert target_key("agent_adapter.exec.executor") == "exec"
    assert target_key("agent_adapter.ToolSpec") == "root"
    assert target_key("agent_adapter") == "root"
    assert target_key("agent_adapter._version") == "_version"
    assert module_key("_version.py") == "_version"


def test_dag_detector_checks_each_imported_name():
    def schema(source: str) -> list[str]:
        return dag_offenders("schema/models.py", source)

    assert schema("from agent_adapter.core import errors, tool") == ["agent_adapter.core.tool"]
    assert schema("from agent_adapter.core import errors, context, types") == []
    assert schema("from agent_adapter.core.errors import SchemaNotPortable") == []
    assert schema("from agent_adapter.schema.portable import check_portable") == []
    assert schema("import agent_adapter.core.registry") == ["agent_adapter.core.registry"]
    assert schema("from agent_adapter import ToolSpec") == ["agent_adapter.ToolSpec"]
    assert schema("from agent_adapter.core.errors_extra import X") == [
        "agent_adapter.core.errors_extra"
    ]
    assert schema("def f():\n    from agent_adapter.runtime import ids\n") == [
        "agent_adapter.runtime.ids"
    ]


def test_dag_detector_follows_the_table():
    assert dag_offenders("core/types.py", "from agent_adapter.core import errors") == []
    assert dag_offenders("core/types.py", "from agent_adapter.core import tool") == [
        "agent_adapter.core.tool"
    ]
    assert dag_offenders("runtime/sync.py", "from agent_adapter.core.tool import T") == [
        "agent_adapter.core.tool"
    ]
    assert dag_offenders("hooks/base.py", "from agent_adapter.core.tool import ToolSpec") == []
    assert dag_offenders("cache/keys.py", "from agent_adapter.tracing import events") == [
        "agent_adapter.tracing.events"
    ]
    assert dag_offenders("exec/executor.py", "from agent_adapter.flow import Flow") == [
        "agent_adapter.flow.Flow"
    ]
    assert dag_offenders("server/service.py", "from agent_adapter.optimiser import x") == [
        "agent_adapter.optimiser.x"
    ]
    assert (
        dag_offenders("integrations/mcp/server.py", "from agent_adapter import __version__") == []
    )
    assert dag_offenders("optimiser/run.py", "import agent_adapter.integrations.mcp") == [
        "agent_adapter.integrations.mcp"
    ]
    assert dag_offenders("__init__.py", "from agent_adapter.server import x") == []


def test_every_package_may_import_the_version_and_cost_edges():
    source = "from agent_adapter._version import __version__"
    for rel in ("core/errors.py", "schema/models.py", "runtime/ids.py", "integrations/mcp/a.py"):
        assert dag_offenders(rel, source) == []
    assert dag_offenders("_version.py", "import agent_adapter._version") == [
        "agent_adapter._version"
    ]
    assert dag_offenders("cache/llm_cache.py", "from agent_adapter.cost import prices") == []
    assert dag_offenders("models/instrument.py", "from agent_adapter.cost import ledger") == []
    assert dag_offenders("tracing/sinks.py", "from agent_adapter.cost import ledger") == [
        "agent_adapter.cost.ledger"
    ]


def test_dag_detector_limits_schema_imports_in_core():
    source = "from agent_adapter.schema.portable import check_portable"
    assert dag_offenders("core/registry.py", source) == []
    assert dag_offenders("core/imported.py", source) == []
    assert dag_offenders("core/tool.py", source) == ["agent_adapter.schema.portable"]


def test_base_third_party_detector():
    source = "import pydantic\nfrom typing_extensions import Self\nimport json\nimport httpx\n"
    assert base_third_party("core/types.py", source) == ["httpx"]
    assert base_third_party("core/tool.py", source) == []


def test_lazy_import_detector_sees_top_level_framework_imports():
    source = (
        "import openai\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import anthropic\n"
        "if typing.TYPE_CHECKING:\n    import mcp\n"
        "try:\n    import langgraph\nexcept ImportError:\n    pass\n"
        "class A:\n    import yaml\n"
        "def f():\n    import google\n"
    )
    found = [roots(n)[0] for n in top_level_imports(ast.parse(source).body)]
    assert found == ["openai", "typing", "langgraph", "yaml"]


def test_lazy_import_detector_treats_every_module_level_block_as_eager():
    source = (
        "for _ in ():\n    import a1\nelse:\n    import a2\n"
        "while False:\n    import b1\nelse:\n    import b2\n"
        "async with x:\n    import c1\n"
        "try:\n    import d1\nexcept* ValueError:\n    import d2\n"
        "match x:\n    case 1:\n        import e1\n    case _:\n        import e2\n"
        "async def f():\n    async for _ in y:\n        import lazy\n"
    )
    found = [roots(n)[0] for n in top_level_imports(ast.parse(source).body)]
    assert found == ["a1", "a2", "b1", "b2", "c1", "d1", "d2", "e1", "e2"]


# Each error module may import only the error modules listed for it; error_base is the leaf.
ERROR_MODULE_IMPORTS = {
    "error_base": set(),
    "model_errors": {"agent_adapter.core.error_base"},
    "errors": {"agent_adapter.core.error_base", "agent_adapter.core.model_errors"},
}


@pytest.mark.parametrize("name", sorted(ERROR_MODULE_IMPORTS))
def test_error_modules_import_only_each_other(name):
    tree = ast.parse((PACKAGE / "core" / f"{name}.py").read_text(encoding="utf-8"))
    package_imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in all_imports(tree)
        for alias in node.names
        if "agent_adapter" in roots(node)
    }
    assert package_imports <= ERROR_MODULE_IMPORTS[name]


def test_odm_detector():
    lazy = "def f():\n    import redis.asyncio\n"
    assert odm_offenders("cache/redis.py", lazy) == []
    assert odm_offenders("cache/redis.py", "import redis\n") == ["eager redis"]
    assert odm_offenders("cache/redis.py", "def f():\n    import openai\n") == ["lazy openai"]
    assert odm_offenders("integrations/mcp/server.py", "def f():\n    import mcp, anyio\n") == []
    assert odm_offenders("integrations/gemini.py", "def f():\n    import mcp\n") == ["lazy mcp"]
    assert odm_offenders("integrations/gemini.py", "import pydantic\nimport json\n") == []
    assert odm_offenders("cache/keys.py", "import redis\n") == []  # not an ODM: R7 applies


def test_odm_roots_lookup():
    assert odm_roots("models/openai.py") == {"openai", "httpx2"}
    assert odm_roots("integrations/langgraph/compile.py") == {
        "langgraph",
        "langchain_core",
        "httpx",
        "httpx2",
        "anyio",
    }
    assert odm_roots("integrations/base.py") is None
    assert odm_roots("integrations/discovery.py") is None
    assert odm_roots("models/fallback.py") is None


def test_non_odm_third_party_detector():
    source = "import pydantic\ndef f():\n    import numpy\n"
    assert non_odm_third_party("exec/plan.py", source) == ["numpy"]
    assert non_odm_third_party("models/openai.py", source) == []


def test_dynamic_import_detector():
    source = "import importlib\nfrom importlib import metadata\nx = __import__('os')\n"
    assert dynamic_imports("core/tool.py", source) == ["importlib", "importlib", "__import__"]
    assert dynamic_imports("core/tool.py", "y = builtins.__import__\n") == ["__import__"]
    assert dynamic_imports("integrations/discovery.py", source) == []
    assert dynamic_imports("optimiser/__main__.py", source) == []


def test_forbidden_root_detector():
    source = "def f():\n    import langchain_core\n    from app import x\n"
    assert forbidden_roots("hooks/base.py", source) == ["langchain_core", "app"]
    assert forbidden_roots("integrations/langchain.py", source) == []
    assert {"openai", "httpx", "httpx2", "redis", "a2wsgi", "yaml", "app"} <= FORBIDDEN_ROOTS


def test_seam_detector_flags_private_flow_and_exec_internals():
    source = (
        "from agent_adapter.flow._nodes import run_node\n"
        "from agent_adapter.exec.parallel import _gather_unwrapped\n"
        "from agent_adapter.flow.visit import run_node\n"
        "from agent_adapter.exec.integration import make_answer\n"
        "from agent_adapter.core._attempt import ATTEMPT\n"
    )
    expected = ["agent_adapter.flow._nodes", "agent_adapter.exec.parallel._gather_unwrapped"]
    assert private_internals("integrations/adk/_steps.py", source) == expected
    assert private_internals("server/service.py", source) == expected
    assert private_internals("flow/visit.py", source) == []
    assert private_internals("exec/integration.py", source) == []
