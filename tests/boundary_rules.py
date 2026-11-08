"""Import rules R1-R10 (contract §2.3): tables and AST detectors used by test_boundaries."""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "agent_adapter"
STDLIB = set(sys.stdlib_module_names) | {"__future__"}
ALLOWED_THIRD_PARTY = {"pydantic", "typing_extensions"}
BASE_NAMES = {
    "errors",
    "error_base",
    "model_errors",
    "types",
    "messages",
    "ports",
    "context",
    "control",
    "callables",
    "_partial_json",  # stdlib-only JSON prefix parser: models (streams) and viewer (chat drafts)
}
SUBPACKAGES = {
    "core",
    "schema",
    "runtime",
    "tracing",
    "cost",
    "cache",
    "approval",
    "hooks",
    "models",
    "exec",
    "flow",
    "integrations",
    "server",
    "optimiser",
    "evals",
    "embeddings",
    "memory",
    "retrieval",
    "chat",
    "sandbox",
    "policy",
    "audit",
    "context",
    "handlers",
    "cli",
    "viewer",
}
# `handlers` (queues, serverless), `cli` (the command), `viewer` (run recorder and local
# page) and `audit` (the audit sink) are leaf entry points: no other package imports them.
_ALL = {"base", "root"} | (SUBPACKAGES - {"handlers", "cli", "viewer", "audit"})
# R5: package keys each package may import ("root" = `agent_adapter` itself, outside the DAG).
# Every package may also import the leaf `_version` (see VERSION).
ALLOWED = {
    "_version": set(),
    "facade": {"base"},  # agent_adapter/errors.py re-exports the base error and value types
    "base": {"base"},
    "schema": {"base", "schema"},
    "core": {"base", "core", "schema"},  # schema only from SCHEMA_IMPORTERS (R4)
    "runtime": {"base", "runtime"},
    "tracing": {"base", "runtime", "tracing"},
    "cost": {"base", "runtime", "cost"},
    "cache": {"base", "runtime", "cost", "cache"},
    "embeddings": {"base", "runtime", "tracing", "cache", "embeddings"},
    "memory": {"base", "core", "runtime", "tracing", "embeddings", "memory"},
    # exec only for the shared sync loop (`retrieve_sync`).
    "retrieval": {"base", "core", "runtime", "tracing", "embeddings", "exec", "retrieval"},
    "sandbox": {"base", "core", "runtime", "tracing", "sandbox"},
    # The policy file builds hooks, approvals and limits from existing parts; nothing imports it.
    "policy": {"base", "core", "hooks", "approval", "exec", "memory", "policy"},
    # hooks: secret and PII scrubbers; exec: audit(executor) attaches like viewer.record.
    "audit": {"base", "core", "runtime", "tracing", "hooks", "exec", "audit"},
    # The chat contract above exec and memory, so neither imports the other for it.
    "chat": {"base", "core", "exec", "memory", "chat"},
    # hooks only for `is_secret_field` (approval memory never names secret fields).
    "approval": {"base", "runtime", "tracing", "hooks", "approval"},
    "hooks": {"base", "core", "runtime", "tracing", "hooks"},
    # models and exec import context lazily: ports honour LLMRequest.compression, and
    # run_tool_loop(compression=) compresses its history.
    "models": {
        "base",
        "schema",
        "runtime",
        "hooks",
        "cache",
        "cost",
        "tracing",
        "context",
        "models",
    },
    "exec": {
        "base",
        "core",
        "runtime",
        "hooks",
        "approval",
        "cache",
        "tracing",
        "models",
        "context",
        "exec",
    },
    "flow": {"base", "core", "runtime", "hooks", "approval", "tracing", "exec", "flow"},
    # Context compression: core (recall tool spec), cache (digests, envelopes), hooks (redact
    # before summarising), cost (ledger, model caps), tracing (events), embeddings (scorers).
    # Never exec, approval or models: the tool loop and model ports import it, not back.
    "context": {
        "base",
        "core",
        "runtime",
        "cache",
        "cost",
        "hooks",
        "tracing",
        "embeddings",
        "context",
    },
    "integrations": _ALL - {"server", "optimiser"},
    "server": _ALL - {"optimiser"},
    "optimiser": _ALL - {"integrations", "server", "root"},
    "evals": _ALL - {"integrations", "server", "optimiser", "root"},
    "handlers": {"base", "runtime", "approval", "exec", "handlers"},
    "cli": {"base", "integrations", "viewer", "cli", "evals", "sandbox", "policy", "audit"},
    # Core packages plus evals (reports), memory (chat recall) and the chat contract: never a
    # framework, integration or server.
    "viewer": {
        "base",
        "core",
        "runtime",
        "tracing",
        "hooks",
        "approval",
        "exec",
        "viewer",
        "evals",
        "memory",
        "chat",
    },
}
SCHEMA_IMPORTERS = {"core/registry.py", "core/imported.py"}
VERSION = "_version"

# R6: optional-dependency modules and the third-party roots each may import lazily.
_INTEGRATION_COMMON = ("httpx", "httpx2", "anyio")
INTEGRATION_ROOTS = {
    "openai": (),
    "gemini": ("google",),
    "anthropic": (),
    "langchain": ("langchain_core",),
    "langgraph": ("langgraph", "langchain_core"),
    "adk": ("google",),
    "mcp": ("mcp",),
    "openai_agents": ("agents",),
    "pydantic_ai": ("pydantic_ai",),
    "llamaindex": ("llama_index",),
    "crewai": ("crewai",),
    "openapi": ("httpx", "yaml"),
    "agent_framework": ("agent_framework",),
    "agno": ("agno",),
    "dspy": ("dspy",),
    "strands": ("strands",),
    "haystack": ("haystack",),
    "claude_agent": ("claude_agent_sdk", "mcp"),
    "smolagents": ("smolagents",),
    "autogen": ("autogen_core", "autogen_agentchat"),
    "semantic_kernel": ("semantic_kernel",),
    # google: protobuf json_format; httpcore: the SSRF guard pins the connection to the checked IP.
    "a2a": ("a2a", "google", "starlette", "httpcore"),
    "ag_ui": ("ag_ui", "starlette"),
    "camel": ("camel", "jsonschema"),
    "ag2": ("ag2",),
    "temporal": ("temporalio",),
    "dbos": ("dbos",),
}
ODM_ROOTS: dict[str, set[str]] = {
    "models/openai.py": {"openai", "httpx2"},
    "models/gemini.py": {"google", "httpx"},
    "models/anthropic.py": {"anthropic", "httpx2"},
    "cache/redis.py": {"redis"},
    "tracing/otel.py": {"opentelemetry"},
    "server/asgi.py": {"starlette", "uvicorn", "anyio"},
    "server/__main__.py": {"starlette", "uvicorn", "anyio"},
    "server/wsgi.py": {"a2wsgi", "starlette"},
    "approval/sqlalchemy_store.py": {"sqlalchemy"},
    "approval/sqlalchemy_schema.py": {"sqlalchemy"},
    "hooks/builtin/budgets/sqlalchemy_store.py": {"sqlalchemy"},
    "exec/checkpoints/sqlalchemy_store.py": {"sqlalchemy"},
    "context/formats/langchain.py": {"langchain_core"},
    "hooks/builtin/guards/presidio.py": {"presidio_analyzer", "presidio_anonymizer"},
    "hooks/builtin/guards/nemo.py": {"nemoguardrails"},
    "evals/pytest_plugin.py": {"pytest"},
    "memory/mem0.py": {"mem0"},
    "policy/_load.py": {"yaml"},
    "memory/zep.py": {"zep_cloud"},
    "memory/graphiti.py": {"graphiti_core"},
    "retrieval/llamaindex.py": {"llama_index"},
    **{
        f"integrations/{name}": {*own, *_INTEGRATION_COMMON}
        for name, own in INTEGRATION_ROOTS.items()
    },
}
# R9: never imported by a non-ODM module, not even lazily.
# Plus "langchain" (only langchain_core is imported) and "app" (the host).
FORBIDDEN_ROOTS = {root for group in ODM_ROOTS.values() for root in group} | {"langchain", "app"}
# R8: the only modules that may import dynamically.
DYNAMIC_IMPORTERS = {
    "__init__.py",  # PEP 562 lazy top level
    "integrations/discovery.py",
    "integrations/base.py",
    "server/__main__.py",
    "optimiser/__main__.py",
    "cli/doctor.py",  # importlib.metadata only: installed versions and extras
    "viewer/chat/targets.py",  # `agent-adapter ui --agent module:attr` names the host's code
}

Import = ast.Import | ast.ImportFrom


# --- helpers -------------------------------------------------------------------------------


def _rel(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix()


def sources() -> list[tuple[str, str]]:
    return [(_rel(p), p.read_text(encoding="utf-8")) for p in sorted(PACKAGE.rglob("*.py"))]


def all_imports(tree: ast.AST) -> list[Import]:
    return [n for n in ast.walk(tree) if isinstance(n, Import)]


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def top_level_imports(body: list[ast.stmt]) -> Iterator[Import]:
    """Imports that run at module import time; function bodies and TYPE_CHECKING are lazy."""
    for node in body:
        if isinstance(node, Import):
            yield node
        elif isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                yield from top_level_imports(node.body)
            yield from top_level_imports(node.orelse)
        elif isinstance(node, ast.Try | ast.TryStar):
            for block in (node.body, node.orelse, node.finalbody):
                yield from top_level_imports(block)
            for handler in node.handlers:
                yield from top_level_imports(handler.body)
        elif isinstance(node, ast.For | ast.AsyncFor | ast.While):
            yield from top_level_imports(node.body)
            yield from top_level_imports(node.orelse)
        elif isinstance(node, ast.With | ast.AsyncWith | ast.ClassDef):
            yield from top_level_imports(node.body)
        elif isinstance(node, ast.Match):
            for case in node.cases:
                yield from top_level_imports(case.body)


def roots(node: Import) -> list[str]:
    """Top-level module names an absolute import statement pulls in."""
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if node.level > 0:
        return []
    return [(node.module or "").split(".")[0]]


def _third_party(root: str) -> bool:
    return root not in STDLIB and root != "agent_adapter"


def _package_targets(node: Import) -> list[str]:
    """Package modules an import binds, one per imported name of a package."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names if alias.name.startswith("agent_adapter")]
    module = node.module or ""
    if node.level > 0 or not (module == "agent_adapter" or module.startswith("agent_adapter.")):
        return []
    parts = module.split(".")
    is_package = len(parts) == 1 or (len(parts) == 2 and parts[1] in SUBPACKAGES)
    if is_package:
        return [f"{module}.{alias.name}" for alias in node.names]
    return [module]


def module_key(rel: str) -> str | None:
    """DAG key of a package file; None for the top-level `__init__.py`."""
    if rel == "__init__.py":
        return None
    if rel == f"{VERSION}.py":
        return VERSION
    if rel == "errors.py":
        return "facade"
    if rel == "__main__.py":  # `python -m agent_adapter` is the `agent-adapter` command
        return "cli"
    parts = rel.split("/")
    if parts[0] == "core":
        return "base" if parts[-1].removesuffix(".py") in BASE_NAMES else "core"
    return parts[0]


def target_key(module: str) -> str:
    """DAG key of an imported dotted module path inside the package."""
    parts = module.split(".")
    if len(parts) > 1 and parts[1] == VERSION:
        return VERSION
    if len(parts) == 1 or parts[1] not in SUBPACKAGES:
        return "root"
    if parts[1] == "core" and len(parts) > 2 and parts[2] in BASE_NAMES:
        return "base"
    return parts[1]


def odm_roots(rel: str) -> set[str] | None:
    """Lazy roots an optional-dependency module may import; None for non-ODMs."""
    if rel in ODM_ROOTS:
        return ODM_ROOTS[rel]
    parts = rel.split("/")
    if parts[0] == "integrations" and len(parts) >= 2:
        # `<name>_compress.py` (that framework's compression format) shares its roots.
        name = parts[1].removesuffix(".py").removesuffix("_compress")
        return ODM_ROOTS.get(f"integrations/{name}")
    return None


# --- detectors -----------------------------------------------------------------------------


def relative_imports(source: str) -> list[str]:
    """R1: every relative import, written as in source."""
    return [
        "." * node.level + (node.module or "")
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]


def dag_offenders(rel: str, source: str) -> list[str]:
    """R2-R5: package imports outside the module's allowed set (per imported name)."""
    key = module_key(rel)
    if key is None:
        return []
    offenders = []
    for node in all_imports(ast.parse(source)):
        for target in _package_targets(node):
            target_pkg = target_key(target)
            allowed = target_pkg in ALLOWED[key] or (target_pkg == VERSION and key != VERSION)
            if key == "core" and target_pkg == "schema" and rel not in SCHEMA_IMPORTERS:
                allowed = False
            if not allowed:
                offenders.append(target)
    return offenders


def base_third_party(rel: str, source: str) -> list[str]:
    """R2: third-party roots in a base module other than pydantic and typing_extensions."""
    if module_key(rel) != "base":
        return []
    return [
        root
        for node in all_imports(ast.parse(source))
        for root in roots(node)
        if _third_party(root) and root not in ALLOWED_THIRD_PARTY
    ]


def odm_offenders(rel: str, source: str) -> list[str]:
    """R6: eager third-party imports in an ODM, or lazy ones outside its allowed roots."""
    allowed = odm_roots(rel)
    if allowed is None:
        return []
    tree = ast.parse(source)
    eager = {id(n) for n in top_level_imports(tree.body)}
    offenders = []
    for node in all_imports(tree):
        for root in roots(node):
            if not _third_party(root) or root in ALLOWED_THIRD_PARTY:
                continue
            if id(node) in eager:
                offenders.append(f"eager {root}")
            elif root not in allowed:
                offenders.append(f"lazy {root}")
    return offenders


def non_odm_third_party(rel: str, source: str) -> list[str]:
    """R7: third-party roots other than pydantic/typing_extensions in a non-ODM module."""
    if odm_roots(rel) is not None:
        return []
    return [
        root
        for node in all_imports(ast.parse(source))
        for root in roots(node)
        if _third_party(root) and root not in ALLOWED_THIRD_PARTY
    ]


def dynamic_imports(rel: str, source: str) -> list[str]:
    """R8: importlib or __import__ outside the allowed modules."""
    if rel in DYNAMIC_IMPORTERS:
        return []
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, Import) and "importlib" in roots(node):
            found.append("importlib")
        if isinstance(node, ast.Name) and node.id == "__import__":
            found.append("__import__")
        if isinstance(node, ast.Attribute) and node.attr == "__import__":
            found.append("__import__")
    return found


def forbidden_roots(rel: str, source: str) -> list[str]:
    """R9: a FORBIDDEN_ROOTS import (even lazy) in a non-ODM module."""
    if odm_roots(rel) is not None:
        return []
    return [
        root
        for node in all_imports(ast.parse(source))
        for root in roots(node)
        if root in FORBIDDEN_ROOTS
    ]


# R10: packages outside flow/exec reach their internals only through these seams.
SEAM_OWNERS = {"flow", "exec"}
SEAM_USERS = {"integrations", "server", "optimiser", "evals"}


def private_internals(rel: str, source: str) -> list[str]:
    """R10: a flow/exec private module or name imported by integrations, server or optimiser."""
    if module_key(rel) not in SEAM_USERS:
        return []
    found = []
    for node in all_imports(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom) or node.level > 0 or not node.module:
            continue
        parts = node.module.split(".")
        if parts[0] != "agent_adapter" or len(parts) < 2 or parts[1] not in SEAM_OWNERS:
            continue
        if any(part.startswith("_") for part in parts[2:]):
            found.append(node.module)
        found.extend(f"{node.module}.{a.name}" for a in node.names if a.name.startswith("_"))
    return found


RULES = {
    "R2-R5 dag": dag_offenders,
    "R2 base third party": base_third_party,
    "R6 odm": odm_offenders,
    "R7 non-odm third party": non_odm_third_party,
    "R8 dynamic import": dynamic_imports,
    "R9 forbidden root": forbidden_roots,
    "R10 seams": private_internals,
}
