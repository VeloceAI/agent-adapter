"""The `pip install` line for an example, detected from its imports."""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"

# Third-party module (or dotted prefix) -> the extra that brings it.
EXTRA_OF_MODULE = {
    "openai": "openai", "google.genai": "gemini", "anthropic": "anthropic", "redis": "redis",
    "opentelemetry": "otel", "langchain_core": "langchain", "langgraph": "langgraph",
    "google.adk": "adk", "mcp": "mcp", "agents": "openai-agents", "pydantic_ai": "pydantic-ai",
    "llama_index": "llamaindex", "crewai": "crewai", "yaml": "openapi",
    "agent_framework": "agent-framework", "agno": "agno", "dspy": "dspy", "strands": "strands",
    "haystack": "haystack", "smolagents": "smolagents", "claude_agent_sdk": "claude-agent",
    "autogen_core": "autogen", "autogen_agentchat": "autogen", "semantic_kernel": "semantic-kernel",
    "camel": "camel", "ag2": "ag2", "a2a": "a2a", "ag_ui": "ag-ui", "sqlalchemy": "sqlalchemy",
    "aioboto3": "dynamodb", "temporalio": "temporal", "dbos": "dbos", "starlette": "server",
    "uvicorn": "server", "a2wsgi": "server-wsgi",
}  # fmt: skip
# Modules that need an extra and a package: the `otel` extra installs only the API.
BOTH_OF_MODULE = {
    "opentelemetry.sdk": ({"otel"}, {"opentelemetry-sdk"}),
    "opentelemetry.exporter.otlp": ({"otel"}, {"opentelemetry-exporter-otlp-proto-http"}),
}
# Third-party modules no extra brings: installed by name.
PACKAGE_OF_MODULE = {
    "fastapi": "fastapi", "flask": "flask", "werkzeug": "flask", "django": "django",
    "celery": "celery", "aiohttp": "aiohttp", "fakeredis": "fakeredis", "pytest": "pytest",
    "httpx": "httpx", "httpx2": "httpx2", "boto3": "boto3", "mistralai": "mistralai",
    "groq": "groq", "traceloop": "traceloop-sdk",
}  # fmt: skip
# extra -> extras and packages it already installs, dropped from the line.
IMPLIES = {
    "a2a": {"httpx"},
    "adk": {"gemini"},
    "camel": {"openai"},
    "claude-agent": {"mcp", "httpx", "httpx2"},
    "langgraph": {"langchain"},
    "mcp": {"httpx", "httpx2"},
    "openai-agents": {"openai"},
    "openapi": {"httpx"},
    "server": {"mcp", "httpx", "httpx2"},
    "server-wsgi": {"server", "mcp", "httpx", "httpx2"},
}
# Renderers and helpers: they need no SDK; SDK clients show up as the example's own imports.
PURE_INTEGRATIONS = {"openai", "anthropic", "gemini", "base", "discovery"}
# agent_adapter.server names that need the server extras (the rest is pure Python).
SERVER_NAMES = {"create_app": "server", "serve": "server", "create_wsgi_app": "server-wsgi"}
# Names (extras or packages) a demo only uses on a path its `main()` does not take.
OPTIONAL = {
    "approval_stores": {"redis"},
    "approval_stores_sql": {"dynamodb"},
    "compression_backend": {"fastapi"},  # the route; the script part runs on the core
    "deploy/app": {"redis"},  # only when REDIS_URL is set
    # vendor SDKs: main() runs on fakes; each builder imports its SDK on first use
    "models_bedrock": {"boto3"},
    "models_mistral": {"mistralai"},
    "models_openai_compatible": {"openai", "groq"},
    "traces_otlp": {"opentelemetry-exporter-otlp-proto-http", "traceloop-sdk"},
}
# Demos whose every framework is optional: each part needs only the framework it shows.
OPTIONAL_ALL = {"compression_framework_hooks", "harness_frameworks"}
# Imports that exist to show what happens when the package is missing.
NOT_NEEDED = {"guide_troubleshooting": {"crewai"}}
IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "MissingDependency"}


@dataclass
class Req:
    extras: set[str] = field(default_factory=set)
    packages: set[str] = field(default_factory=set)

    def add(self, other: Req) -> None:
        self.extras |= other.extras
        self.packages |= other.packages

    def take(self, names: set[str]) -> Req:
        """Remove `names` from this requirement and return them as their own."""
        moved = Req(self.extras & names, self.packages & names)
        self.extras -= names
        self.packages -= names
        return moved


@dataclass
class Needs:
    required: Req = field(default_factory=Req)
    optional: Req = field(default_factory=Req)

    def merge(self, other: Needs) -> None:
        self.required.add(other.required)
        self.optional.add(other.optional)


def _guarded(tree: ast.Module) -> set[int]:
    """ids of import nodes inside a `try` that catches an import error."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        caught = {ast.unparse(h.type) for h in node.handlers if h.type is not None}
        if any(name in text for text in caught for name in IMPORT_ERRORS):
            found |= {id(n) for s in node.body for n in ast.walk(s)}
    return found


def _imports(tree: ast.Module) -> Iterable[tuple[str, tuple[str, ...], bool]]:
    """(module, imported names, optional) for every absolute import, lazy ones included."""
    guarded = _guarded(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, (), id(node) in guarded
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, tuple(a.name for a in node.names), id(node) in guarded


def _integration_extra(name: str) -> str:
    """The extra an `agent_adapter.integrations.<name>` export needs ('' for none)."""
    if name in PURE_INTEGRATIONS:
        return ""
    module = importlib.import_module(f"agent_adapter.integrations.{name}")  # lazy: no SDK import
    return str(getattr(getattr(module, "INTEGRATION", None), "extra", "") or "")


def _matches(module: str, key: str) -> bool:
    return module == key or module.startswith(key + ".")


def _one(module: str, names: tuple[str, ...]) -> tuple[Req, str | None]:
    """What one import statement needs, and the sibling example it imports (if any)."""
    parts = module.split(".")
    if parts[0] == "agent_adapter":
        if parts[1:2] == ["integrations"] and len(parts) > 2:
            extra = _integration_extra(parts[2])
            return Req({extra} if extra else set()), None
        if parts[1:2] == ["server"]:
            return Req({SERVER_NAMES[n] for n in names if n in SERVER_NAMES}), None
        return Req(), None
    for key, (extras, packages) in BOTH_OF_MODULE.items():
        if _matches(module, key):
            return Req(set(extras), set(packages)), None
    for key, extra in EXTRA_OF_MODULE.items():
        if _matches(module, key):
            return Req({extra}), None
    if parts[0] in PACKAGE_OF_MODULE:
        return Req(packages={PACKAGE_OF_MODULE[parts[0]]}), None
    if (EXAMPLES / f"{parts[0]}.py").is_file():
        return Req(), parts[0]
    return Req(), None  # the standard library, or pydantic (a core dependency)


def needs(rel: str, seen: set[str] | None = None) -> Needs:
    """What an example needs, through the sibling helpers it imports."""
    seen = set() if seen is None else seen
    seen.add(rel)
    tree = ast.parse((EXAMPLES / f"{rel}.py").read_text(encoding="utf-8"))
    found = Needs()
    for module, names, optional in _imports(tree):
        req, sibling = _one(module, names)
        (found.optional if optional else found.required).add(req)
        if sibling and sibling not in seen:
            found.merge(needs(sibling, seen))
    found.required.take(NOT_NEEDED.get(rel, set()))
    everything = found.required.extras | found.required.packages
    moved = everything if rel in OPTIONAL_ALL else OPTIONAL.get(rel, set())
    found.optional.add(found.required.take(moved))
    return found


def _line(req: Req, implied: set[str], *, base: bool) -> str:
    """One `pip install` command; `base` keeps `agent-adapter` when nothing else is needed."""
    extras = sorted(req.extras - implied)
    packages = sorted(req.packages - implied)
    target = [f'"agent-adapter[{",".join(extras)}]"'] if extras else []
    if not target and base:
        target = ["agent-adapter"]
    return " ".join(["pip install", *target, *packages]) if target or packages else ""


def install_line(rel: str) -> tuple[str, str]:
    """`pip install ...` for an example, and a second line for the parts that are optional."""
    found = needs(rel)
    implied = set().union(*(IMPLIES.get(e, set()) for e in found.required.extras))
    required = _line(found.required, implied, base=True)
    have = implied | found.required.extras | found.required.packages
    have |= set().union(*(IMPLIES.get(e, set()) for e in found.optional.extras))
    return required, _line(found.optional, have, base=False)
