"""Layout of the API reference pages: which public module goes on which page.

Rewrite the generated parts of every page after an `__all__` changes:
    python -m tests.docs.reference_layout
"""

import importlib
import pkgutil
from functools import cache
from pathlib import Path

import agent_adapter

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "docs" / "reference"
# Shared names that two subpackages export: errors live in agent_adapter.errors.
PREFERRED = ("agent_adapter", "agent_adapter.errors")
API_HEADING = "## API\n"  # errors.md: everything above it is hand-written (the catalogue)

_I = "agent_adapter.integrations."
_FRAMEWORKS = (
    "langgraph",
    "adk",
    "openai_agents",
    "pydantic_ai",
    "llamaindex",
    "crewai",
    "agent_framework",
    "agno",
    "strands",
    "haystack",
    "claude_agent",
    "semantic_kernel",
    "dspy",
    "smolagents",
    "autogen",
    "camel",
    "ag2",
)
_SUBMODULES = ("adk.compile", "adk.tools", "langgraph.approval", "langgraph.compile")
_INSTALL = "Each module imports its SDK lazily: install the extra first"
_INTEGRATIONS = f"{_INSTALL} ([Integrations](../integrations/index.md))."

# page -> (title, intro, modules); the top-level page also carries TOP_LEVEL_NOTE.
PAGES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "top-level": (
        "Top level",
        "The everyday names, imported from `agent_adapter`. Each loads on first use, so "
        "`import agent_adapter` stays cheap. Concepts: [Tools](../concepts/tools.md), "
        "[Registry](../concepts/registry.md), [Executor](../concepts/executor.md).",
        ("agent_adapter",),
    ),
    "exec": (
        "Executor",
        "Running tool calls: retries, plans and the tool loop. Concepts: "
        "[Executor](../concepts/executor.md), "
        "[Executor policies](../concepts/executor-policies.md).\n\n"
        "`agent_adapter.exec.integration` is a seam for flow compilers and framework "
        "bridges; most code never needs it.",
        ("agent_adapter.exec", "agent_adapter.exec.control", "agent_adapter.exec.integration"),
    ),
    "hooks": (
        "Hooks",
        "Hooks around tool, model and node calls, and the built-in hooks. "
        "Concepts: [Hooks](../concepts/hooks.md).",
        ("agent_adapter.hooks", "agent_adapter.hooks.builtin"),
    ),
    "approval": (
        "Approval",
        "Confirm rules, approval ports and stores. Concepts: "
        "[Approval](../concepts/approval.md), [Approval stores](../concepts/approval-stores.md).",
        ("agent_adapter.approval",),
    ),
    "flow": (
        "Flows",
        "Flow graphs, nodes and the runner. Concepts: [Flows](../concepts/flows.md), "
        "[Flow pause and resume](../concepts/flow-resume.md).\n\n"
        "`agent_adapter.flow.visit` is a seam for flow compilers; most code never needs it.",
        ("agent_adapter.flow", "agent_adapter.flow.runner", "agent_adapter.flow.visit"),
    ),
    "cost": (
        "Cost",
        "Ledger sinks, summaries and resolved prices. Concepts: [Cost](../concepts/cost.md).",
        ("agent_adapter.cost",),
    ),
    "cache": (
        "Cache",
        "Tool-result and model-response caches, keys and provider prompt caches. "
        "Concepts: [Cache](../concepts/cache.md).",
        ("agent_adapter.cache", "agent_adapter.cache.providers"),
    ),
    "embeddings": (
        "Embeddings",
        "The embedding port, a callable embedder and its cache. "
        "Concepts: [Embeddings](../concepts/embeddings.md).",
        ("agent_adapter.embeddings",),
    ),
    "context": (
        "Context compression",
        "Compression policies, steps, scorers, `CompressingModel` and the message-format "
        "converters. Concepts: [Context compression](../concepts/context-compression.md).",
        ("agent_adapter.context", "agent_adapter.context.formats"),
    ),
    "runtime": (
        "Runtime",
        "The current tool context, ids and loop-local state. "
        "Concepts: [Runtime](../concepts/runtime.md).",
        ("agent_adapter.runtime",),
    ),
    "tracing": (
        "Tracing",
        "Events, event sinks and OpenTelemetry spans. Concepts: [Tracing](../concepts/tracing.md).",
        ("agent_adapter.tracing",),
    ),
    "models": (
        "Models",
        "Model ports, provider models and usage helpers. Concepts: "
        "[Model ports](../concepts/models.md), [Tool loop](../concepts/tool-loop.md).",
        tuple(
            f"agent_adapter.models{suffix}"
            for suffix in ("", ".openai", ".gemini", ".anthropic", ".errors")
        ),
    ),
    "server": (
        "HTTP server",
        "The tool service, its ASGI and WSGI apps, auth and problems. "
        "Guide: [HTTP server](../cross-language/http-server.md).",
        ("agent_adapter.server", "agent_adapter.server.asgi", "agent_adapter.server.service"),
    ),
    "handlers": (
        "Handlers",
        "Run tool calls from queue messages, serverless events and pipelines.",
        ("agent_adapter.handlers",),
    ),
    "viewer": (
        "Run viewer",
        "Record runs to a folder and browse them on 127.0.0.1 (`agent-adapter ui`), or chat "
        "with an agent there. Guides: [See your runs](../how-to/see-your-runs.md), "
        "[Chat with your agent](../how-to/chat-with-your-agent.md).",
        ("agent_adapter.viewer", "agent_adapter.viewer.chat"),
    ),
    "integrations-providers": (
        "Integrations: model APIs, protocols and importers",
        "Export tools to model APIs and agent protocols (MCP, A2A, AG-UI); import tools from "
        "LangChain, OpenAPI and MCP. " + _INTEGRATIONS,
        tuple(
            _I + name
            for name in (
                "openai",
                "gemini",
                "anthropic",
                "langchain",
                "openapi",
                "openapi.auth",
                "mcp",
                "mcp.annotations",
                "mcp.client",
                "mcp.oauth",
                "mcp.server",
                "mcp.tasks",
                "a2a",
                "ag_ui",
            )
        ),
    ),
    "integrations-frameworks": (
        "Integrations: agent frameworks",
        "Export tools, approval and flows to agent frameworks. " + _INTEGRATIONS,
        tuple(_I + name for name in (*_FRAMEWORKS, *_SUBMODULES, "langgraph.tools")),
    ),
    "integrations-base": (
        "Integrations: protocol and discovery",
        "The integration protocol and entry-point discovery, for authors of new integrations "
        "([Writing an integration](../contributing/writing-an-integration.md)).",
        (_I + "base", _I + "discovery"),
    ),
    "optimiser": (
        "Optimiser",
        "Improve tool descriptions and prompts against eval cases ([Optimiser](../optimiser.md)).",
        ("agent_adapter.optimiser", "agent_adapter.optimiser.graders"),
    ),
}
ERRORS_PAGE = "errors"
TOP_LEVEL_NOTE = """Four names have no docstring of their own, so here is what they are:

| Name | What it is |
|---|---|
| `default_registry` | The process-wide `Registry`, created on first access, then reused |
| `tool`, `wrap` | `default_registry.tool` and `default_registry.wrap`: register a function on it |
| `Decision` | An alias of `ApprovalDecision` |
| `__version__` | The installed version string |
"""


@cache
def public_modules() -> dict[str, tuple[str, ...]]:
    """Module name -> `__all__`, for `agent_adapter` and every non-private module with one."""
    found = {"agent_adapter": tuple(agent_adapter.__all__)}
    for info in pkgutil.walk_packages(agent_adapter.__path__, "agent_adapter."):
        if any(part.startswith("_") for part in info.name.split(".")[1:]):
            continue
        module = importlib.import_module(info.name)
        if hasattr(module, "__all__"):
            found[info.name] = tuple(module.__all__)
    return found


def _value(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


def canonical_home(module: str, name: str) -> str:
    """The import path of `module.name`: the top level, then errors, then the shallowest home."""
    value = _value(module, name)
    homes = [
        home
        for home, names in public_modules().items()
        if name in names and _value(home, name) is value
    ]
    for preferred in PREFERRED:
        if preferred in homes:
            return preferred
    return min(homes, key=lambda home: (home.count("."), home))


def page_of() -> dict[str, str]:
    """Module -> the page (file stem) that documents it."""
    pages = {module: page for page, (_, _, modules) in PAGES.items() for module in modules}
    pages["agent_adapter.errors"] = ERRORS_PAGE
    return pages


def module_block(module: str, page: str, *, heading: bool) -> list[str]:
    """The markdown for one module: its directive and links to names documented elsewhere."""
    names = public_modules()[module]
    own = sorted(name for name in names if canonical_home(module, name) == module)
    others = sorted((name, canonical_home(module, name)) for name in names if name not in own)
    lines = [f"## `{module}`", ""] if heading else []
    if own:
        lines += [f"::: {module}", "    options:", f"      members: [{', '.join(own)}]", ""]
    if others:
        pages = page_of()
        links = ", ".join(
            f"[`{name}`]({'' if pages[home] == page else pages[home] + '.md'}#{home}.{name})"
            for name, home in others
        )
        lead = "Also exported here" if own else f"`{module}` re-exports"
        lines += [f"{lead}, documented under the canonical path: {links}.", ""]
    return lines


def render(page: str) -> str:
    """The full text of a generated page."""
    title, intro, modules = PAGES[page]
    lines = [f"# {title}", "", intro, ""]
    if page == "top-level":
        lines += [TOP_LEVEL_NOTE]
    for module in modules:
        lines += module_block(module, page, heading=len(modules) > 1)
    return "\n".join(lines).rstrip() + "\n"


def render_errors_api(current: str) -> str:
    """errors.md with its generated API section rewritten."""
    head = current.split(API_HEADING, 1)[0]
    block = module_block("agent_adapter.errors", ERRORS_PAGE, heading=False)
    return head + API_HEADING + "\n" + "\n".join(block).rstrip() + "\n"


def main() -> None:
    """Rewrite every generated reference page."""
    unplaced = sorted(set(public_modules()) - set(page_of()))
    if unplaced:
        raise SystemExit(f"add these modules to PAGES in {__file__}: {unplaced}")
    for page in PAGES:
        (FOLDER / f"{page}.md").write_text(render(page), encoding="utf-8")
    errors = FOLDER / f"{ERRORS_PAGE}.md"
    errors.write_text(render_errors_api(errors.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
