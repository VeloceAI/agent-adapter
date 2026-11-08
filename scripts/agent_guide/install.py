"""The `## Install` section, built from `pyproject.toml` extras and `noxfile.py`."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GATE = re.compile(r"^_?SUPPORTED(?:_SDK)? = \(\(([\d, ]+)\), \(([\d, ]+)\)\)", re.M)
OWN_ENV = re.compile(r"([a-z][a-z0-9-]*) \(([^)]+)\)")
NL = "\n"
CHECKOUT_ONLY = {"dev", "docs"}  # tooling for this repository, installed from a checkout

# extra -> what you need it for, in plain words. Every extra in pyproject.toml needs a row.
NEEDS: dict[str, str] = {
    "openai": "OpenAI models (`OpenAIModel`, also Ollama, vLLM, LM Studio)",
    "gemini": "Gemini models (`GeminiModel`, `genai_tool()`)",
    "anthropic": "Anthropic models (`AnthropicModel`)",
    "redis": "Redis cache and approval store",
    "otel": "OpenTelemetry spans (`OTelSink`)",
    "langchain": "LangChain tools, export and import",
    "langgraph": "LangGraph tool nodes, approvals, compiled flows",
    "adk": "Google ADK tools and flows",
    "mcp": "MCP server, client import, Streamable HTTP",
    "openai-agents": "OpenAI Agents SDK tools",
    "pydantic-ai": "Pydantic AI tools",
    "llamaindex": "LlamaIndex tools",
    "crewai": "CrewAI tools",
    "openapi": "tools from an OpenAPI document (`import_openapi`)",
    "agent-framework": "Microsoft Agent Framework tools",
    "agno": "Agno tools",
    "dspy": "DSPy tools",
    "strands": "Strands Agents tools",
    "haystack": "Haystack tools",
    "smolagents": "smolagents tools",
    "claude-agent": "Claude Agent SDK tools",
    "autogen": "AutoGen tools",
    "semantic-kernel": "Semantic Kernel functions",
    "camel": "CAMEL-AI tools",
    "ag2": "AG2 tools",
    "a2a": "A2A server and client",
    "ag-ui": "AG-UI event stream (CopilotKit and other UIs)",
    "sqlalchemy": "Postgres or any SQLAlchemy database as the approval store",
    "dynamodb": "DynamoDB approval store (AWS serverless)",
    "server": "HTTP tool server (ASGI) with MCP mounted",
    "server-wsgi": "HTTP tool server under WSGI (Flask, Django)",
    "presidio": "Presidio PII guard (`PresidioGuard`; also needs a spaCy model)",
    "nemo": "NeMo Guardrails guard (`NemoGuard`)",
    "mem0": "Mem0 memory backend (`Mem0Memory`)",
    "zep": "Zep memory backend (`ZepMemory`)",
    "graphiti": "Graphiti knowledge-graph memory (`GraphitiMemory`)",
    "temporal": "flows as Temporal workflows (`agent_adapter.integrations.temporal`)",
    "dbos": "flows as DBOS workflows (`DbosFlows`)",
    "all": "every extra that installs together in one environment",
    "dev": "running this repository's tests and linters",
    "docs": "building this repository's docs site",
}


def pyproject() -> dict[str, Any]:
    """The parsed `pyproject.toml`."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _nox_constant(name: str) -> Any:
    """A literal module-level constant of `noxfile.py` (plain or annotated assignment)."""
    tree = ast.parse((ROOT / "noxfile.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(value)
    raise LookupError(f"noxfile.py has no {name} constant")


def nox_pythons() -> list[str]:
    """`PYTHONS` from `noxfile.py`: the versions every contract session runs on."""
    return list(_nox_constant("PYTHONS"))


def nox_targets() -> dict[str, str]:
    """`EXTRAS` from `noxfile.py`: integration module -> pip extra."""
    return dict(_nox_constant("EXTRAS"))


def in_all(extras: dict[str, list[str]]) -> set[str]:
    """The extras the `all` extra pulls in."""
    members: set[str] = set()
    for requirement in extras["all"]:
        members.update(requirement.split("[", 1)[1].rstrip("]").split(","))
    return members


def own_env_reasons() -> dict[str, str]:
    """`extra -> reason` from the pyproject comment above `all = [`."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    comment = text.split("Own environment:", 1)[1].split("all = [", 1)[0]
    comment = " ".join(line.lstrip("# ").strip() for line in comment.splitlines())
    return dict(OWN_ENV.findall(comment))


def gate(target: str) -> str:
    """The version range an integration checks at use, or ""."""
    base = ROOT / "agent_adapter" / "integrations"
    for path in (base / f"{target}.py", base / target / "_support.py"):
        if path.is_file() and (match := GATE.search(path.read_text(encoding="utf-8"))):
            low, high = (".".join(p.strip() for p in g.split(",")) for g in match.groups())
            return f"{low} to below {high}"
    return ""


def pins(requirements: list[str]) -> str:
    """The extra's own pins, plus the extras it includes."""
    own = [f"`{r}`" for r in requirements if not r.startswith("agent-adapter[")]
    via = [r.split("[", 1)[1].rstrip("]") for r in requirements if r.startswith("agent-adapter[")]
    return ", ".join(own + [f"extra `{v}`" for v in via]) or "none"


def command(extra: str) -> str:
    """A pip one-liner that works in bash, zsh, PowerShell and cmd."""
    return f'`pip install "agent-adapter[{extra}]"`'


def _rows(*, full: bool) -> list[str]:
    project = pyproject()["project"]
    extras: dict[str, list[str]] = project["optional-dependencies"]
    missing = sorted(set(extras) - set(NEEDS))
    if missing:
        raise LookupError(f"add these extras to NEEDS in {__file__}: {missing}")
    shared, reasons = in_all(extras), own_env_reasons()
    gates = {extra: gate(target) for target, extra in nox_targets().items() if extra}
    core = ", ".join(f"`{dep}`" for dep in project["dependencies"])
    rows = [
        "| I need | pip command | Supported versions |",
        "|---|---|---|",
        "| tools, executor, approval, budgets, cache, flows, compression, plain backends, "
        f"queues, serverless | `pip install agent-adapter` | {core} |",
    ]
    for extra, requirements in extras.items():
        if extra in CHECKOUT_ONLY:
            if full:  # repository tooling: only the full guide lists it
                cmd = f'`pip install -e ".[{extra}]"` (in a checkout)'
                rows.append(f"| {NEEDS[extra]} | {cmd} | {pins(requirements)} |")
            continue
        if extra == "all":
            outside = sorted(set(extras) - shared - CHECKOUT_ONLY - {"all"})
            note = "every extra above except " + ", ".join(f"`{e}`" for e in outside)
            rows.append(f"| {NEEDS[extra]} | {command(extra)} | {note} |")
            continue
        notes = [pins(requirements)]
        if full and gates.get(extra):
            notes.append(f"checked at use: {gates[extra]}")
        if extra in reasons and extra not in shared:
            notes.append(f"own environment ({reasons[extra]})")
        rows.append(f"| {NEEDS[extra]} | {command(extra)} | {'; '.join(notes)} |")
    return rows


def render(*, full: bool = False) -> str:
    """The `## Install` section; the full tier adds the repository's dev and docs extras."""
    project = pyproject()["project"]
    pythons = ", ".join(nox_pythons())
    return NL.join(
        [
            "## Install",
            "",
            "`agent-adapter` is on PyPI (Apache-2.0). Install the core, or the extras you need:",
            "",
            "```bash",
            'pip install "agent-adapter[openai]"',
            "```",
            "",
            'From a checkout, use `pip install -e ".[openai]"` instead. Check the setup with',
            "`agent-adapter doctor`.",
            "",
            f"Python: `{project['requires-python']}`; tested on {pythons}. Framework versions:",
            "the newest release inside each pinned range is tested. An extra marked",
            '"own environment" conflicts with another extra: give it its own venv.',
            "",
            *_rows(full=full),
            "",
        ]
    )
