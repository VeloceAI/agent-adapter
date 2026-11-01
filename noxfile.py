"""Nox sessions. Sessions whose tests do not exist yet pass without running."""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import nox

nox.options.sessions = ["core", "core-floor", "lint", "typecheck"]
nox.options.reuse_existing_virtualenvs = True

ROOT = Path(__file__).parent
PYTHONS = ["3.11", "3.12", "3.13"]
PYDANTIC_FLOOR = "pydantic==2.12.*"

# Contract target -> pip extra ("" = core only); each entry makes a `contract-<target>` session.
EXTRAS: dict[str, str] = {
    "openai": "",
    "gemini": "gemini",
    "anthropic": "",
    "langchain": "langchain",
    "langgraph": "langgraph",
    "adk": "adk",
    "mcp": "mcp",
    "openai_agents": "openai-agents",
    "pydantic_ai": "pydantic-ai",
    "llamaindex": "llamaindex",
    "crewai": "crewai",
    "openapi": "openapi",
    "agent_framework": "agent-framework",
    "agno": "agno",
    "dspy": "dspy",
    "strands": "strands",
    "haystack": "haystack",
    "smolagents": "smolagents",
    "claude_agent": "claude-agent",
    "autogen": "autogen",
    "semantic_kernel": "semantic-kernel",
    "camel": "camel",
    "ag2": "ag2",
    "a2a": "a2a",
    "ag_ui": "ag-ui",
}
SESSION_ENV: dict[str, dict[str, str]] = {"pydantic_ai": {"PYDANTIC_AI_NO_BANNER": "1"}}
# Packages a contract session installs beyond its extra.
SESSION_PACKAGES: dict[str, list[str]] = {"ag_ui": ["starlette>=1.0,<2", "fastapi"]}
# Sessions whose test directory also holds unmarked SDK tests, so no `-m contract` filter.
ALL_MARKERS = {"a2a"}
MODEL_ODMS = [
    "agent_adapter.models.openai",
    "agent_adapter.models.gemini",
    "agent_adapter.models.anthropic",
    "agent_adapter.cache.redis",
    "agent_adapter.tracing.otel",
]


def _coverage_env(session: nox.Session) -> dict[str, str]:
    """One coverage data file per session, so `coverage` can combine them."""
    name = "".join(c if c.isalnum() else "-" for c in session.name)
    return {"COVERAGE_FILE": str(ROOT / f".coverage.{name}")}


# pyproject's coverage config omits every optional-dependency module (ODM) from the core
# run; ODM sessions and the combined report use this config instead.
ODM_COVERAGE_CONFIG = """\
[run]
branch = True
source = agent_adapter
[report]
show_missing = True
skip_covered = True
"""


def _odm_config(session: nox.Session) -> str:
    path = Path(session.create_tmp()) / "coverage-odm.ini"
    path.write_text(ODM_COVERAGE_CONFIG, encoding="utf-8")
    return str(path)


def _odm_cov(session: nox.Session, *modules: str) -> list[str]:
    """pytest-cov arguments measuring the given ODMs (only those that exist yet), >= 90 %."""
    present = [m for m in modules if _module_path(m).exists()]
    if not present:
        session.log("none of the measured modules exist yet; running without coverage")
        return []
    cov = [f"--cov={module}" for module in present]
    return [f"--cov-config={_odm_config(session)}", *cov, "--cov-fail-under=90"]


def _module_path(module: str) -> Path:
    base = ROOT.joinpath(*module.split("."))
    return base if base.is_dir() else base.with_suffix(".py")


def _core_omitted() -> set[str]:
    """Module paths the core coverage run omits; their own sessions measure them."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(config["tool"]["coverage"]["run"]["omit"])


def _contract_cov_modules(target: str) -> list[str]:
    """The integration module, plus its compression converter when the core run omits it."""
    modules = [f"agent_adapter.integrations.{target}"]
    if f"agent_adapter/integrations/{target}_compress.py" in _core_omitted():
        modules.append(f"agent_adapter.integrations.{target}_compress")
    return modules


def _existing(session: nox.Session, *paths: str) -> list[str]:
    """The given test paths that exist; logs and returns [] when none do."""
    found = [p for p in paths if (ROOT / p).exists()]
    if not found:
        session.log(f"no tests yet at {', '.join(paths)}; nothing to run")
    return found


@nox.session(python=PYTHONS)
def core(session: nox.Session) -> None:
    """Offline suite with the latest pydantic; fails under 90 % coverage."""
    session.install("-e", ".[dev]")
    session.run("pytest", "-m", "not contract", "--cov", env=_coverage_env(session))


@nox.session(name="core-floor", python=PYTHONS)
def core_floor(session: nox.Session) -> None:
    """Offline suite on the lowest supported pydantic."""
    session.install("-e", ".[dev]", PYDANTIC_FLOOR)
    session.run("python", "-c", "import pydantic; print('pydantic', pydantic.VERSION)")
    session.run("pytest", "-m", "not contract")


@nox.session(python="3.12")
def lint(session: nox.Session) -> None:
    """Ruff lint and format check."""
    session.install("-e", ".[dev]")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python="3.12")
def typecheck(session: nox.Session) -> None:
    """mypy --strict over the package."""
    # httpx (and its httpcore) must be real types: the A2A SSRF guard subclasses their transports.
    session.install("-e", ".[dev,openapi]")
    session.run("mypy")


@nox.session(python="3.12")
def docs(session: nox.Session) -> None:
    """Build the docs site with warnings as errors."""
    session.install("-e", ".[docs]")
    session.run("mkdocs", "build", "--strict")


@nox.session(python=PYTHONS, default=False)
def models(session: nox.Session) -> None:
    """Provider ports, Redis cache and the OTel sink with their SDKs installed."""
    paths = _existing(session, "tests/models", "tests/cache", "tests/tracing")
    if not paths:
        return
    session.install("-e", ".[dev,openai,gemini,anthropic,redis,otel]", "opentelemetry-sdk")
    session.run("pytest", *paths, *_odm_cov(session, *MODEL_ODMS), env=_coverage_env(session))


@nox.session(name="docs-examples", python=PYTHONS, default=False)
def docs_examples(session: nox.Session) -> None:
    """Backend, tutorial and agent-guide examples with their frameworks and SDKs installed."""
    tests = _existing(session, "tests/docs")
    if not tests:
        return
    session.install(
        "-e",
        # openai, anthropic and a2a: the agent-guide recipes for hosted models and A2A run here
        ".[dev,server,server-wsgi,langgraph,langchain,openai,anthropic,a2a]",
        "fastapi",
        "flask",
        "django",
        "celery",
        "httpx",
    )
    session.run("pytest", *tests, env=_coverage_env(session))


# Contract sessions whose extra changes the core pins; the session says so in its log.
SESSION_NOTES: dict[str, str] = {
    "semantic_kernel": "semantic-kernel pins pydantic<2.14: this env runs on pydantic 2.13.x",
    "adk": "google-adk pins opentelemetry-api/sdk<=1.44: this env is separate from `all`",
    "strands": "strands-agents needs mcp<2.2: this env is separate from `all`",
    "crewai": "crewai needs mcp<2: this env is separate from `all`",
    "camel": "camel-ai needs mcp<2 and pins pydantic<=2.12.0: this env is separate from `all`",
}


def _contract_session(target: str, extra: str) -> None:
    @nox.session(name=f"contract-{target}", python=PYTHONS, default=False)
    def contract(session: nox.Session) -> None:
        tests = _existing(session, f"tests/integrations/test_{target}_contract.py")
        # A package's own test directory (tests/integrations/<target>/) and its compression
        # tests (tests/integrations/test_<target>_compression.py) run in the same session.
        if tests and (ROOT / "tests" / "integrations" / target).is_dir():
            tests.append(f"tests/integrations/{target}")
        compression = f"tests/integrations/test_{target}_compression.py"
        if tests and (ROOT / compression).is_file():
            tests.append(compression)
        if not tests:
            return
        session.install("-e", f".[dev,{extra}]" if extra else ".[dev]")
        if target in SESSION_PACKAGES:
            session.install(*SESSION_PACKAGES[target])
        if target in SESSION_NOTES:
            session.log(SESSION_NOTES[target])
        session.run("python", "-c", "import pydantic; print('pydantic', pydantic.VERSION)")
        env = {**_coverage_env(session), **SESSION_ENV.get(target, {})}
        cov = _odm_cov(session, *_contract_cov_modules(target))
        marker = [] if target in ALL_MARKERS else ["-m", "contract"]
        session.run("pytest", *marker, *tests, *cov, env=env)


for _target, _extra in EXTRAS.items():
    _contract_session(_target, _extra)


@nox.session(name="contract-all-compatible", python=PYTHONS, default=False)
def contract_all_compatible(session: nox.Session) -> None:
    """Every extra in `all` installed together: pip check, then every contract test."""
    session.install("-e", ".[dev,all]")
    session.run("pip", "check")
    if _existing(session, "tests/integrations"):
        session.run(
            "pytest",
            "-m",
            "contract",
            "tests/integrations",
            success_codes=[0, 5],  # 5: no contract tests collected yet
            env={"PYDANTIC_AI_NO_BANNER": "1"},
        )


@nox.session(python=PYTHONS, default=False)
def plugin(session: nox.Session) -> None:
    """Third-party entry-point discovery with the demo integration installed."""
    fixture = "tests/fixtures/demo_integration"
    tests = _existing(session, "tests/integrations/test_discovery.py")
    if not tests or not _existing(session, fixture):
        return
    session.install("-e", ".[dev]")  # dev brings hatchling, the fixture's build backend
    session.install("--no-build-isolation", f"./{fixture}")
    kit = ["tests/integrations/test_kit_selftest.py", "tests/integrations/test_kit_faults.py"]
    session.run("pytest", *tests, *kit)


@nox.session(python=PYTHONS, default=False)
def server(session: nox.Session) -> None:
    """HTTP tool server, WSGI app and MCP HTTP app."""
    tests = _existing(session, "tests/server")
    if not tests:
        return
    session.install("-e", ".[dev,server,server-wsgi,mcp]")
    cov = _odm_cov(session, "agent_adapter.server")
    session.run("pytest", *tests, *cov, env=_coverage_env(session))


@nox.session(python="3.12", default=False)
def ts(session: nox.Session) -> None:
    """TypeScript client tests against a live server (offline; needs Node >= 22.18)."""
    tests = _existing(session, "tests/server/test_ts_client.py")
    if not tests:
        return
    if shutil.which("node") is None:
        session.error("the ts session needs Node >= 22.18 on PATH")
    session.install("-e", ".[dev,server]")
    session.run("pytest", *tests)


@nox.session(name="ts-build", python=False, default=False)
def ts_build(session: nox.Session) -> None:
    """Build and lint the TypeScript package (online: needs the npm registry)."""
    client = ROOT / "clients" / "typescript"
    if not (client / "package.json").exists():
        session.log("no TypeScript client yet; nothing to build")
        return
    session.chdir(str(client))
    session.run("npm", "ci", external=True)
    # tests and examples are outside the build: type-check and format-check them too
    session.run("npx", "tsc", "-p", "tsconfig.json", external=True)
    session.run("npx", "prettier", "--check", ".", external=True)
    session.run("npx", "tsdown", external=True)
    session.run("npx", "publint", external=True)
    session.run("npx", "attw", "--pack", ".", external=True)


@nox.session(python="3.12")
def coverage(session: nox.Session) -> None:
    """Combine the data files written by the other sessions and enforce 90 %."""
    if not list(ROOT.glob(".coverage.*")):
        session.log("no coverage data files; run core (and the extra sessions) first")
        return
    session.install("coverage[toml]")
    config = f"--rcfile={_odm_config(session)}"
    session.run("coverage", "combine", config, *map(str, ROOT.glob(".coverage.*")))
    session.run("coverage", "report", config, "--fail-under=90")
