"""Optional-feature checks for `agent-adapter doctor`: one `Feature` row per extra backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_adapter.cli.findings import Finding, Level

if TYPE_CHECKING:
    from agent_adapter.sandbox.docker import DockerStatus

Version = Callable[[str], str | None]  # distribution -> installed version, or None


@dataclass(frozen=True)
class Feature:
    """An optional feature: the distributions it needs, all of `needs` and one of `any_of`.

    `extra` names the `agent-adapter[...]` extra that installs it, when there is one;
    `partial` is the level shown when only some of it is installed.
    """

    topic: str
    needs: tuple[str, ...]
    extra: str = ""
    any_of: tuple[str, ...] = ()
    purpose: str = ""
    partial: Level = "warn"


# Later features add a row here; `diagnose()` runs each one.
FEATURES: tuple[Feature, ...] = (
    Feature(
        "guard presidio",
        ("presidio-analyzer", "presidio-anonymizer"),
        extra="presidio",
        purpose="PII detection for guardrails",
    ),
    Feature(
        "guard nemo",
        ("nemoguardrails",),
        extra="nemo",
        purpose="NeMo Guardrails rails as a guard",
    ),
    Feature(
        "otel export",
        ("opentelemetry-api", "opentelemetry-sdk"),
        extra="otel",
        any_of=(
            "opentelemetry-exporter-otlp-proto-http",
            "opentelemetry-exporter-otlp-proto-grpc",
        ),
        purpose="OTLP spans for Langfuse, Phoenix and other OpenTelemetry backends",
        partial="info",  # other libraries often install opentelemetry-api alone
    ),
    Feature("memory mem0", ("mem0ai",), extra="mem0", purpose="Mem0 as a memory backend"),
    Feature("memory zep", ("zep-cloud",), extra="zep", purpose="Zep as a memory backend"),
    Feature(
        "memory graphiti",
        ("graphiti-core",),
        extra="graphiti",
        purpose="a Graphiti knowledge graph as a memory backend",
    ),
    Feature(
        "retrieval llamaindex",
        ("llama-index-core",),
        extra="llamaindex",
        purpose="LlamaIndexRetriever over an index or retriever",
    ),
    Feature(
        "retrieval langchain",
        ("langchain-core",),
        extra="langchain",
        purpose="LangChainRetriever over a LangChain retriever",
    ),
    Feature(
        "budget redis",
        ("redis",),
        extra="redis",
        purpose="RedisBudgetStore: budgets shared by every worker and host",
    ),
    Feature(
        "budget sqlalchemy",
        ("sqlalchemy",),
        extra="sqlalchemy",
        any_of=("asyncpg", "aiosqlite"),
        purpose="SQLAlchemyBudgetStore: budgets in Postgres or SQLite",
    ),
    Feature(
        "checkpoints redis",
        ("redis",),
        extra="redis",
        purpose="RedisCheckpointStore for durable runs",
    ),
    Feature(
        "checkpoints sqlalchemy",
        ("sqlalchemy",),
        extra="sqlalchemy",
        any_of=("asyncpg", "aiosqlite"),
        purpose="SQLAlchemyCheckpointStore for durable runs",
    ),
    Feature("temporal", ("temporalio",), extra="temporal", purpose="flows as Temporal workflows"),
    Feature(
        "dbos",
        ("dbos",),
        extra="dbos",
        purpose="flows as DBOS workflows (SQLite or Postgres)",
    ),
)


def _hint(feature: Feature, missing: Iterable[str], extras: Mapping[str, object]) -> str:
    """The install command: the extra when this build has it, else the distributions."""
    names = sorted(missing)
    if feature.extra and feature.extra in extras and not set(names) & set(feature.any_of):
        return f'pip install "agent-adapter[{feature.extra}]"'
    return "pip install " + " ".join(names)


def check_feature(feature: Feature, extras: Mapping[str, object], version: Version) -> Finding:
    """`ok` when ready, `info` when absent, `feature.partial` when half installed."""
    found = {dist: version(dist) for dist in (*feature.needs, *feature.any_of)}
    missing = [dist for dist in feature.needs if found[dist] is None]
    exporters = [dist for dist in feature.any_of if found[dist] is not None]
    if feature.any_of and not exporters:
        missing.append(feature.any_of[0])
    if not missing:
        ready = (*feature.needs, *exporters)
        return Finding("ok", feature.topic, "ready " + ", ".join(f"{d} {found[d]}" for d in ready))
    why = f" ({feature.purpose})" if feature.purpose else ""
    hint = _hint(feature, missing, extras)
    if len(missing) == len(feature.needs) + (1 if feature.any_of else 0):
        return Finding("info", feature.topic, f"not installed{why}: {hint}")
    gaps = ", ".join(sorted(missing))
    return Finding(feature.partial, feature.topic, f"{gaps} missing{why}: {hint}")


def feature_findings(
    extras: Mapping[str, object], features: Sequence[Feature], version: Version
) -> list[Finding]:
    """One finding per optional feature, in table order."""
    return [check_feature(feature, extras, version) for feature in features]


def docker_finding(status: DockerStatus | None = None) -> Finding:
    """`ok` with versions when Docker runs Linux containers here; `info` with the reason if not."""
    from agent_adapter.sandbox.docker import docker_status

    found = status if status is not None else docker_status(timeout_s=5.0)
    if found.available:
        message = f"ready docker {found.client_version} (daemon {found.server_version})"
        return Finding("ok", "sandbox docker", message)
    why = f"not available ({found.reason}); DockerSandbox needs Docker with Linux containers"
    return Finding("info", "sandbox docker", why)
