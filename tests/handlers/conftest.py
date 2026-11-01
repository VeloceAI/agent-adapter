"""A small registry and an executor that parks confirm tools in a memory store."""

from dataclasses import dataclass, field

import pytest

from agent_adapter import DeferredApproval, Executor, MemoryApprovalStore, Registry

SECRET = "s3cr3t-value"


@dataclass
class Tools:
    """The test tools, and a log of what the side-effecting one did."""

    registry: Registry
    filed: list[tuple[str, int]] = field(default_factory=list)


@pytest.fixture
def tools() -> Tools:
    registry = Registry()
    made = Tools(registry)

    @registry.tool(effects="read_only")
    def echo(text: str) -> dict[str, object]:
        """Echo the text back."""
        return {"text": text, "pair": (1, 2)}

    @registry.tool(confirm=True, effects="destructive")
    def file_return(pan: str, year: int) -> str:
        """File a return (needs approval)."""
        made.filed.append((pan, year))
        return f"filed {pan} {year}"

    @registry.tool
    def broken(token: str) -> str:
        """Always fails."""
        raise RuntimeError(f"boom {token}")

    return made


@pytest.fixture
def store() -> MemoryApprovalStore:
    return MemoryApprovalStore()


@pytest.fixture
def executor(tools: Tools, store: MemoryApprovalStore) -> Executor:
    return Executor(tools.registry, approval=DeferredApproval(store))
