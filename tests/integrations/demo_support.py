"""The demo plugin as the kit self-tests use it: import path, entry point and its harness."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.discovery import available_integrations

DEMO_SRC = Path(__file__).resolve().parents[1] / "fixtures" / "demo_integration"


def demo_installed() -> bool:
    """True when the demo plugin is pip-installed (the `plugin` nox session)."""
    try:
        importlib.metadata.distribution("agent-adapter-demo-integration")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def demo() -> Any:
    """The demo plugin module: installed (plugin session) or from the fixture tree."""
    return importlib.import_module("demo_integration")


def use_demo_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the installed plugin (core session): the fixture tree and its entry point."""
    if not demo_installed():
        if importlib.util.find_spec("demo_integration") is None:
            monkeypatch.syspath_prepend(str(DEMO_SRC))
        real = importlib.metadata.entry_points

        def with_demo(*, group: str) -> list[Any]:
            demo_ep = importlib.metadata.EntryPoint("demo", "demo_integration:INTEGRATION", group)
            return [*real(group=group), demo_ep]

        monkeypatch.setattr(importlib.metadata, "entry_points", with_demo)
    available_integrations(refresh=True)


# --8<-- [start:harness]
class DemoHarness:
    """Drives the demo framework for the contract kit."""

    name = "demo"
    edits_supported = True  # the executor applies edited args
    approval = "port"  # the executor's approval port decides

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return demo().INTEGRATION.export_tool(spec, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await native(demo().DemoCall(id=call_id, arguments=args))

    def value(self, raw: Any) -> Any:
        return raw

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, dict) and set(raw) == {"error"}

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await self.call(native, args, call_id="call_demo_approval")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        return native.name, native.description, native.parameters

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return spec.json_schema()  # the demo framework takes the neutral schema


# --8<-- [end:harness]
