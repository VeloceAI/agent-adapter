"""Contract kit C1-C8 for Google ADK tools, through `LlmAgent` with a scripted `BaseLlm`."""

import asyncio
from typing import Any

import pytest

pytest.importorskip("google.adk")

from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations import adk
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.schema.transforms import to_gemini_parameters
from tests.fakes.approval import RecordingApproval
from tests.integrations._kit_support import Probe, wait_until
from tests.integrations.adk.support import (
    AgentRun,
    confirmations,
    function_call,
    registry_of,
    responses,
    run_tool,
    text,
    user,
)
from tests.integrations.kit import contract_tests

pytestmark = pytest.mark.contract


class GatePort:
    """Answers with the decision the harness sets for the next approval flow."""

    def __init__(self) -> None:
        self.decision = ApprovalDecision(approved=False)

    async def request(self, request: Any) -> ApprovalDecision:
        return self.decision


class AdkHarness:
    """Drives ADK's `LlmAgent` tool loop; confirmations go through `resolve_confirmations`."""

    name = "adk"
    edits_supported = True
    approval = "native"
    native_uses_port = True
    port_asks = 1  # resolve_confirmations asks; the resumed call takes its token

    def __init__(self) -> None:
        self.gate = GatePort()

    def approval_port(self) -> Any:
        return self.gate

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return adk.adk_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return adk.adk_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        run = AgentRun([native], function_call(native.name, args, call_id), text("done"))
        (response,) = responses(await run.turn(user("go")), call_id)
        return response

    def value(self, raw: Any) -> Any:
        return raw["result"] if set(raw) == {"result"} else raw

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, Exception) or set(raw) == {"error"}

    async def approve_flow(
        self,
        native: Any,
        args: dict[str, Any],
        decision: ApprovalDecision,
        *,
        port: RecordingApproval | None = None,
    ) -> Any:
        assert port is not None
        self.gate.decision = decision
        run = AgentRun([native], function_call(native.name, args, "call_adk_ok"), text("done"))
        paused = await run.turn(user("go"))
        assert len(confirmations(paused)) == 1
        message = await adk.resolve_confirmations(paused, port, registry_of(native.spec))
        *_, response = responses(await run.turn(message), "call_adk_ok")
        return response

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        declaration = native._get_declaration()
        return declaration.name, declaration.description, declaration.parameters_json_schema

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return to_gemini_parameters(spec.json_schema())


class AdkToolHarness(AdkHarness):
    """Calls each tool the way ADK's tool step does: `run_async` with a real `ToolContext`."""

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await run_tool(native, args, call_id)

    async def approve_flow(self, *args: Any, **kwargs: Any) -> Any:
        response = await super().approve_flow(*args, **kwargs)
        return response["result"] if set(response) == {"result"} else response  # ADK's wrap

    def value(self, raw: Any) -> Any:
        return raw

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, Exception) or (isinstance(raw, dict) and set(raw) == {"error"})


globals().update((item.__name__, item) for item in contract_tests(AdkToolHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(AdkToolHarness, runner_factory=Executor)
)
# ADK's Runner runs the agent in a task of its own and ends the turn quietly when that task is
# cancelled, so a CancelledError raised inside a tool cannot reach the caller through
# `Runner.run_async`; `AdkToolHarness` covers C7 on ADK's own tool step.
globals().update(
    (f"{item.__name__}_agent", item)
    for item in contract_tests(AdkHarness, runner_factory=Executor)
    if not item.__name__.startswith("test_c7_")
)


async def test_a_cancelled_agent_run_cancels_the_tool():
    """From outside, cancelling `Runner.run_async` reaches the tool (C7, first half)."""
    probe = Probe()
    native = adk.adk_tool(probe.wait, executor=Executor())
    run = AgentRun([native], function_call(native.name, {"label": "x"}, "call_c7"), text("done"))

    task = asyncio.ensure_future(run.turn(user("go")))
    try:
        assert await wait_until(probe.started), "the tool never started"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await wait_until(probe.cancelled, 2.0), "the tool kept running"
    finally:
        probe.released.set()
