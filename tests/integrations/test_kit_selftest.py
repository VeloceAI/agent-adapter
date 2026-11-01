"""Kit self-test: C1-C8 against the demo plugin (fake and real executor) and the kit's options."""

import asyncio
from typing import Any

import pytest

from agent_adapter.core.errors import ApprovalDenied
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.exec.defaults import default_executor, set_default_executor
from agent_adapter.integrations.base import ToolRunner, model_visible_error
from agent_adapter.integrations.discovery import available_integrations
from agent_adapter.runtime.sync import run_sync
from tests.fakes.approval import RecordingApproval
from tests.fakes.hooks import RecordingHook
from tests.integrations.demo_support import DemoHarness, demo, use_demo_plugin
from tests.integrations.kit import CHECKS, FakeRunner, Probe, contract_tests, harness_session


@pytest.fixture(autouse=True)
def demo_plugin(monkeypatch):
    use_demo_plugin(monkeypatch)
    yield
    monkeypatch.undo()
    available_integrations(refresh=True)


# --8<-- [start:install]
globals().update((item.__name__, item) for item in contract_tests(DemoHarness))
# --8<-- [end:install]

# The same checks through the real executor (its constructor matches the runner factory).
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(DemoHarness, runner_factory=Executor)
)


async def test_demo_export_without_an_executor_runs_through_the_default_executor():
    previous, hook = default_executor(), RecordingHook()
    set_default_executor(Executor(hooks=[hook]))
    try:
        native = demo().INTEGRATION.export_tool(Probe().add)
        assert native.runner is default_executor()
        assert await native(demo().DemoCall(id="call_default", arguments={"a": 4})) == 6
    finally:
        set_default_executor(previous)
    assert hook.phases() == ["before_tool", "after_tool"]


def test_contract_tests_returns_the_fixture_then_c1_to_c8():
    fixture, *tests = contract_tests(DemoHarness)
    assert fixture.name.startswith("aa_kit_harness_")
    assert [t.__name__.split("_")[1] for t in tests] == [f"c{i}" for i in range(1, 9)]


# --- managed harness: entered once per contract_tests call ---------------------------------


class ManagedDemoHarness(DemoHarness):
    """An async context manager harness (like MCP / ADK / LangGraph sessions)."""

    entries = 0
    exits = 0

    def __init__(self) -> None:
        self.inside = False

    async def __aenter__(self) -> "ManagedDemoHarness":
        type(self).entries += 1
        assert type(self).entries - type(self).exits == 1, "entered twice"
        self.inside = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        type(self).exits += 1
        self.inside = False

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        assert self.inside, "used outside its session"
        return super().export(spec, executor)


globals().update((f"{item.__name__}_managed", item) for item in contract_tests(ManagedDemoHarness))


async def test_harness_session_enters_and_exits_a_managed_harness_once():
    class Counting(ManagedDemoHarness):
        entries = 0
        exits = 0

    async with harness_session(Counting) as harness:
        assert isinstance(harness, Counting) and harness.inside
        assert (Counting.entries, Counting.exits) == (1, 0)
    assert (Counting.entries, Counting.exits) == (1, 1)


async def test_harness_session_exits_when_the_block_fails():
    class Counting(ManagedDemoHarness):
        entries = 0
        exits = 0

    with pytest.raises(RuntimeError):
        async with harness_session(Counting):
            raise RuntimeError("check failed")
    assert Counting.exits == 1


async def test_harness_session_yields_a_plain_harness_unchanged():
    async with harness_session(DemoHarness) as harness:
        assert type(harness) is DemoHarness


async def test_harness_session_keeps_the_harness_when_aenter_returns_none():
    class ReturnsNone(ManagedDemoHarness):
        entries = 0
        exits = 0

        async def __aenter__(self) -> None:  # type: ignore[override]
            await super().__aenter__()

    async with harness_session(ReturnsNone) as harness:
        assert isinstance(harness, ReturnsNone)


# --- optional harness features ---------------------------------------------------------------


class NativeDemoHarness(DemoHarness):
    """The demo framework with its own approval gate: the decision travels on the call."""

    approval = "native"

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        if not decision.approved:  # the framework rejects without calling the tool
            return model_visible_error(ApprovalDenied(native.name, reason=decision.reason))
        call = demo().DemoCall(id="call_demo_native", arguments=args, approval=decision)
        return await native(call)


class GatePort:
    """An ApprovalPort answering with whatever decision the framework's gate holds."""

    def __init__(self) -> None:
        self.decision: ApprovalDecision | None = None

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        assert self.decision is not None
        return self.decision


class PortGateHarness(DemoHarness):
    """Native approval implemented as an ApprovalPort (as LangGraph's `interrupt` port is)."""

    approval = "native"
    native_uses_port = True

    def __init__(self) -> None:
        self.gate = GatePort()
        self.ports: list[RecordingApproval | None] = []

    def approval_port(self) -> ApprovalPort:
        return self.gate

    async def approve_flow(
        self,
        native: Any,
        args: dict[str, Any],
        decision: ApprovalDecision,
        *,
        port: RecordingApproval | None = None,
    ) -> Any:
        self.ports.append(port)
        self.gate.decision = decision  # the resume value
        return await self.call(native, args, call_id="call_demo_gate")


async def test_c6_native_approval_without_a_port():
    _, check = CHECKS["C6"]
    await check(NativeDemoHarness(), FakeRunner)


async def test_c6_native_approval_through_the_harness_port():
    _, check = CHECKS["C6"]
    harness = PortGateHarness()
    await check(harness, FakeRunner)
    assert len(harness.ports) == 3
    assert all(p is not None and p.port is harness.gate for p in harness.ports)


async def test_c6_catches_a_native_gate_that_skips_its_port():
    class SkipsPort(PortGateHarness):
        async def approve_flow(self, native, args, decision, *, port=None):
            if not decision.approved:
                return model_visible_error(ApprovalDenied(native.name))
            return await native(demo().DemoCall(id="c", arguments=args, approval=decision))

    _, check = CHECKS["C6"]
    with pytest.raises(AssertionError, match="never asked the approval port"):
        await check(SkipsPort(), FakeRunner)


async def test_c6_catches_a_native_uses_port_harness_without_a_port():
    class NoPort(PortGateHarness):
        def approval_port(self) -> None:  # type: ignore[override]
            return None

        async def approve_flow(self, native, args, decision, **kwargs):
            return await NativeDemoHarness.approve_flow(self, native, args, decision)

    _, check = CHECKS["C6"]
    with pytest.raises(AssertionError, match="needs approval_port"):
        await check(NoPort(), FakeRunner)


class NoCallIdHarness(DemoHarness):
    native_call_ids = False

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await native(demo().DemoCall(id="", arguments=args))


async def test_c5_accepts_a_generated_call_id_when_the_framework_has_none():
    _, check = CHECKS["C5"]
    await check(NoCallIdHarness(), FakeRunner)


async def test_c4_runs_against_a_policy_runner():
    _, check = CHECKS["C4"]
    await check(DemoHarness(), FakeRunner)


class ExportAllHarness(DemoHarness):
    """Also checks a module-level `demo_tools(registry, executor)` style exporter."""

    def __init__(self, *, ignore_targets: bool = False) -> None:
        self.ignore_targets = ignore_targets

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        specs = registry if self.ignore_targets else registry.for_target(self.name)
        return [self.export(spec, executor) for spec in specs]


async def test_c1_checks_export_all_against_targets():
    _, check = CHECKS["C1"]
    await check(ExportAllHarness(), FakeRunner)
    with pytest.raises(AssertionError, match="export_all ignored targets"):
        await check(ExportAllHarness(ignore_targets=True), FakeRunner)


async def test_c1_catches_export_tools_that_ignore_targets(monkeypatch):
    integration = demo().INTEGRATION
    monkeypatch.setattr(
        integration,
        "export_tools",
        lambda registry, *, executor=None: [
            integration.export_tool(s, executor=executor) for s in registry
        ],
    )
    _, check = CHECKS["C1"]
    with pytest.raises(AssertionError, match="export_tools ignored targets"):
        await check(DemoHarness(), FakeRunner)


# --- sync bridge: the tool runs on another thread's loop -------------------------------------


class SyncBridgeHarness(DemoHarness):
    """A sync-only framework: the call runs `run_sync` in a worker thread."""

    cancels_tool = False

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        call = demo().DemoCall(id=call_id, arguments=args)
        return await asyncio.to_thread(lambda: run_sync(lambda: native(call)))


async def test_c7_sync_bridge_ends_and_its_thread_exits():
    _, check = CHECKS["C7"]
    await check(SyncBridgeHarness(), FakeRunner)
    loop = asyncio.get_running_loop()
    await asyncio.wait_for(loop.shutdown_default_executor(), timeout=5)  # the tool thread ended


async def test_c7_catches_a_sync_bridge_claiming_it_cancels_the_tool():
    class Claims(SyncBridgeHarness):
        cancels_tool = True

    _, check = CHECKS["C7"]
    with pytest.raises(AssertionError, match="tool kept running"):
        await check(Claims(), FakeRunner)
    loop = asyncio.get_running_loop()
    await asyncio.wait_for(loop.shutdown_default_executor(), timeout=5)


async def test_c2_runs_through_a_sync_bridge():
    _, check = CHECKS["C2"]
    await check(SyncBridgeHarness(), FakeRunner)


@pytest.mark.parametrize(("asks", "passes"), [(1, True), (2, False), (None, True)])
async def test_c6_port_asks_pins_the_number_of_native_port_requests(asks, passes):
    class Pinned(PortGateHarness):
        port_asks = asks

    _, check = CHECKS["C6"]
    if passes:
        await check(Pinned(), FakeRunner)
    else:
        with pytest.raises(AssertionError, match="C6: port asked 1 times, expected 2"):
            await check(Pinned(), FakeRunner)
