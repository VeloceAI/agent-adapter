"""Kit self-test, negative side: every check fails on a faulty bridge; FakeRunner and Probe."""

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalDenied, IntegrationError, ToolArgsInvalid
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.hooks.base import ShortCircuit
from agent_adapter.integrations.base import ToolRunner, model_visible_error
from agent_adapter.integrations.discovery import available_integrations
from tests.fakes.approval import ScriptedApproval
from tests.fakes.hooks import RecordingHook
from tests.integrations.demo_support import DemoHarness, demo, use_demo_plugin
from tests.integrations.kit import CHECKS, FakeRunner, Probe


@pytest.fixture(autouse=True)
def demo_plugin(monkeypatch):
    use_demo_plugin(monkeypatch)
    yield
    monkeypatch.undo()
    available_integrations(refresh=True)


# --- faulty bridges: each check must fail -------------------------------------------------


DETACHED: set[asyncio.Future[Any]] = set()


@dataclass
class Faulty:
    """A broken bridge around a working demo tool."""

    tool: Any
    fault: str

    async def __call__(self, call: Any) -> Any:
        spec, runner = self.tool.spec, self.tool.runner
        if self.fault == "wrong_value":
            return "wrong"
        if self.fault == "leak_validation":
            spec.args_model.model_validate(dict(call.arguments))
        if self.fault == "bypass_runner":
            try:
                return await spec.invoke(call.arguments, ToolContext(call_id=call.id))
            except ToolArgsInvalid as exc:
                return model_visible_error(exc)
        if self.fault == "drop_call_id":
            return await runner.run_spec(spec, call.arguments)
        if self.fault == "raise_errors":
            return await runner.run_spec(spec, call.arguments, call_id=call.id)
        if self.fault == "swallow_cancel":
            try:
                return await self.tool(call)
            except BaseException as exc:
                return model_visible_error(exc)
        if self.fault == "detach" and spec.name == "kit_wait":  # returns, tool keeps running
            DETACHED.add(asyncio.ensure_future(self.tool(call)))
            return "detached"
        if self.fault == "never_start" and spec.name == "kit_wait":
            await asyncio.sleep(3600)
        return await self.tool(call)


class FaultyHarness(DemoHarness):
    def __init__(self, fault: str) -> None:
        self.fault = fault
        if fault == "unregistered":
            self.name = "not_registered"

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return Faulty(super().export(spec, executor), self.fault)

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        name, description, schema = super().describe(getattr(native, "tool", native))
        if self.fault == "bad_description":
            return name, description + "!", schema
        if self.fault == "bad_schema":
            return name, description, {**schema, "properties": {}}
        return name, description, schema


@pytest.mark.parametrize(
    ("check_id", "fault", "message"),
    [
        ("C1", "bad_description", "description differs"),
        ("C1", "bad_schema", "schema differs"),
        ("C2", "wrong_value", "!= "),
        ("C3", "leak_validation", "raw ValidationError"),
        ("C3", "raise_errors", "not reported as an error"),
        ("C4", "bypass_runner", "hooks saw"),
        ("C5", "drop_call_id", "tool saw"),
        ("C6", "bypass_runner", "denied call ran"),
        ("C7", "swallow_cancel", "gave result"),
        ("C7", "detach", "ended before the tool started|tool kept running"),
    ],
)
async def test_each_check_catches_a_faulty_bridge(check_id, fault, message):
    _, check = CHECKS[check_id]
    with pytest.raises(AssertionError, match=f"^{check_id}: .*({message})"):
        await check(FaultyHarness(fault), FakeRunner)


async def test_c8_catches_an_unregistered_integration():
    _, check = CHECKS["C8"]
    with pytest.raises(IntegrationError, match="not_registered"):
        await check(FaultyHarness("unregistered"), FakeRunner)


async def test_c7_reports_a_tool_that_never_starts(monkeypatch):
    monkeypatch.setattr("tests.integrations.kit.WAIT_LIMIT_S", 0.05)
    _, check = CHECKS["C7"]
    with pytest.raises(AssertionError, match="C7: the tool never started"):
        await check(FaultyHarness("never_start"), FakeRunner)


async def test_c6_catches_ignored_edits():
    class IgnoresEdits(FakeRunner):
        async def _approve(self, inv: Any, validated: Any) -> Any:
            await super()._approve(inv, validated)
            return validated

    _, check = CHECKS["C6"]
    with pytest.raises(AssertionError, match="C6"):
        await check(DemoHarness(), IgnoresEdits)


# --- FakeRunner ---------------------------------------------------------------------------


async def test_fake_runner_without_a_port_denies_confirm_tools():
    probe = Probe()
    with pytest.raises(ApprovalDenied, match="no approval port configured"):
        await FakeRunner().run_spec(probe.remove, {"item": "a"})
    assert probe.executed == []


async def test_fake_runner_uses_ctx_approval_before_the_port():
    probe, port = Probe(), ScriptedApproval([])
    ctx = ToolContext(approval=ApprovalDecision(approved=True))
    assert await FakeRunner(approval=port).run_spec(probe.remove, {"item": "a"}, ctx) == "removed a"
    assert port.requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_fake_runner_evaluates_callable_confirm_rules(asynchronous):
    def rule(call: Any, ctx: ToolContext) -> bool:
        return call.arguments["item"] == "risky"

    async def async_rule(call: Any, ctx: ToolContext) -> bool:
        return rule(call, ctx)

    def remove(item: str) -> str:
        """Remove an item."""
        return item

    spec = build_spec(remove, confirm=async_rule if asynchronous else rule)
    runner = FakeRunner()
    assert await runner.run_spec(spec, {"item": "safe"}) == "safe"
    with pytest.raises(ApprovalDenied):
        await runner.run_spec(spec, {"item": "risky"})


async def test_fake_runner_short_circuit_still_runs_after_tool():
    probe = Probe()
    hook = RecordingHook(returns={"before_tool": ShortCircuit(41)})
    assert await FakeRunner(hooks=[hook]).run_spec(probe.add, {"a": 1}) == 41
    assert probe.executed == []
    assert hook.phases() == ["before_tool", "after_tool"]
    result = hook.calls[1][1][1]
    assert (result.value, result.attempts, result.status) == (41, 0, "ok")


async def test_fake_runner_keeps_call_ids():
    probe, runner = Probe(), FakeRunner()
    await runner.run_spec(probe.add, {"a": 1}, ToolContext(run_id="run_x", call_id="call_ctx"))
    await runner.run_spec(probe.add, {"a": 1}, call_id="call_arg")
    first = runner.calls[0].ctx
    assert (first.run_id, first.call_id) == ("run_x", "call_ctx")
    assert runner.calls[1].ctx.call_id == "call_arg"
    assert runner.calls[1].ctx.run_id.startswith("run_")


async def test_fake_runner_reraises_a_tool_timeout_error_that_is_not_its_own():
    async def flaky() -> str:
        """Raise TimeoutError itself."""
        raise TimeoutError("upstream")

    with pytest.raises(TimeoutError, match="upstream"):
        await FakeRunner().run_spec(build_spec(flaky, timeout=5.0), {})


async def test_probe_wait_is_bounded(monkeypatch):
    monkeypatch.setattr("tests.integrations._kit_support.WAIT_LIMIT_S", 0.02)
    with pytest.raises(TimeoutError):
        await Probe().wait.invoke({"label": "x"})


async def test_demo_tool_passes_framework_approval_into_the_context():
    probe = Probe()
    tool = demo().INTEGRATION.export_tool(probe.remove, executor=FakeRunner())
    decision = ApprovalDecision(approved=True)
    call = demo().DemoCall(id="c1", arguments={"item": "x"}, approval=decision)
    assert await tool(call) == "removed x"
    assert await tool(demo().DemoCall(id="c2", arguments={"item": "y"})) == {
        "error": {
            "kind": "approval-denied",
            "message": "approval denied for tool 'kit_remove': no approval port configured",
        }
    }


class ShrugsOffCancel(DemoHarness):
    """Ignores the first cancel of the call; only a second cancel ends it."""

    def __init__(self) -> None:
        self.tasks: list[asyncio.Task[Any]] = []

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        task = asyncio.current_task()
        assert task is not None
        self.tasks.append(task)
        inner = asyncio.ensure_future(super().call(native, args, call_id=call_id))
        try:
            return await asyncio.shield(inner)
        except asyncio.CancelledError:
            pass  # the bug under test: cancellation swallowed
        await asyncio.sleep(3600)
        return None


async def test_c7_failure_leaves_no_pending_call_on_the_loop(monkeypatch):
    monkeypatch.setattr("tests.integrations.kit.WAIT_LIMIT_S", 0.2)
    harness = ShrugsOffCancel()
    _, check = CHECKS["C7"]
    with pytest.raises(AssertionError, match="C7: the cancelled call did not end"):
        await check(harness, FakeRunner)
    assert harness.tasks and all(task.done() for task in harness.tasks)
