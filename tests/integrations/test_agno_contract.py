"""Contract kit C1-C8 for Agno, plus the bridge's own edge cases."""

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("agno")

from agno.agent import Agent
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.tools.function import FunctionCall

from agent_adapter.approval import CallbackApproval, DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalPending, IntegrationError, ToolDefinitionError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations import agno
from agent_adapter.integrations.base import ToolRunner
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations._two_phase import TwoPhaseApproval
from tests.integrations.kit import FakeRunner, contract_tests
from tests.test_examples import _run

pytestmark = pytest.mark.contract


@dataclass
class ScriptedModel(Model):
    """An Agno model that answers from a script, offline."""

    id: str = "scripted"
    name: str = "Scripted"
    provider: str = "scripted"
    replies: list[ModelResponse] = field(default_factory=list)

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self.replies.pop(0)

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


def tool_call(call_id: str, name: str, args: dict[str, Any]) -> ModelResponse:
    function = {"name": name, "arguments": json.dumps(args)}
    return ModelResponse(
        role="assistant",
        content="",
        tool_calls=[{"id": call_id, "type": "function", "function": function}],
    )


def done() -> ModelResponse:
    return ModelResponse(role="assistant", content="done")


class GatePort:
    """Answers with the decision the harness sets for the next approval flow."""

    def __init__(self) -> None:
        self.decision = ApprovalDecision(approved=False)

    async def request(self, request: Any) -> ApprovalDecision:
        return self.decision


SPECS: dict[int, ToolSpec] = {}


def spec_of(native: Any) -> ToolSpec:
    return SPECS[id(native)]


def registry_of(*specs: ToolSpec) -> Registry:
    registry = Registry()
    registry.add_many(specs)
    return registry


def agent(native: Any, *replies: ModelResponse) -> Agent:
    return Agent(model=ScriptedModel(replies=list(replies)), tools=[native], telemetry=False)


class AgnoHarness:
    """Drives Agno's `Agent.arun` / `acontinue_run` with a scripted model."""

    name = "agno"
    edits_supported = False
    approval = "native"
    native_uses_port = True
    port_asks = 1  # resolve_requirements asks; the run takes its token, the executor does not ask

    def __init__(self) -> None:
        self.gate = GatePort()

    def approval_port(self) -> Any:
        return self.gate

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        native = agno.agno_function(spec, executor=executor)
        SPECS[id(native)] = spec
        return native

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return agno.agno_functions(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        output = await agent(native, tool_call(call_id, native.name, args), done()).arun("go")
        (execution,) = output.tools
        return execution

    def value(self, raw: Any) -> Any:
        try:
            return json.loads(raw.result)
        except (TypeError, ValueError):
            return raw.result

    def is_error(self, raw: Any) -> bool:
        if isinstance(raw, Exception) or raw.tool_call_error:
            return True
        value = self.value(raw)
        return isinstance(value, dict) and set(value) == {"error"}

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
        runner = agent(native, tool_call("call_agno_ok", native.name, args), done())
        paused = await runner.arun("go")
        assert paused.is_paused
        await agno.resolve_requirements(paused, port, registry_of(spec_of(native)))
        output = await runner.acontinue_run(run_response=paused)
        (execution,) = output.tools
        return execution

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        return native.name, native.description, native.parameters

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return spec.json_schema()  # Agno takes the neutral JSON schema as given


globals().update((item.__name__, item) for item in contract_tests(AgnoHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(AgnoHarness, runner_factory=Executor)
)


# --- bridge specifics -------------------------------------------------------------------------


async def remove(item: str) -> str:
    """Remove an item."""
    return f"removed {item}"


def test_requires_confirmation_follows_confirm():
    assert agno.agno_function(build_spec(remove, confirm=True)).requires_confirmation is True
    rule = build_spec(remove, confirm=lambda call, ctx: False)
    assert agno.agno_function(rule).requires_confirmation is True
    assert agno.agno_function(build_spec(remove)).requires_confirmation is False


async def test_a_call_outside_a_run_asks_the_executor_port():
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="nope")])
    function = agno.agno_function(
        build_spec(remove, confirm=True), executor=FakeRunner(approval=port)
    )
    call = FunctionCall(function=function, arguments={"item": "a"}, call_id="c1")
    outcome = await call.aexecute()
    assert outcome.status == "success"
    assert json.loads(outcome.result)["error"]["kind"] == "approval-denied"
    assert [(r.tool, r.call_id) for r in port.requests] == [("remove", "c1")]


async def test_a_run_supplies_run_id_and_session():
    runner = FakeRunner()
    native = agno.agno_function(build_spec(remove), executor=runner)
    output = await agent(native, tool_call("c9", "remove", {"item": "x"}), done()).arun("go")
    (inv,) = runner.calls
    assert (inv.ctx.run_id, inv.ctx.call_id) == (output.run_id, "c9")
    assert inv.ctx.thread_id == output.session_id and inv.ctx.approval is None


async def paused_wipe(port: Any, call_id: str = "call_w") -> tuple[Agent, Any, list[str], ToolSpec]:
    """An agent paused on `wipe(item="alpha")`, which records what actually ran."""
    ran: list[str] = []

    async def wipe(item: str) -> str:
        """Wipe an item."""
        ran.append(item)
        return f"wiped {item}"

    spec = build_spec(wipe, confirm=True)
    native = agno.agno_function(spec, executor=Executor(approval=port))
    runner = agent(native, tool_call(call_id, "wipe", {"item": "alpha"}), done(), done())
    paused = await runner.arun("go")
    assert paused.is_paused
    return runner, paused, ran, spec


async def approved_by_two_phase() -> tuple[Agent, Any, list[str], TwoPhaseApproval]:
    port = TwoPhaseApproval()
    runner, paused, ran, spec = await paused_wipe(port)
    with pytest.raises(ApprovalPending):
        await agno.resolve_requirements(paused, port, registry_of(spec))
    await port.decide_all(ApprovalDecision(approved=True, reason="ok"))
    await agno.resolve_requirements(paused, port, registry_of(spec))
    (record,) = await port.store.list()  # consumed now; the call carries a one-use token
    assert record.consumed and record.request.run_id == paused.run_id
    return runner, paused, ran, port


async def test_resolve_with_a_two_phase_port_runs_once_without_a_second_ask():
    runner, paused, ran, port = await approved_by_two_phase()
    await runner.acontinue_run(run_response=paused)
    assert ran == ["alpha"]
    assert len(await port.store.list()) == 1  # the run used the token: no new record


async def test_arguments_edited_after_approval_do_not_run():
    """Regression: confirm() then edited tool_args ran the edited arguments."""
    runner, paused, ran, _ = await approved_by_two_phase()
    paused.tools[0].tool_args = {"item": "PROD"}
    await runner.acontinue_run(run_response=paused)
    assert ran == []


async def test_a_forged_call_id_does_not_run():
    """Regression: a confirmed requirement with another tool_call_id ran."""
    runner, paused, ran, _ = await approved_by_two_phase()
    paused.tools[0].tool_call_id = "forged"
    await runner.acontinue_run(run_response=paused)
    assert ran == []


async def test_a_replayed_confirmation_does_not_run_again():
    """Regression: result=None on a finished call made continue_run execute it again."""
    runner, paused, ran, _ = await approved_by_two_phase()
    output = await runner.acontinue_run(run_response=paused)
    (execution,) = output.tools
    execution.requires_confirmation, execution.confirmed, execution.result = True, True, None
    await runner.acontinue_run(run_response=output)
    assert ran == ["alpha"]


async def test_a_store_less_port_is_asked_again_with_the_arguments_that_run():
    def only_alpha(request: Any) -> ApprovalDecision:
        return ApprovalDecision(approved=request.arguments == {"item": "alpha"})

    port = RecordingApproval(CallbackApproval(only_alpha))
    runner, paused, ran, spec = await paused_wipe(port)
    await agno.resolve_requirements(paused, port, registry_of(spec))
    paused.tools[0].tool_args = {"item": "PROD"}
    await runner.acontinue_run(run_response=paused)
    assert ran == []
    assert [r.arguments for r in port.requests] == [{"item": "alpha"}, {"item": "PROD"}]


async def test_resolve_rejects_with_the_reason_and_skips_other_requirements():
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="not today")])
    _, paused, _, spec = await paused_wipe(port)
    await agno.resolve_requirements(paused, port, registry_of(spec))
    (requirement,) = paused.requirements
    assert requirement.confirmation is False and requirement.confirmation_note == "not today"
    await agno.resolve_requirements(SimpleNamespace(run_id="r"), port, registry_of(spec))


def test_an_unsupported_agno_release_is_refused(monkeypatch):
    import agno as agno_pkg

    monkeypatch.setattr(agno_pkg, "__version__", "4.0.0")
    with pytest.raises(IntegrationError, match="outside the supported range"):
        agno.agno_function(build_spec(remove))


def test_bind_context_without_a_run():
    ctx = agno.INTEGRATION.bind_context(object())
    assert (ctx.run_id, ctx.call_id, ctx.thread_id, ctx.approval) == ("", "", "", None)


def test_an_argument_named_fc_is_rejected():
    async def clash(fc: str) -> str:
        """Clash with Agno's injected parameter."""
        return fc

    with pytest.raises(ToolDefinitionError, match="reserves the argument name 'fc'"):
        agno.agno_function(build_spec(clash))


def test_the_example_runs(capsys):
    assert _run("agno_tools.py", capsys).splitlines() == [
        "[('shipping_rate', None, '5.0')]",
        "[('send_invoice', True, 'invoiced cus_1024 for 49.50 USD')]",
        "[('send_invoice', False, None)]",
    ]


async def test_a_pre_merge_deferred_port_decided_one_call_at_a_time_converges():
    """Regression: with a store-backed port, resolving c1, c2 one by one ping-ponged pending."""
    ran: list[str] = []

    async def wipe(item: str) -> str:
        """Wipe an item."""
        ran.append(item)
        return f"wiped {item}"

    spec = build_spec(wipe, confirm=True)
    port = DeferredApproval(MemoryApprovalStore())
    native = agno.agno_function(spec, executor=Executor(approval=port))
    two = [
        {"id": cid, "type": "function", "function": {"name": "wipe", "arguments": json.dumps(a)}}
        for cid, a in (("c1", {"item": "a"}), ("c2", {"item": "b"}))
    ]
    runner = agent(native, ModelResponse(role="assistant", content="", tool_calls=two), done())
    paused = await runner.arun("go")
    for call_id in ("c1", "c2"):
        with pytest.raises(ApprovalPending):
            await agno.resolve_requirements(paused, port, registry_of(spec))
        record = next(r for r in await port.store.list() if r.request.call_id == call_id)
        await port.store.decide(record.approval_id, ApprovalDecision(approved=True))
    await agno.resolve_requirements(paused, port, registry_of(spec))
    await runner.acontinue_run(run_response=paused)
    assert sorted(ran) == ["a", "b"] and len(await port.store.list()) == 2
