"""Contract kit C1-C8 for AG2, plus the approval gate's forged, replayed and cross-run cases."""

import copy
import json
from typing import Any

import pytest

pytest.importorskip("ag2")

from ag2 import Agent
from ag2.context import ConversationContext
from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent, ToolResultsEvent
from ag2.middleware.builtin.tools import approval_required
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import IntegrationError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations import ag2
from agent_adapter.integrations.base import ToolRunner
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations.kit import FakeRunner, contract_tests
from tests.test_examples import _run

pytestmark = pytest.mark.contract


class GatePort:
    """Answers with the decision the harness sets for the next approval flow."""

    def __init__(self) -> None:
        self.decision = ApprovalDecision(approved=False)

    async def request(self, request: Any) -> ApprovalDecision:
        return self.decision


def call_event(call_id: str, name: str, args: Any) -> ToolCallEvent:
    text = args if isinstance(args, str) else json.dumps(args)
    return ToolCallEvent(name=name, arguments=text, id=call_id)


async def ask(native: Any, *calls: ToolCallEvent, hitl: Any = None) -> list[ToolResultEvent]:
    """Run one agent turn over `calls`; the tool result events, in order."""
    config = TestConfig(*calls, "done", raise_tool_errors=False)
    reply = await Agent("t", config=config, tools=[native], hitl_hook=hitl).ask("go")
    events = await reply.history.get_events()
    return [r for e in events if isinstance(e, ToolResultsEvent) for r in e.results]


def shown(event: ToolResultEvent) -> Any:
    """What the model sees: a data part's data, else text (JSON text decoded)."""
    part = event.result.parts[0]
    if hasattr(part, "data"):
        return part.data
    try:
        return json.loads(part.content)
    except ValueError:
        return part.content


class AG2Harness:
    """Drives `Agent.ask` with AG2's scripted `TestConfig` model."""

    name = "ag2"
    edits_supported = False
    approval = "native"
    native_uses_port = True
    port_asks = 1  # the gate asks; the call takes its token, the executor does not ask

    def __init__(self) -> None:
        self.gate = GatePort()

    def approval_port(self) -> Any:
        return self.gate

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return ag2.ag2_tool(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return ag2.ag2_tools(registry, executor=executor)

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        (result,) = await ask(native, call_event(call_id, native.name, args))
        return result

    def value(self, raw: Any) -> Any:
        return shown(raw)

    def is_error(self, raw: Any) -> bool:
        if isinstance(raw, Exception | ToolErrorEvent):
            return True
        value = shown(raw)
        return isinstance(value, dict) and set(value) == {"error"}

    async def approve_flow(
        self,
        native: Any,
        args: dict[str, Any],
        decision: ApprovalDecision,
        *,
        port: RecordingApproval | None = None,
    ) -> Any:
        self.gate.decision = decision
        gated = ag2.ag2_tool(native.spec, executor=native.runner, approval=port)
        return await self.call(gated, args, call_id="call_ag2_ok")

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        function = native.schema.function
        return function.name, function.description, function.parameters

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        schema = spec.json_schema()
        schema.pop("title", None)  # AG2's FunctionDefinition drops it
        return schema


globals().update((item.__name__, item) for item in contract_tests(AG2Harness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(AG2Harness, runner_factory=Executor)
)


# --- bridge specifics -------------------------------------------------------------------------


def wipe_tool(port: Any, executor_port: Any = None) -> tuple[Any, list[str]]:
    """A gated `wipe` whose executor asks `executor_port` (default: a port that denies)."""
    ran: list[str] = []

    def wipe(item: str) -> str:
        """Wipe an item."""
        ran.append(item)
        return f"wiped {item}"

    denying = executor_port or ScriptedApproval([ApprovalDecision(approved=False, reason="no")])
    executor = Executor(approval=denying)
    return ag2.ag2_tool(build_spec(wipe, confirm=True), executor=executor, approval=port), ran


def context() -> ConversationContext:
    return ConversationContext(stream=MemoryStream())


async def test_an_approved_call_runs_once_without_asking_the_executor():
    gate = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=True)]))
    executor_port = RecordingApproval(ScriptedApproval([]))
    native, ran = wipe_tool(gate, executor_port)
    (result,) = await ask(native, call_event("c1", "wipe", {"item": "a"}))
    assert shown(result) == "wiped a" and ran == ["a"]
    (request,) = gate.requests
    assert (request.call_id, request.arguments) == ("c1", {"item": "a"}) and request.run_id
    assert executor_port.requests == [] and len(native.tokens) == 0


async def test_a_forged_call_id_does_not_take_the_token():
    """A token issued for one call id is not taken by a call with another id."""
    native, ran = wipe_tool(ScriptedApproval([]))
    ctx = context()
    run_id = ag2._run_id(ctx)
    native.tokens.issue(
        "wipe", {"item": "a"}, ApprovalDecision(approved=True), call_id="c1", run_id=run_id
    )
    result = await native(call_event("forged", "wipe", {"item": "a"}), ctx)
    assert shown(result)["error"]["kind"] == "approval-denied" and ran == []


async def test_a_token_from_another_run_is_not_taken():
    native, ran = wipe_tool(ScriptedApproval([]))
    other = ag2._run_id(context())
    native.tokens.issue(
        "wipe", {"item": "a"}, ApprovalDecision(approved=True), call_id="c1", run_id=other
    )
    result = await native(call_event("c1", "wipe", {"item": "a"}), context())
    assert shown(result)["error"]["kind"] == "approval-denied" and ran == []


async def test_a_replayed_call_does_not_run_again():
    """The token is used up by the approved call; replaying the event asks the executor."""
    gate = ScriptedApproval([ApprovalDecision(approved=True)])
    native, ran = wipe_tool(gate)
    ctx, event = context(), call_event("c1", "wipe", {"item": "a"})
    gated = native._middleware[-1]
    first = await gated(native, event, ctx)
    replay = await native(event, ctx)
    assert shown(first) == "wiped a" and ran == ["a"]
    assert shown(replay)["error"]["kind"] == "approval-denied"


async def test_a_token_the_call_did_not_take_is_dropped():
    gate = ScriptedApproval([ApprovalDecision(approved=True)])
    native, ran = wipe_tool(gate)
    event = call_event("c1", "wipe", {"item": "a"})

    async def skip(event: Any, ctx: Any) -> Any:
        return ToolResultEvent.from_call(event, result="skipped")

    await native._middleware[-1](skip, event, context())
    assert len(native.tokens) == 0 and ran == []


@pytest.mark.parametrize("answer", ["y", "always"])
async def test_ag2_approval_required_never_approves_on_its_own(answer):
    """AG2's own "y" or "always" answer adds a gate; the executor's port still decides."""

    def human(request: Any) -> str:
        return answer

    native, ran = wipe_tool(None)
    native = native.with_middleware(approval_required())
    (result,) = await ask(native, call_event("c1", "wipe", {"item": "a"}), hitl=human)
    assert shown(result)["error"]["kind"] == "approval-denied" and ran == []


async def test_an_approval_that_edits_arguments_is_a_denial():
    edit = ApprovalDecision(approved=True, edited_args={"item": "b"})
    native, ran = wipe_tool(ScriptedApproval([edit]))
    (result,) = await ask(native, call_event("c1", "wipe", {"item": "a"}))
    assert shown(result)["error"]["kind"] == "approval-denied" and ran == []
    assert len(native.tokens) == 0


async def test_a_denial_is_model_visible_and_skips_the_tool():
    gate = ScriptedApproval([ApprovalDecision(approved=False, reason="not today")])
    native, ran = wipe_tool(gate)
    (result,) = await ask(native, call_event("c1", "wipe", {"item": "a"}))
    error = shown(result)["error"]
    assert error["kind"] == "approval-denied" and "not today" in error["message"] and ran == []


async def test_a_pending_decision_skips_the_tool_and_names_the_approval():
    port = DeferredApproval(MemoryApprovalStore())
    native, ran = wipe_tool(port)
    (result,) = await ask(native, call_event("c1", "wipe", {"item": "a"}))
    (record,) = await port.store.list()
    assert shown(result) == f"approval required: {record.approval_id}" and ran == []


async def test_a_call_with_no_confirm_rule_is_not_asked():
    gate = RecordingApproval(ScriptedApproval([]))

    def look(item: str) -> str:
        """Look at an item."""
        return f"saw {item}"

    native = ag2.ag2_tool(build_spec(look), executor=FakeRunner(), approval=gate)
    (result,) = await ask(native, call_event("c1", "look", {"item": "a"}))
    assert shown(result) == "saw a" and gate.requests == []


async def test_arguments_that_are_not_a_json_object_are_invalid():
    gate = RecordingApproval(ScriptedApproval([]))
    native, ran = wipe_tool(gate)
    for text in ("[1]", "not json"):
        (result,) = await ask(native, call_event("c1", "wipe", text))
        assert isinstance(result, ToolErrorEvent)
        assert shown(result)["error"]["kind"] == "tool-args-invalid"
    assert ran == [] and gate.requests == []


async def test_a_foreign_error_is_a_tool_error_without_its_traceback():
    def boom(item: str) -> str:
        """Fail."""
        raise RuntimeError("secret detail")

    native = ag2.ag2_tool(build_spec(boom), executor=Executor())
    (result,) = await ask(native, call_event("c1", "boom", {"item": "a"}))
    assert isinstance(result, ToolErrorEvent) and type(result.error).__name__ == "ToolFailed"
    assert shown(result)["error"]["kind"] == "tool-failed"
    assert "secret detail" not in result.result.parts[0].content


def test_registration_copies_share_the_spec_runner_and_tokens():
    native, _ = wipe_tool(ScriptedApproval([]))
    clone = copy.deepcopy(native)
    assert (clone.spec, clone.runner, clone.tokens) == (native.spec, native.runner, native.tokens)
    assert clone.source == native.source and clone._middleware == native._middleware


def test_bind_context_reads_the_event_and_stream():
    ctx = context()
    bound = ag2.INTEGRATION.bind_context((call_event("c9", "x", {}), ctx))
    assert (bound.call_id, bound.run_id) == ("c9", str(ctx.stream.id))
    assert bound.thread_id == bound.run_id and bound.approval is None
    assert ag2.INTEGRATION.bind_context(call_event("c9", "x", {})).run_id == ""
    assert ag2.INTEGRATION.bind_approval(ScriptedApproval([])) is None


def peek(item: str) -> str:
    """Peek at an item."""
    return item


def test_an_unsupported_ag2_release_is_refused(monkeypatch):
    import ag2 as ag2_pkg

    monkeypatch.setattr(ag2_pkg, "__version__", "2.0.0")
    ag2._tool_class.cache_clear()
    try:
        with pytest.raises(IntegrationError, match="outside the supported range"):
            ag2.ag2_tool(build_spec(peek))
    finally:
        ag2._tool_class.cache_clear()


def test_the_example_runs(capsys):
    assert _run("ag2_tools.py", capsys).splitlines() == [
        "[('shipping_rate', 5.0)]",
        "[('send_invoice', 'invoiced cus_1024 for 49.50 USD')]",
        '[(\'send_invoice\', \'{"error": {"kind": "approval-denied", "message": '
        "\"approval denied for tool \\'send_invoice\\': unknown customer\"}}')]",
    ]
