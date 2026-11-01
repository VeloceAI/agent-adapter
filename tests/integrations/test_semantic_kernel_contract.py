"""Contract kit C1-C8 for Semantic Kernel, plus the approval filter's own edge cases."""

import json
from typing import Any, ClassVar

import pytest
from pydantic import Field

pytest.importorskip("semantic_kernel")

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_calling_utils import (
    kernel_function_metadata_to_function_call_format,
)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import (
    AuthorRole,
    ChatHistory,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
)
from semantic_kernel.filters import FilterTypes

from agent_adapter.core.errors import ApprovalPending, ToolDefinitionError
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.integrations import semantic_kernel as sk
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.schema.transforms import inline_refs
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations.kit import FakeRunner, contract_tests
from tests.test_examples import _run

pytestmark = pytest.mark.contract

PLUGIN = "agent_adapter"
# Semantic Kernel's own results for calls it refuses or that raise.
SK_ERRORS = (
    "Missing required argument",
    "Received unexpected argument",
    "The tool call arguments are malformed",
    "There are `",
    "An error occurred while invoking",
)


async def auto_invoke(
    native: Any, call_id: str, args: dict[str, Any], *, flt: Any = None, plugin: str = PLUGIN
) -> Any:
    """One auto-invoked function call, as SK's function-calling loop runs it."""
    kernel = Kernel()
    kernel.add_function(plugin_name=plugin, function=native)
    if flt is not None:
        kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, flt)
    history = ChatHistory()
    call = FunctionCallContent(
        id=call_id, name=f"{plugin}-{native.name}", arguments=json.dumps(args)
    )
    await kernel.invoke_function_call(call, history)
    (message,) = history.messages
    (result,) = message.items
    return result


class GatePort:
    """Answers with the decision the harness sets for the next approval flow."""

    def __init__(self) -> None:
        self.decision = ApprovalDecision(approved=False)

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        return self.decision


class SemanticKernelHarness:
    """Drives `Kernel.invoke_function_call`; C6 goes through the auto-invocation filter."""

    name = "semantic_kernel"
    edits_supported = False
    approval = "native"
    native_uses_port = True
    port_asks = 1  # the filter asks; the executor must not ask again

    def __init__(self) -> None:
        self.gate = GatePort()

    def approval_port(self) -> ApprovalPort:
        return self.gate

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any:
        return sk.kernel_function(spec, executor=executor)

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[Any]:
        return list(sk.kernel_plugin(registry, executor=executor).functions.values())

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        return await auto_invoke(native, call_id, args)

    def value(self, raw: Any) -> Any:
        try:
            return json.loads(raw.result)
        except (TypeError, ValueError):
            return raw.result

    def is_error(self, raw: Any) -> bool:
        if isinstance(raw, Exception):
            return True
        value = self.value(raw)
        if isinstance(value, dict) and set(value) == {"error"}:
            return True
        return isinstance(value, str) and value.startswith(SK_ERRORS)

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
        registry = Registry()
        registry.add(getattr(native.method, sk._SPEC_ATTR))
        flt = sk.approval_filter(port, registry)
        return await auto_invoke(native, "call_sk_ok", args, flt=flt)

    def describe(self, native: Any) -> tuple[str, str, dict[str, Any]]:
        tool = kernel_function_metadata_to_function_call_format(native.metadata)
        return native.name, native.description, tool["function"]["parameters"]

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        schema = inline_refs(spec.json_schema())  # SK takes one schema per parameter
        properties = schema.get("properties", {})
        required = [name for name in properties if name in schema.get("required", [])]
        return {"type": "object", "properties": properties, "required": required}


globals().update((item.__name__, item) for item in contract_tests(SemanticKernelHarness))
globals().update(
    (f"{item.__name__}_executor", item)
    for item in contract_tests(SemanticKernelHarness, runner_factory=Executor)
)


# --- bridge specifics -------------------------------------------------------------------------


async def remove(item: str) -> str:
    """Remove an item."""
    return f"removed {item}"


def registry_of(*specs: ToolSpec) -> Registry:
    registry = Registry()
    registry.add_many(specs)
    return registry


async def test_without_the_filter_a_confirm_tool_asks_the_executor_port():
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="nope")])
    native = sk.kernel_function(
        build_spec(remove, confirm=True), executor=FakeRunner(approval=port)
    )
    result = await auto_invoke(native, "c1", {"item": "a"})
    assert json.loads(result.result)["error"]["kind"] == "approval-denied"
    assert [r.call_id for r in port.requests] == ["c1"]


async def test_a_pending_approval_becomes_the_result_without_a_call():
    spec = build_spec(remove, confirm=True)
    flt = sk.approval_filter(ScriptedApproval([ApprovalPending("apr_3")]), registry_of(spec))
    runner = FakeRunner()
    result = await auto_invoke(
        sk.kernel_function(spec, executor=runner), "c2", {"item": "a"}, flt=flt
    )
    assert result.result == "approval required: apr_3" and runner.calls == []


async def test_the_filter_passes_other_plugins_and_unneeded_approvals_through():
    spec = build_spec(remove, confirm=True)
    port = ScriptedApproval([])
    approving = FakeRunner(approval=ScriptedApproval([ApprovalDecision(approved=True)]))
    flt = sk.approval_filter(port, registry_of(spec), plugin=PLUGIN)
    native = sk.kernel_function(spec, executor=approving)
    result = await auto_invoke(native, "c3", {"item": "a"}, flt=flt, plugin="other")
    assert result.result == "removed a" and port.requests == []
    rule = build_spec(remove, confirm=lambda call, ctx: False)
    flt = sk.approval_filter(port, registry_of(rule))
    result = await auto_invoke(sk.kernel_function(rule), "c4", {"item": "b"}, flt=flt)
    assert result.result == "removed b" and port.requests == []


async def test_bind_approval_reads_the_spec_behind_the_function():
    spec = build_spec(remove, confirm=True)
    port = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=True)]))
    runner = FakeRunner(approval=ScriptedApproval([]))
    native = sk.kernel_function(spec, executor=runner)
    flt = sk.INTEGRATION.bind_approval(port)
    result = await auto_invoke(native, "c5", {"item": "a"}, flt=flt)
    assert result.result == "removed a" and [r.call_id for r in port.requests] == ["c5"]
    assert runner.calls[0].ctx.approval is not None and runner.calls[0].ctx.call_id == "c5"


async def test_the_filter_ignores_functions_that_are_not_ours():
    from semantic_kernel.functions import kernel_function as sk_kernel_function

    @sk_kernel_function(name="native_fn", description="Plain SK function.")
    async def native_fn(item: str) -> str:
        return f"plain {item}"

    port = ScriptedApproval([])
    kernel = Kernel()
    kernel.add_function(plugin_name=PLUGIN, function=native_fn)
    kernel.add_filter(FilterTypes.AUTO_FUNCTION_INVOCATION, sk.INTEGRATION.bind_approval(port))
    history = ChatHistory()
    call = FunctionCallContent(id="c6", name=f"{PLUGIN}-native_fn", arguments='{"item": "x"}')
    await kernel.invoke_function_call(call, history)
    assert history.messages[0].items[0].result == "plain x" and port.requests == []


async def test_kernel_invoke_works_outside_function_calling():
    runner = FakeRunner()
    plugin = sk.kernel_plugin(registry_of(build_spec(remove)), executor=runner, name="tools")
    kernel = Kernel()
    kernel.add_plugin(plugin)
    result = await kernel.invoke(plugin_name="tools", function_name="remove", item="z")
    assert result.value == "removed z" and runner.calls[0].ctx.call_id.startswith("call_")


def test_registry_name_maps_plugin_dash_function():
    registry = registry_of(build_spec(remove))
    assert sk._registry_name("agent_adapter-remove", registry, PLUGIN) == "remove"
    assert sk._registry_name("agent_adapter-remove", registry, None) == "remove"
    assert sk._registry_name("remove", registry, None) == "remove"
    assert sk._registry_name("tools-remove", registry, PLUGIN) == ""
    assert sk._registry_name("tools-missing", registry, None) == ""


async def test_the_default_filter_only_gates_its_own_plugin():
    """Regression: plugin=None by default mapped any plugin's functions onto the registry."""
    spec = build_spec(remove, confirm=True)
    filter_port = ScriptedApproval([])
    executor_port = ScriptedApproval([ApprovalDecision(approved=False, reason="executor")])
    native = sk.kernel_function(spec, executor=FakeRunner(approval=executor_port))
    flt = sk.approval_filter(filter_port, registry_of(spec))
    result = await auto_invoke(native, "c7", {"item": "a"}, flt=flt, plugin="evil")
    assert json.loads(result.result)["error"]["kind"] == "approval-denied"
    assert filter_port.requests == [] and len(executor_port.requests) == 1


async def test_a_call_without_an_id_is_denied_by_the_filter():
    spec = build_spec(remove, confirm=True)
    port = ScriptedApproval([])
    runner = FakeRunner()
    native = sk.kernel_function(spec, executor=runner)
    flt = sk.approval_filter(port, registry_of(spec))
    result = await auto_invoke(native, "", {"item": "a"}, flt=flt)
    assert json.loads(result.result)["error"]["kind"] == "approval-denied"
    assert port.requests == [] and runner.calls == []


async def test_a_token_only_approves_its_own_invocation():
    spec = build_spec(remove, confirm=True)
    port = ScriptedApproval([ApprovalDecision(approved=False, reason="asked")])
    native = sk.kernel_function(spec, executor=FakeRunner(approval=port))
    tokens = sk.TokenTable()
    approve = ApprovalDecision(approved=True)
    tokens.issue("agent_adapter-remove", {"item": "a"}, approve, call_id="c1", run_id="r")
    gate = sk._GATE.set(sk._Gate("r", tokens))
    call = sk._CALL.set(("c2", "agent_adapter-remove"))  # another call id
    try:
        ctx = sk.INTEGRATION.bind_context((spec, {"item": "a"}))
    finally:
        sk._CALL.reset(call)
        sk._GATE.reset(gate)
    assert (ctx.run_id, ctx.call_id, ctx.approval) == ("r", "c2", None)
    assert native is not None


async def test_filter_and_calls_share_the_conversations_run_id():
    spec = build_spec(remove, confirm=True)
    port = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=True)] * 2))
    runner = FakeRunner()
    kernel = Kernel()
    kernel.add_function(plugin_name=PLUGIN, function=sk.kernel_function(spec, executor=runner))
    kernel.add_filter(
        FilterTypes.AUTO_FUNCTION_INVOCATION, sk.approval_filter(port, registry_of(spec))
    )
    history = ChatHistory()
    for call_id in ("r1", "r2"):
        call = FunctionCallContent(id=call_id, name=f"{PLUGIN}-remove", arguments='{"item": "a"}')
        await kernel.invoke_function_call(call, history)
    runs = {r.run_id for r in port.requests} | {inv.ctx.run_id for inv in runner.calls}
    assert len(runs) == 1 and next(iter(runs)).startswith("run_")
    assert sk._run_id(None, "fallback") == "fallback" and sk._run_id(1, "fb") == "fb"
    other = ChatHistory()
    assert sk._run_id(other, "fb") not in runs | {"fb"}
    key = id(other)
    del other
    assert key not in sk._RUN_IDS


class ScriptedChat(ChatCompletionClientBase):
    """A chat service that asks for one tool call, then answers with the tool's result."""

    SUPPORTS_FUNCTION_CALLING: ClassVar[bool] = True
    calls: list[FunctionCallContent] = Field(default_factory=list)

    async def _inner_get_chat_message_contents(
        self, chat_history: ChatHistory, settings: PromptExecutionSettings
    ) -> list[ChatMessageContent]:
        results = [
            item.result
            for message in chat_history.messages
            for item in message.items
            if isinstance(item, FunctionResultContent)
        ]
        if results:
            return [ChatMessageContent(role=AuthorRole.ASSISTANT, content=str(results[-1]))]
        return [ChatMessageContent(role=AuthorRole.ASSISTANT, items=list(self.calls))]


@pytest.mark.parametrize(("approved", "answer"), [(True, "removed a"), (False, "approval-denied")])
async def test_full_auto_invoke_loop_through_a_chat_service(approved, answer):
    spec = build_spec(remove, confirm=True)
    port = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=approved)]))
    runner = FakeRunner(approval=ScriptedApproval([]))  # asking again would fail the test
    kernel = Kernel()
    kernel.add_plugin(sk.kernel_plugin(registry_of(spec), executor=runner))
    kernel.add_filter(
        FilterTypes.AUTO_FUNCTION_INVOCATION, sk.approval_filter(port, registry_of(spec))
    )
    call = FunctionCallContent(id="loop_1", name=f"{PLUGIN}-remove", arguments='{"item": "a"}')
    service = ScriptedChat(ai_model_id="scripted", calls=[call])
    settings = PromptExecutionSettings(function_choice_behavior=FunctionChoiceBehavior.Auto())
    history = ChatHistory()
    history.add_user_message("remove a")
    (reply,) = await service.get_chat_message_contents(history, settings, kernel=kernel)
    assert answer in str(reply.content)
    assert [(r.tool, r.call_id) for r in port.requests] == [("remove", "loop_1")]
    assert len(runner.calls) == (1 if approved else 0)


def test_reserved_argument_names_are_rejected():
    async def clash(kernel: str, arguments: str) -> str:
        """Clash with names Semantic Kernel injects."""
        return kernel

    with pytest.raises(ToolDefinitionError, match="reserves the argument name"):
        sk.kernel_function(build_spec(clash))


def test_parameters_carry_the_inlined_property_schemas():
    async def pick(choice: str | None = None, tags: list[str] | None = None) -> str:
        """Pick something."""
        return str(choice)

    native = sk.kernel_function(build_spec(pick))
    params = {p.name: p for p in native.parameters}
    assert params["choice"].type_ == "object" and params["choice"].is_required is False
    assert (
        params["tags"].schema_data
        == inline_refs(build_spec(pick).json_schema())["properties"]["tags"]
    )


def test_the_example_runs(capsys):
    rate, filed, denied = _run("semantic_kernel_tools.py", capsys).splitlines()
    assert (rate, filed) == ("5.0", "invoiced cus_1024 for 49.50 USD")
    assert json.loads(denied)["error"]["kind"] == "approval-denied"
