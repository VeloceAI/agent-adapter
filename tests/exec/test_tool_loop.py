"""run_tool_loop: steps, stop reasons, tool messages, request options and resume."""

import json

import pytest

from agent_adapter import Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ModelBadRequest, ToolUsageError
from agent_adapter.core.messages import LLMResponse, Message, OutputSchema
from agent_adapter.core.types import ApprovalDecision, ToolCall, Usage
from agent_adapter.exec import Executor, ToolLoopResult, run_tool_loop
from agent_adapter.models import InstrumentedModel
from tests.fakes.models import ScriptedModel, calls, text
from tests.fakes.sinks import RecordingLedger

registry = Registry()
sent: list[str] = []


@registry.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@registry.tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"hello {name}"


@registry.tool
def info(key: str) -> dict:
    """Look something up."""
    return {"key": key, "value": 1}


@registry.tool
def broken() -> str:
    """Always fails."""
    raise ValueError("nope")


@registry.tool(confirm=True)
def send(to: str) -> str:
    """Send money (needs approval)."""
    sent.append(to)
    return f"sent to {to}"


USER = (Message(role="user", content="go"),)


@pytest.fixture(autouse=True)
def reset():
    sent.clear()


def tool_messages(result: ToolLoopResult) -> list[Message]:
    return [m for m in result.messages if m.role == "tool"]


async def test_a_reply_without_tool_calls_stops_after_one_step():
    model = ScriptedModel([text("done", usage=Usage(input_tokens=3, output_tokens=2))])
    result = await run_tool_loop(model, registry, USER)
    assert (result.finish, result.steps, result.final.text) == ("stop", 1, "done")
    assert result.messages == (*USER, result.final.message)
    assert result.tool_results == () and result.usage == Usage(input_tokens=3, output_tokens=2)


async def test_tool_calls_run_and_their_results_go_back_to_the_model():
    model = ScriptedModel(
        [
            calls(
                ("add", {"a": 1, "b": 2}), ("greet", {"name": "ann"}), usage=Usage(input_tokens=5)
            ),
            text("3, hello ann", usage=Usage(input_tokens=7, output_tokens=1)),
        ]
    )
    result = await run_tool_loop(model, registry, USER)
    assert (result.finish, result.steps) == ("stop", 2)
    assert [r.value for r in result.tool_results] == [3, "hello ann"]
    first, second = tool_messages(result)
    assert (first.tool_call_id, first.name, first.content) == ("call_0", "add", "3")
    assert (second.tool_call_id, second.name, second.content) == ("call_1", "greet", "hello ann")
    assert model.requests[1].messages == result.messages[:-1]
    assert result.usage == Usage(input_tokens=12, output_tokens=1)


async def test_each_request_carries_the_tools_options_and_stage():
    schema = OutputSchema(name="A", json_schema={"type": "object"})
    model = ScriptedModel([text('{"a": 1}')])
    result = await run_tool_loop(
        model,
        registry,
        USER,
        output_schema=schema,
        tool_choice="required",
        stage="agent",
        request_options={"temperature": 0.2, "metadata": {"team": "x"}},
    )
    [request] = model.requests
    assert result.final.parsed == {"a": 1}
    assert [t.name for t in request.tools] == [s.name for s in registry]
    assert request.tools[0] == next(iter(registry)).tool_def()
    assert (request.tool_choice, request.output_schema, request.stage) == (
        "required",
        schema,
        "agent",
    )
    assert (request.temperature, request.metadata) == (0.2, {"team": "x"})


@pytest.mark.parametrize("key", ["messages", "tools", "stage", "nonsense"])
async def test_reserved_or_unknown_request_options_are_rejected(key):
    with pytest.raises(ToolUsageError, match=key):
        await run_tool_loop(ScriptedModel([]), registry, USER, request_options={key: 1})


async def test_max_steps_must_be_positive():
    with pytest.raises(ToolUsageError, match="max_steps"):
        await run_tool_loop(ScriptedModel([]), registry, USER, max_steps=0)


async def test_values_errors_and_unknown_tools_become_tool_messages():
    model = ScriptedModel(
        [calls(("info", {"key": "k"}), ("broken", {}), ("missing", {})), text("ok")]
    )
    result = await run_tool_loop(model, registry, USER)
    value, failed, unknown = (m.content for m in tool_messages(result))
    assert json.loads(value) == {"key": "k", "value": 1}
    error = json.loads(failed)["error"]
    assert set(error) == {"kind", "message"} and error["kind"] == "tool-failed"
    assert json.loads(unknown)["error"]["kind"] == "tool-not-found"
    assert result.finish == "stop"


async def test_non_json_values_are_stringified():
    reg = Registry()

    @reg.tool
    def when() -> object:
        """A value json cannot encode."""
        return {1, 2}

    model = ScriptedModel([calls(("when", {})), text("ok")])
    result = await run_tool_loop(model, reg, USER)
    assert tool_messages(result)[0].content == json.dumps("{1, 2}")


@pytest.mark.parametrize("reason", ["length", "content_filter"])
async def test_length_and_content_filter_stop_without_running_tools(reason):
    cut = calls(("greet", {"name": "x"})).model_copy(update={"finish_reason": reason})
    result = await run_tool_loop(ScriptedModel([cut]), registry, USER)
    assert (result.finish, result.steps, result.tool_results) == (reason, 1, ())
    assert result.messages[-1] == cut.message


async def test_max_steps_stops_before_running_the_last_calls():
    model = ScriptedModel([calls(("add", {"a": 1, "b": 1})), calls(("add", {"a": 2, "b": 2}))])
    result = await run_tool_loop(model, registry, USER, max_steps=2)
    assert (result.finish, result.steps) == ("max_steps", 2)
    assert [r.value for r in result.tool_results] == [2]
    assert result.messages[-1].tool_calls[0].arguments == {"a": 2, "b": 2}


async def test_continuing_runs_the_trailing_calls_first():
    model = ScriptedModel([calls(("add", {"a": 1, "b": 1}))])
    stopped = await run_tool_loop(model, registry, USER, max_steps=1)
    model.responses.append(text("two"))
    result = await run_tool_loop(model, registry, stopped.messages)
    assert (result.finish, result.steps, result.final.text) == ("stop", 1, "two")
    assert [r.value for r in result.tool_results] == [2]
    assert model.requests[-1].messages[-1].role == "tool"


async def test_a_pending_approval_stops_the_loop_and_resumes_after_the_decision():
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    ctx = executor.context(run_id="run_hitl")
    model = ScriptedModel([calls(("greet", {"name": "a"}), ("send", {"to": "bob"}))])
    paused = await run_tool_loop(model, registry, USER, executor=executor, ctx=ctx)
    assert (paused.finish, paused.steps, sent) == ("pending", 1, [])
    assert [r.status for r in paused.tool_results] == ["ok", "pending"]
    assert [m.role for m in paused.messages] == ["user", "assistant", "tool"]  # greet answered
    approval_id = paused.tool_results[1].error.details["approval_id"]
    await store.decide(approval_id, ApprovalDecision(approved=True))
    model.responses.append(text("sent"))
    done = await run_tool_loop(model, registry, paused.messages, executor=executor, ctx=ctx)
    assert (done.finish, done.final.text, sent) == ("stop", "sent", ["bob"])
    assert [r.name for r in done.tool_results] == ["send"]
    assert [m.name for m in done.messages if m.role == "tool"] == ["greet", "send"]


async def test_resuming_while_still_pending_stops_without_a_model_call():
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    ctx = executor.context(run_id="run_wait")
    model = ScriptedModel([calls(("send", {"to": "bob"}))])
    paused = await run_tool_loop(model, registry, USER, executor=executor, ctx=ctx)
    again = await run_tool_loop(model, registry, paused.messages, executor=executor, ctx=ctx)
    assert (again.finish, again.steps, len(model.requests)) == ("pending", 0, 1)
    assert again.final.tool_calls == paused.final.tool_calls
    assert again.tool_results[0].error.details == paused.tool_results[0].error.details


async def test_calls_without_ids_get_stable_ids_written_into_the_tool_message():
    model = ScriptedModel(
        [
            calls(ToolCall(name="add", arguments={"a": 1, "b": 1})),
            text("ok"),
        ]
    )
    ctx = ToolContext(run_id="run_ids")
    result = await run_tool_loop(model, registry, USER, ctx=ctx)
    call_id = result.tool_results[0].call_id
    prefix, conversation, turn, index = call_id.rsplit(":", 3)
    assert (prefix, turn, index, len(conversation)) == ("run_ids:tool_loop", "1", "0", 12)
    assert tool_messages(result)[0].tool_call_id == call_id


async def test_without_a_context_one_run_spans_the_whole_loop():
    ledger = RecordingLedger()
    executor = Executor(registry, ledger=ledger)
    model = ScriptedModel([calls(("add", {"a": 1, "b": 1})), text("ok")])
    await run_tool_loop(model, registry, USER, executor=executor)
    runs = {entry.run_id for entry in ledger.entries()}
    assert len(runs) == 1 and len(ledger.entries()) == 3  # two llm entries, one tool entry


async def test_an_empty_run_id_gets_one_for_the_whole_loop():
    ledger = RecordingLedger()
    model = ScriptedModel([calls(("add", {"a": 1, "b": 1})), text("ok")])
    await run_tool_loop(
        model, registry, USER, executor=Executor(registry, ledger=ledger), ctx=ToolContext()
    )
    assert len({entry.run_id for entry in ledger.entries()}) == 1


async def test_without_an_executor_one_is_made_for_the_registry():
    model = ScriptedModel([calls(("add", {"a": 2, "b": 3})), text("ok")])
    result = await run_tool_loop(model, registry, USER)
    assert result.tool_results[0].value == 5


async def test_names_resolve_against_the_given_registry_not_the_executors():
    other = Registry()

    @other.tool
    def add(a: int, b: int) -> int:
        """Subtract, to tell the registries apart."""
        return a - b

    model = ScriptedModel([calls(("add", {"a": 5, "b": 3})), text("ok")])
    result = await run_tool_loop(model, other, USER, executor=Executor(registry))
    assert result.tool_results[0].value == 2


async def test_an_instrumented_model_is_bound_not_wrapped_again():
    own, executors = RecordingLedger(), RecordingLedger()
    model = InstrumentedModel(ScriptedModel([text("ok")]), ledger=own)
    ctx = ToolContext(run_id="run_bound")
    await run_tool_loop(
        model, registry, USER, executor=Executor(registry, ledger=executors), ctx=ctx
    )
    assert [e.run_id for e in own.entries()] == ["run_bound"]
    assert executors.entries() == []


async def test_a_model_error_propagates():
    error = ModelBadRequest("bad", provider="scripted")
    with pytest.raises(ModelBadRequest):
        await run_tool_loop(ScriptedModel([error]), registry, USER)


async def test_concurrency_is_passed_to_the_batch():
    seen = []

    class Spy(Executor):
        async def run_many(self, calls, ctx=None, **kwargs):
            seen.append(kwargs)
            return await super().run_many(calls, ctx, **kwargs)

    model = ScriptedModel([calls(("add", {"a": 1, "b": 1})), text("ok")])
    await run_tool_loop(model, registry, USER, executor=Spy(registry), concurrency=3)
    assert seen[0]["concurrency"] == 3 and seen[0]["registry"] is registry


def test_the_result_is_a_frozen_dataclass():
    result = ToolLoopResult(
        messages=(),
        final=LLMResponse(provider="p", model="m", message=Message(role="assistant")),
        steps=0,
        finish="stop",
        tool_results=(),
        usage=Usage(),
    )
    with pytest.raises(AttributeError):
        result.steps = 1


async def test_a_fully_answered_history_goes_straight_to_the_model():
    history = (
        *USER,
        Message(role="assistant", tool_calls=(ToolCall(name="add", id="c1"),)),
        Message(role="tool", tool_call_id="c1", name="add", content="2"),
    )
    model = ScriptedModel([text("two")])
    result = await run_tool_loop(model, registry, history)
    assert (result.finish, result.tool_results) == ("stop", ())
    assert model.requests[0].messages == history


def test_trailing_turn_and_unanswered_helpers():
    from agent_adapter.exec.tool_loop import _trailing_turn, _unanswered

    tool = Message(role="tool", content="x")
    assert _trailing_turn([tool]) is None
    assert _trailing_turn([Message(role="assistant", content="hi")]) is None
    a = ToolCall(name="a", id="s1")
    b = ToolCall(name="b", id="s2")
    c = ToolCall(name="c", id="i")
    # a legacy tool message without an id answers the first id-less (synthetic) call
    assert _unanswered([(a, False), (b, False), (c, True)], [tool]) == [b, c]
    answered = Message(role="tool", tool_call_id="s2", content="x")
    assert _unanswered([(a, False), (b, False)], [answered]) == [a]


def test_an_error_result_without_error_info_still_renders():
    from agent_adapter.core.types import ToolResult
    from agent_adapter.exec.tool_loop import _tool_content

    result = ToolResult(call_id="c", name="t", status="error")
    assert json.loads(_tool_content(result)) == {"error": {"kind": "internal-error", "message": ""}}


# --- review fix round 1 ---------------------------------------------------------------------

calls_made: list[str] = []


@registry.tool
def weather(city: str) -> str:
    """Today's weather (counts its calls)."""
    calls_made.append(city)
    return f"sunny in {city}"


@registry.tool(once_per_run=True)
def launch(target: str) -> str:
    """Launch once per run."""
    return f"launched {target}"


async def test_resuming_id_less_calls_runs_the_approved_one_once_and_not_the_done_one():
    calls_made.clear()
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    ctx = executor.context(run_id="run_anon")
    turn = calls(
        ToolCall(name="send", arguments={"to": "bob"}),
        ToolCall(name="weather", arguments={"city": "Oslo"}),
    )
    model = ScriptedModel([turn])
    paused = await run_tool_loop(model, registry, USER, executor=executor, ctx=ctx)
    assert paused.finish == "pending" and calls_made == ["Oslo"]
    [record] = await store.list(run_id="run_anon", status="pending")
    await store.decide(record.approval_id, ApprovalDecision(approved=True))
    model.responses.append(text("done"))
    done = await run_tool_loop(model, registry, paused.messages, executor=executor, ctx=ctx)
    assert (done.finish, sent, calls_made) == ("stop", ["bob"], ["Oslo"])
    assert [r.name for r in done.tool_results] == ["send"]
    ids = [m.tool_call_id for m in done.messages if m.role == "tool"]
    assert len(set(ids)) == 2 and all(ids)


async def test_a_fresh_context_with_the_same_run_id_does_not_overwrite_paid_entries():
    ledger = RecordingLedger()
    executor = Executor(registry, ledger=ledger)
    model = ScriptedModel([calls(("add", {"a": 1, "b": 1})), text("two")])
    stopped = await run_tool_loop(
        model, registry, USER, executor=executor, ctx=executor.context(run_id="run_r"), max_steps=1
    )
    # continue elsewhere: same run id, fresh extras
    await run_tool_loop(
        model, registry, stopped.messages, executor=executor, ctx=executor.context(run_id="run_r")
    )
    llm = [e.entry_id for e in ledger.entries("run_r") if e.kind == "llm"]
    assert llm == ["run_r:llm:tool_loop:0", "run_r:llm:tool_loop:1"]  # two paid calls


async def test_id_less_calls_in_two_loops_of_one_run_are_distinct_calls():
    executor = Executor(registry)
    ctx = executor.context(run_id="run_once")
    first = ScriptedModel([calls(ToolCall(name="launch", arguments={"target": "a"})), text("ok")])
    second = ScriptedModel([calls(ToolCall(name="launch", arguments={"target": "a"})), text("ok")])
    one = await run_tool_loop(first, registry, USER, executor=executor, ctx=ctx)
    other = (Message(role="user", content="again"),)
    two = await run_tool_loop(second, registry, other, executor=executor, ctx=ctx)
    assert one.tool_results[0].ok
    assert two.tool_results[0].error.kind == "run-limit-exceeded"


async def test_request_options_may_not_fix_the_ledger_call_id():
    with pytest.raises(ToolUsageError, match="call_id"):
        await run_tool_loop(
            ScriptedModel([]), registry, USER, request_options={"metadata": {"call_id": "x"}}
        )


async def test_invalid_request_options_fail_before_a_resumed_tool_runs():
    calls_made.clear()
    history = (
        *USER,
        Message(
            role="assistant",
            tool_calls=(ToolCall(name="weather", id="w", arguments={"city": "x"}),),
        ),
    )
    with pytest.raises(ToolUsageError, match="invalid"):
        await run_tool_loop(
            ScriptedModel([]), registry, history, request_options={"temperature": "hot"}
        )
    assert calls_made == []


async def test_an_error_finish_without_tool_calls_stops_with_a_warning(caplog):
    broken = text("").model_copy(
        update={"finish_reason": "error", "raw_finish_reason": "MALFORMED"}
    )
    with caplog.at_level("WARNING", logger="agent_adapter.exec.tool_loop"):
        result = await run_tool_loop(ScriptedModel([broken]), registry, USER)
    assert result.finish == "stop" and "MALFORMED" in caplog.text


def test_unserialisable_history_still_gets_a_digest():
    from agent_adapter.exec.tool_loop import _history_digest

    odd = Message(role="assistant", tool_calls=(ToolCall(name="t", arguments={"x": object()}),))
    assert len(_history_digest([odd])) == 12


async def test_a_resumed_history_without_raw_items_does_not_rerun_done_id_less_calls():
    from agent_adapter.core.messages import ProviderRaw

    calls_made.clear()
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    ctx = executor.context(run_id="run_raw")
    turn = calls(
        ToolCall(name="send", arguments={"to": "eve"}),
        ToolCall(name="weather", arguments={"city": "Rome"}),
    )
    raw = ProviderRaw(provider="gemini", items=[{"thought_signature": "c2ln"}])
    turn = turn.model_copy(update={"message": turn.message.model_copy(update={"raw": raw})})
    model = ScriptedModel([turn])
    paused = await run_tool_loop(model, registry, USER, executor=executor, ctx=ctx)
    [record] = await store.list(run_id="run_raw", status="pending")
    await store.decide(record.approval_id, ApprovalDecision(approved=True))
    stripped = [m.model_copy(update={"raw": None}) for m in paused.messages]  # e.g. a JSON store
    model.responses.append(text("done"))
    done = await run_tool_loop(model, registry, stripped, executor=executor, ctx=ctx)
    assert (done.finish, calls_made) == ("stop", ["Rome"])
    assert [r.name for r in done.tool_results] == ["send"] and sent == ["eve"]
