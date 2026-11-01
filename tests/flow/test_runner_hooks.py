import asyncio
import dataclasses

import pytest

from agent_adapter.core.errors import FlowError, GuardrailBlocked
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow, FlowRunner, Map, node_model, node_prompt
from agent_adapter.hooks.base import Hook, ShortCircuit
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.tracing.events import REQUIRED_DATA
from tests.fakes.hooks import RaisingHook, RecordingHook
from tests.fakes.models import ScriptedModel
from tests.fakes.sinks import RecordingLedger, RecordingSink
from tests.flow.helpers import State, registry, runner


def user(text):
    return Message(role="user", content=text)


def one_node(action, **options):
    return Flow("hooked", state=State).node("a", action, **options).end("a")


async def test_hooks_see_before_and_after_with_the_update():
    hook = RecordingHook()
    run, _, _ = runner(hooks=[hook])
    await run.run(one_node(lambda s: {"n": 1}), {})
    assert hook.phases() == ["before_node", "after_node"]
    inv, update = hook.calls[1][1]
    assert inv.flow == "hooked" and inv.node == "a" and inv.ctx.call_id == "hooked:a:1"
    assert update == {"n": 1}


async def test_executor_and_runner_hooks_both_run_once_each():
    shared = RecordingHook()
    extra = RecordingHook()
    run, _, _ = runner(hooks=[shared, extra])
    run.executor = run.executor.replace(hooks=[shared])
    await run.run(one_node(lambda s: None), {})
    assert shared.phases() == ["before_node", "after_node"]
    assert extra.phases() == ["before_node", "after_node"]


async def test_before_node_short_circuit_skips_the_action():
    ran = []
    hook = RecordingHook(returns={"before_node": ShortCircuit({"note": "cached"})})
    run, _, _ = runner(hooks=[hook])
    result = await run.run(one_node(lambda s: ran.append(1)), {})
    assert result.state.note == "cached" and ran == []
    none_hook = RecordingHook(returns={"before_node": ShortCircuit(None)})
    run, _, _ = runner(hooks=[none_hook])
    assert (await run.run(one_node(lambda s: {"n": 5}), {})).state.n == 0


async def test_on_node_error_recovers_and_after_node_replaces():
    class Recover(Hook):
        async def on_node_error(self, inv, error):
            return ShortCircuit({"note": f"recovered {type(error).__name__}"})

    class Replace(Hook):
        async def after_node(self, inv, update):
            return {"note": update["note"].upper()}

    def broken(state):
        raise ValueError("x")

    run, _, _ = runner(hooks=[Recover(), Replace()])
    assert (await run.run(one_node(broken), {})).state.note == "RECOVERED VALUEERROR"


async def test_after_node_sees_a_map_as_one_list_and_may_replace_it():
    class Seen(Hook):
        def __init__(self):
            self.views = []

        async def after_node(self, inv, update):
            self.views.append(update)
            return {"log": ["replaced"]}

    hook = Seen()
    run, _, _ = runner(hooks=[hook])
    flow = Flow("maps", state=State).node("log", Map("items", fn=lambda i, s, c: i)).end("log")
    result = await run.run(flow, {"items": ["a", "b"]})
    assert hook.views == [{"log": ["a", "b"]}]
    assert result.state.log == ["replaced"]


async def test_a_raising_hook_is_a_node_failure():
    run, _, _ = runner(hooks=[RaisingHook(GuardrailBlocked("no"), phases=["before_node"])])
    with pytest.raises(FlowError, match="guardrail-blocked") as info:
        await run.run(one_node(lambda s: None), {})
    assert info.value.node == "a" and isinstance(info.value.__cause__, GuardrailBlocked)


async def test_a_hook_short_circuit_must_be_a_mapping():
    hook = RecordingHook(returns={"before_node": ShortCircuit("text")})
    run, _, _ = runner(hooks=[hook])
    with pytest.raises(FlowError, match="returned str, expected a mapping or None"):
        await run.run(one_node(lambda s: None), {})


async def test_node_retries_use_the_executor_backoff():
    attempts = []

    def flaky(state):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("down")
        return {"n": len(attempts)}

    run, _, clock = runner()
    result = await run.run(one_node(flaky, retries=2), {})
    assert result.state.n == 3 and clock.sleeps == [0.5, 1.0]


async def test_node_timeout_is_a_flow_error():
    async def slow(state):
        await asyncio.sleep(1)

    run, _, _ = runner()
    with pytest.raises(FlowError, match=r"node 'a' timed out after 0\.01s") as info:
        await run.run(one_node(slow, timeout=0.01), {})
    assert info.value.node == "a"


async def test_a_tool_timeout_inside_a_node_is_not_the_node_timeout():
    spec = dataclasses.replace(registry.get("slow"), timeout=0.01)
    run, _, _ = runner()
    flow = one_node(Call(spec, args={"seconds": 1}), timeout=5)
    with pytest.raises(FlowError, match="tool-timeout"):
        await run.run(flow, {})


async def test_events_follow_the_catalogue_in_order():
    sink = RecordingSink()
    run, _, _ = runner(sink=sink)
    flow = (
        Flow("events", state=State)
        .node("a", lambda s: None)
        .node("b", lambda s: None)
        .edge("a", "b")
        .end("b")
    )
    await run.run(flow, {})
    assert sink.types() == [
        "run.started",
        "flow.started",
        "node.started",
        "node.finished",
        "node.started",
        "node.finished",
        "flow.finished",
        "run.finished",
    ]
    for event in sink.events:
        assert REQUIRED_DATA[event.type] <= set(event.data)
    started = sink.of_type("node.started")[1]
    assert started.call_id == "events:b:1" and started.data == {
        "flow": "events",
        "node": "b",
        "visit": 1,
    }
    assert sink.of_type("flow.finished")[0].data == {"flow": "events", "status": "done", "steps": 2}
    assert sink.of_type("run.finished")[0].data == {"ok": True}


async def test_failed_flows_emit_node_failed_and_flow_finished_failed():
    sink = RecordingSink()
    run, _, _ = runner(flow_sink=sink)

    def broken(state):
        raise ValueError("x")

    with pytest.raises(FlowError):
        await run.run(one_node(broken), {})
    assert sink.types() == [
        "run.started",
        "flow.started",
        "node.started",
        "node.failed",
        "flow.finished",
        "run.finished",
    ]
    assert sink.of_type("node.failed")[0].data["error_kind"] == "flow-error"
    assert sink.of_type("flow.finished")[0].data["status"] == "failed"
    assert sink.of_type("run.finished")[0].data == {"ok": False}


async def test_a_caller_owned_run_gets_no_run_events_and_is_not_ended():
    sink = RecordingSink()
    run, executor, _ = runner(sink=sink)
    async with executor.run_scope() as ctx:
        await run.run(one_node(lambda s: None), {}, ctx)
    assert sink.types().count("run.started") == 1  # only the scope's own


async def test_control_flow_passes_through_unwrapped():
    class GraphBubbleUp(Exception):
        pass

    def interrupt(state):
        raise GraphBubbleUp

    sink = RecordingSink()
    run, _, _ = runner(sink=sink)
    with pytest.raises(GraphBubbleUp):
        await run.run(one_node(interrupt), {})
    assert "node.failed" not in sink.types() and "flow.finished" not in sink.types()

    async def cancelled(state):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run.run(one_node(cancelled), {})


async def test_nodes_read_their_prompt_and_the_runner_model():
    seen = {}
    model = ScriptedModel(["hi"], provider="p", model="m")

    def reads(state, ctx):
        seen["prompt"], seen["model"] = node_prompt(ctx), node_model(ctx)

    run, _, _ = runner(model=model)
    await run.run(one_node(reads, prompt="Be brief."), {})
    assert seen["prompt"] == "Be brief."
    # the model itself, or the executor's instrumented wrapper around it (WP08)
    assert (seen["model"].provider, seen["model"].model) == ("p", "m")
    run, _, _ = runner()
    await run.run(one_node(reads), {})
    assert seen == {"prompt": None, "model": None}


async def node_calls_the_model(model, executor):
    """Run a one-node flow whose node asks `node_model(ctx)` once; return what it saw."""
    seen = {}

    async def asks(state, ctx):
        seen["model"] = node_model(ctx)
        seen["reply"] = await seen["model"].generate(LLMRequest(messages=[user("hi")]))

    await FlowRunner(executor, model=model).run(one_node(asks), {})
    return seen


async def test_the_model_is_instrumented_by_the_executor_and_bound_to_the_node():
    ledger, sink = RecordingLedger(), RecordingSink()
    executor = Executor(registry, ledger=ledger, sink=sink)
    raw = ScriptedModel(["hi"])
    seen = await node_calls_the_model(raw, executor)
    model = seen["model"]
    assert isinstance(model, InstrumentedModel) and model.wrapped is raw
    assert model.ctx.call_id == "hooked:a:1" and seen["reply"].text == "hi"
    (entry,) = [e for e in ledger.entries() if e.kind == "llm"]
    assert entry.entry_id.startswith("hooked:a:1:llm:") and ledger.records == 1
    assert len(sink.of_type("llm.started")) == 1


async def test_an_already_instrumented_model_is_bound_not_wrapped_again():
    ledger, sink = RecordingLedger(), RecordingSink()
    executor = Executor(registry, ledger=ledger, sink=sink)
    raw = ScriptedModel(["hi"])
    instrumented = executor.model(raw)
    seen = await node_calls_the_model(instrumented, executor)
    model = seen["model"]
    assert type(model) is InstrumentedModel and model.wrapped is raw  # bound, not re-wrapped
    assert model.ctx.call_id == "hooked:a:1"
    assert [e.kind for e in ledger.entries()].count("llm") == 1 and ledger.records == 1
    assert len(sink.of_type("llm.started")) == 1  # one instrumented call: billed once


async def test_a_node_timeout_is_retried_like_any_retryable_error():
    attempts = []

    async def slow_once(state):
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(1)
        return {"n": len(attempts)}

    run, _, clock = runner()
    result = await run.run(one_node(slow_once, timeout=0.01, retries=1), {})
    assert result.state.n == 2 and clock.sleeps == [0.5]
