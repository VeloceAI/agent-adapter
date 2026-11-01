"""Executor.model(): an InstrumentedModel with the executor's hooks, cache, ledger and sink."""

from agent_adapter.cache import MemoryCache
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.exec import Executor
from agent_adapter.models import InstrumentedModel
from tests.fakes.hooks import RecordingHook
from tests.fakes.models import ScriptedModel, text
from tests.fakes.sinks import RecordingLedger, RecordingSink

HI = LLMRequest(messages=(Message(role="user", content="hi"),))


def test_model_wraps_the_port_with_the_executors_configuration():
    hook, ledger, sink, cache = RecordingHook(), RecordingLedger(), RecordingSink(), MemoryCache()
    executor = Executor(hooks=[hook], ledger=ledger, sink=sink, cache=cache, namespace="ns")
    port = ScriptedModel([])
    model = executor.model(port, cache_ttl=30)
    assert isinstance(model, InstrumentedModel) and model.wrapped is port
    assert (model.hooks, model.ledger, model.sink, model.cache) == (
        (hook,),
        ledger,
        executor.sink,
        cache,
    )
    assert executor.sink.sinks == (sink,)
    assert (model.namespace, model.cache_ttl, model.ctx) == ("ns", 30, None)


async def test_calls_through_it_reach_the_executors_hooks_ledger_and_sink():
    hook, ledger, sink = RecordingHook(), RecordingLedger(), RecordingSink()
    executor = Executor(hooks=[hook], ledger=ledger, sink=sink)
    async with executor.run_scope() as ctx:
        response = await executor.model(ScriptedModel([text("ok")])).bind(ctx).generate(HI)
    assert response.text == "ok"
    assert hook.phases() == ["before_llm", "after_llm"]
    assert [e.run_id for e in ledger.entries()] == [ctx.run_id]
    assert sink.types() == ["run.started", "llm.started", "llm.finished", "run.finished"]


async def test_a_replaced_executor_builds_models_with_its_own_ports():
    first, second = RecordingLedger(), RecordingLedger()
    executor = Executor(ledger=first).replace(ledger=second)
    await executor.model(ScriptedModel([text("ok")])).generate(HI)
    assert first.entries() == [] and len(second.entries()) == 1
