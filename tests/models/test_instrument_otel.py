"""Nested instrumented calls report one provider call to OpenTelemetry once."""

import pytest

pytest.importorskip("opentelemetry.sdk")

from agent_adapter.core.errors import ModelUnavailable
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.types import Usage
from agent_adapter.cost import Ledger
from agent_adapter.exec import Executor
from agent_adapter.models import CallableModel, FallbackModel, InstrumentedModel
from agent_adapter.tracing import EventLog
from tests.fakes.models import ScriptedModel, text
from tests.tracing.otel_harness import Harness

HI = LLMRequest(messages=(Message(role="user", content="hi"),))
USAGE = Usage(input_tokens=7, output_tokens=3)


def tokens(h):
    """(count, sum) per token type over every data point."""
    found = {}
    for point in h.metrics().get("gen_ai.client.token.usage", []):
        kind = dict(point.attributes)["gen_ai.token.type"]
        count, total = found.get(kind, (0, 0))
        found[kind] = (count + point.count, total + point.sum)
    return found


def durations(h, *, failed=False):
    """Duration observations of finished (or failed) calls."""
    points = h.metrics().get("gen_ai.client.operation.duration", [])
    return sum(p.count for p in points if ("error.type" in dict(p.attributes)) == failed)


def answering(model="b"):
    return ScriptedModel([text("ok", model=model, usage=USAGE)], model=model)


def down(model):
    return ScriptedModel([ModelUnavailable("down", provider="scripted")], model=model)


async def test_an_instrumented_fallback_of_instrumented_hops_reports_tokens_once():
    h = Harness()
    executor = Executor(ledger=Ledger(), sink=h.sink)
    model = executor.model(FallbackModel([executor.model(down("a")), executor.model(answering())]))
    async with executor.run_scope() as ctx:
        await model.bind(ctx).generate(HI)
    assert tokens(h) == {"input": (1, 7), "output": (1, 3)}
    assert (durations(h), durations(h, failed=True)) == (1, 1)
    spans = h.exporter.get_finished_spans()
    [root] = [s for s in spans if s.parent is None]
    hops = [s for s in spans if s.parent is not None]
    assert len(hops) == 2  # the failed hop and the answering hop nest under the call's span
    assert {hop.parent.span_id for hop in hops} == {root.context.span_id}


async def test_a_ledger_less_nested_call_reports_tokens_and_duration_once():
    h = Harness()
    executor = Executor(sink=h.sink)
    await executor.model(executor.model(answering())).generate(HI)
    assert tokens(h) == {"input": (1, 7), "output": (1, 3)}
    assert durations(h) == 1


async def test_an_inner_call_with_its_own_sink_leaves_the_outer_tokens_visible():
    h = Harness()
    ledger = Ledger()
    inner = InstrumentedModel(answering(), ledger=ledger, sink=EventLog())
    await InstrumentedModel(inner, ledger=ledger, sink=h.sink).generate(HI)
    assert tokens(h) == {"input": (1, 7), "output": (1, 3)}
    assert durations(h) == 1


async def test_failing_nested_hops_record_one_failed_duration_per_provider_call():
    h = Harness()
    executor = Executor(ledger=Ledger(), sink=h.sink)
    model = executor.model(FallbackModel([executor.model(down("a")), executor.model(down("b"))]))
    with pytest.raises(ModelUnavailable):
        await model.generate(HI)
    assert (durations(h), durations(h, failed=True)) == (0, 2)
    assert len(h.exporter.get_finished_spans()) == 3  # every span still closes


async def test_an_instrumented_hop_then_a_raw_hop_failing_record_two_failed_durations():
    h = Harness()
    executor = Executor(ledger=Ledger(), sink=h.sink)
    model = executor.model(FallbackModel([executor.model(down("a")), down("b")]))
    with pytest.raises(ModelUnavailable):
        await model.generate(HI)
    assert (durations(h), durations(h, failed=True)) == (0, 2)


async def test_an_outer_error_raised_after_a_nested_success_is_not_nested():
    h = Harness()
    executor = Executor(ledger=Ledger(), sink=h.sink)
    inner = executor.model(answering())

    async def ask_then_fail(request):
        await inner.generate(request)
        raise ModelUnavailable("own failure", provider="scripted")

    model = executor.model(CallableModel(ask_then_fail, provider="scripted", model="outer"))
    with pytest.raises(ModelUnavailable):
        await model.generate(HI)
    assert (durations(h), durations(h, failed=True)) == (1, 1)
