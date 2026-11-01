"""CompressingModel and `LLMRequest.compression` on every built-in port, with the overflow retry."""

import pytest

from agent_adapter.context import CompressingModel, CompressionPolicy, RemoveOldMessages
from agent_adapter.core.errors import ConfigError, ModelContextOverflow, ModelRateLimited
from agent_adapter.core.messages import LLMRequest, Message, StreamDone
from agent_adapter.core.types import ModelCaps
from agent_adapter.models import InstrumentedModel
from tests.fakes.hooks import RecordingHook
from tests.fakes.models import ScriptedModel, text
from tests.fakes.sinks import RecordingLedger
from tests.models.test_port import FakeClient, FakePort

PAD = "x" * 400
HISTORY = (
    Message(role="system", content="s"),
    Message(role="user", content=f"one {PAD}"),
    Message(role="assistant", content=f"two {PAD}"),
    Message(role="user", content=f"three {PAD}"),
    Message(role="assistant", content=f"four {PAD}"),
    Message(role="user", content="now"),
)
STRONG = CompressionPolicy(
    start_when_tokens_over=10_000,
    shrink_to_tokens=5,
    steps=[RemoveOldMessages(keep_first=0, keep_last=1)],
)
POLICY = CompressionPolicy(
    start_when_tokens_over=150,
    shrink_to_tokens=120,
    steps=[RemoveOldMessages(keep_first=0, keep_last=3)],
    if_still_too_long=STRONG,
)


def _overflow():
    return ModelContextOverflow("too long", provider="scripted", model="scripted", status_code=400)


def _request(**fields):
    return LLMRequest(messages=HISTORY, **fields)


async def test_compressing_model_sends_fewer_messages_and_no_policy():
    inner = ScriptedModel([text("ok")])
    reply = await CompressingModel(inner, POLICY).generate(_request())
    assert reply.text == "ok"
    sent = inner.requests[0]
    assert len(sent.messages) < len(HISTORY) and sent.compression is None
    assert sent.messages[0] is HISTORY[0] and sent.messages[-1] is HISTORY[-1]


async def test_a_request_policy_wins_and_when_can_skip():
    inner = ScriptedModel([text("a"), text("b"), text("c")])
    never = CompressingModel(inner, POLICY, when=lambda request, ctx: False)
    await never.generate(_request())
    assert len(inner.requests[0].messages) == len(HISTORY)
    own = CompressionPolicy(
        start_when_tokens_over=50,
        shrink_to_tokens=5,
        steps=[RemoveOldMessages(keep_first=0, keep_last=1)],
    )
    await never.generate(_request(compression=own))
    assert len(inner.requests[1].messages) == 2

    async def only_long(request, ctx):
        return len(request.messages) > 3

    await CompressingModel(inner, POLICY, when=only_long).generate(_request())
    assert len(inner.requests[2].messages) < len(HISTORY)


async def test_overflow_runs_if_still_too_long_once_and_retries_once():
    inner = ScriptedModel([_overflow(), text("ok")])
    reply = await CompressingModel(inner, POLICY).generate(_request())
    assert reply.text == "ok"
    first, second = inner.requests
    assert len(second.messages) < len(first.messages)
    assert [m.role for m in second.messages] == ["system", "user"]


async def test_a_second_overflow_or_no_stronger_policy_propagates():
    inner = ScriptedModel([_overflow(), _overflow()])
    with pytest.raises(ModelContextOverflow):
        await CompressingModel(inner, POLICY).generate(_request())
    assert len(inner.requests) == 2
    plain = CompressionPolicy(start_when_tokens_over=150, shrink_to_tokens=120)
    inner = ScriptedModel([_overflow()])
    with pytest.raises(ModelContextOverflow):
        await CompressingModel(inner, plain).generate(_request())
    assert len(inner.requests) == 1


async def test_overflow_is_not_retried_when_the_stronger_policy_changes_nothing():
    inner = ScriptedModel([_overflow()])
    short = (Message(role="system", content="s"), Message(role="user", content="hi"))
    with pytest.raises(ModelContextOverflow):
        await CompressingModel(inner, POLICY).generate(LLMRequest(messages=short))
    assert len(inner.requests) == 1


async def test_other_errors_are_not_retried():
    inner = ScriptedModel([ModelRateLimited("slow", provider="scripted")])
    with pytest.raises(ModelRateLimited):
        await CompressingModel(inner, POLICY).generate(_request())
    assert len(inner.requests) == 1


async def test_stream_is_compressed_and_retried_before_the_first_event():
    inner = ScriptedModel([_overflow(), text("ok")])
    events = [e async for e in CompressingModel(inner, POLICY).stream(_request())]
    assert isinstance(events[-1], StreamDone) and events[-1].response.text == "ok"
    assert len(inner.requests[1].messages) == 2
    skipped = ScriptedModel([text("raw")])
    model = CompressingModel(skipped, POLICY, when=lambda r, c: False)
    assert [e async for e in model.stream(_request())][-1].response.text == "raw"


async def test_the_window_comes_from_caps_and_is_checked_at_setup():
    percent = CompressionPolicy(
        start_when_full_percent=50,
        shrink_to_percent=40,
        steps=[RemoveOldMessages(keep_first=0, keep_last=1)],
    )
    with pytest.raises(ConfigError, match="no context window is known for model 'scripted'"):
        CompressingModel(ScriptedModel([]), percent)
    inner = ScriptedModel([text("ok")])
    model = CompressingModel(inner, percent, caps=lambda name: ModelCaps(context_window=500))
    await model.generate(_request())
    assert len(inner.requests[0].messages) < len(HISTORY)
    assert model.cache_key_fields()["compression"]["start_when_full_percent"] == 50


def test_setup_checks_the_policy_and_when():
    with pytest.raises(ConfigError, match="CompressionPolicy"):
        CompressingModel(ScriptedModel([]), "policy")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="when must be"):
        CompressingModel(ScriptedModel([]), POLICY, when=3)  # type: ignore[arg-type]


async def test_cache_key_names_each_step_and_the_stronger_policy():
    key = CompressingModel(ScriptedModel([]), POLICY).cache_key_fields()
    assert key["compression"]["steps"][0]["step"] == "remove_old_messages"
    assert key["compression"]["if_still_too_long"]["shrink_to_tokens"] == 5
    assert CompressingModel(ScriptedModel([]), POLICY).provider == "scripted"


async def test_bare_ports_honour_the_request_policy():
    callable_model = ScriptedModel([text("ok"), text("ok")])
    await callable_model.generate(_request(compression=POLICY))
    assert len(callable_model.requests[0].messages) < len(HISTORY)
    events = [e async for e in callable_model.stream(_request(compression=POLICY))]
    assert isinstance(events[-1], StreamDone)
    client = FakeClient(replies=[text("ok", provider="fake")], chunks=["a", "b"])
    port = FakePort(client=client)
    await port.generate(_request(compression=POLICY))
    assert client.calls[0]["n"] < len(HISTORY)
    streamed = [e async for e in port.stream(_request(compression=POLICY))]
    assert isinstance(streamed[-1], StreamDone) and client.calls[1]["n"] < len(HISTORY)


async def test_instrumented_model_compresses_before_hooks_and_the_ledger():
    hook = RecordingHook()
    ledger = RecordingLedger()
    inner = ScriptedModel([text("ok"), text("ok")])
    model = InstrumentedModel(inner, hooks=[hook], ledger=ledger)
    await model.generate(_request(compression=POLICY))
    phase, (invocation,) = hook.calls[0]
    assert phase == "before_llm" and len(invocation.request.messages) < len(HISTORY)
    assert invocation.request.compression is None
    assert len(inner.requests[0].messages) < len(HISTORY)
    assert len(ledger.entries()) == 1
    events = [e async for e in model.stream(_request(compression=POLICY))]
    assert isinstance(events[-1], StreamDone) and len(ledger.entries()) == 2
