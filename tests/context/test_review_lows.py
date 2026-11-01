"""Edge fixes: legacy calls, nested wrappers, cache-key identity, saved-data scope, tokens."""

import pytest

from agent_adapter.context import (
    CompressingModel,
    CompressionPolicy,
    MyOwnStep,
    RemoveOldMessages,
    ShrinkToolResults,
    Summarize,
    estimate_tokens,
    recall_tool,
)
from agent_adapter.context.formats import compress_for_sync, convert
from agent_adapter.context.model import COMPRESSED_MARK, policy_key
from agent_adapter.context.recall import owner_scope, saved_result_id, saved_result_key
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ModelContextOverflow, ToolError
from agent_adapter.core.messages import Message, ProviderRaw
from agent_adapter.exec import run_tool_loop
from tests.context.ai_fakes import FakeSummariser
from tests.context.conversations import context_for, tool_round, user
from tests.context.test_wiring_model import HISTORY, _overflow, _request
from tests.context.test_wiring_model import POLICY as WIRED
from tests.exec.test_tool_loop_compression import registry
from tests.fakes.cache import DictCache
from tests.fakes.models import ScriptedModel, text

PAD = "x" * 400

# --- legacy OpenAI function_call -----------------------------------------------------------


def _legacy_chat():
    return [
        {"role": "user", "content": f"weather? {PAD}"},
        {"role": "assistant", "content": None, "function_call": {"name": "w", "arguments": "{}"}},
        {"role": "function", "name": "w", "content": f"sunny {PAD}"},
        {"role": "assistant", "content": f"It is sunny. {PAD}"},
        {"role": "user", "content": "thanks"},
    ]


def test_a_legacy_function_call_and_its_result_form_one_round():
    messages = convert(_legacy_chat(), "openai").messages
    assert messages[1].tool_calls[0].id == "legacy-1" and messages[2].tool_call_id == "legacy-1"
    policy = CompressionPolicy(
        start_when_tokens_over=50,
        shrink_to_tokens=10,
        steps=[RemoveOldMessages(keep_first=0, keep_last=3)],
    )
    items = _legacy_chat()
    smaller, _ = compress_for_sync(items, policy, format="openai")
    roles = [item["role"] for item in smaller]
    assert ("function" in roles) == any("function_call" in item for item in smaller)
    assert smaller[-1] is items[-1]


# --- one retry, however many layers ---------------------------------------------------------


async def test_a_compressing_model_inside_another_retries_once_in_all():
    inner = ScriptedModel([_overflow(), _overflow(), text("too late")])
    nested = CompressingModel(CompressingModel(inner, WIRED), WIRED)
    with pytest.raises(ModelContextOverflow):
        await nested.generate(_request())
    assert len(inner.requests) == 2
    assert all(r.metadata[COMPRESSED_MARK] == "1" for r in inner.requests)


async def test_run_tool_loop_around_a_compressing_model_retries_once_in_all():
    inner = ScriptedModel([_overflow(), _overflow(), text("too late")])
    with pytest.raises(ModelContextOverflow):
        await run_tool_loop(CompressingModel(inner, WIRED), registry, HISTORY, compression=WIRED)
    assert len(inner.requests) == 2


# --- cache keys tell models and functions apart ---------------------------------------------


def test_policy_keys_differ_by_summary_model():
    one = CompressionPolicy(model_max_tokens=100, steps=[Summarize(FakeSummariser(model="a"))])
    two = CompressionPolicy(model_max_tokens=100, steps=[Summarize(FakeSummariser(model="b"))])
    assert policy_key(one) != policy_key(two)
    assert policy_key(one) == policy_key(
        CompressionPolicy(model_max_tokens=100, steps=[Summarize(FakeSummariser(model="a"))])
    )


def test_policy_keys_differ_by_function_even_for_lambdas_in_one_module():
    first = CompressionPolicy(model_max_tokens=100, steps=[MyOwnStep(lambda m: m)])
    second = CompressionPolicy(model_max_tokens=100, steps=[MyOwnStep(lambda m: m[-3:])])
    assert policy_key(first) != policy_key(second)
    counted = CompressionPolicy(model_max_tokens=100, count_tokens=len)
    assert policy_key(counted) != policy_key(CompressionPolicy(model_max_tokens=100))


def test_compressing_model_cache_fields_follow_the_summary_model():
    inner = ScriptedModel([])
    a, b = (
        CompressingModel(inner, CompressionPolicy(model_max_tokens=100, summary_model=summariser))
        for summariser in (FakeSummariser(model="a"), FakeSummariser(model="b"))
    )
    assert a.cache_key_fields() != b.cache_key_fields()


# --- saved tool results: owner scope and lifetime -------------------------------------------


def _history():
    messages = [user("task")]
    for n in range(3):
        messages += tool_round(f"c{n}", f"{n}" * 1000)
    return [*messages, user("latest")]


async def test_saved_results_live_in_the_owner_scope_with_a_ttl():
    cache = DictCache()
    messages = _history()
    context = context_for(messages, 0, lambda m: sum(len(x.content) for x in m))
    context.ctx = ToolContext(run_id="r1", thread_id="t1")
    step = ShrinkToolResults(bigger_than_tokens=500, save_to=cache, save_for_seconds=60)
    await step.apply(list(messages), context)
    result_id = saved_result_id(messages[2])
    key = saved_result_key(result_id, "thread=t1")
    assert key in cache.data and cache.data[key][1] == cache.clock() + 60
    spec = recall_tool(cache)
    own = await spec.invoke({"id": result_id}, ctx=ToolContext(thread_id="t1"))
    assert own == "0" * 1000
    with pytest.raises(ToolError, match="no saved result"):
        await spec.invoke({"id": result_id}, ctx=ToolContext(thread_id="t2"))
    with pytest.raises(ToolError, match="no saved result"):
        await spec.invoke({"id": result_id})


def test_owner_scope_prefers_the_thread_then_the_run():
    assert owner_scope(ToolContext(run_id="r", thread_id="t")) == "thread=t"
    assert owner_scope(ToolContext(run_id="r")) == "run=r"
    assert owner_scope(None) == owner_scope(ToolContext()) == "global"


def test_save_for_seconds_is_checked():
    assert ShrinkToolResults().save_for_seconds == 86_400
    with pytest.raises(ConfigError, match="save_for_seconds"):
        ShrinkToolResults(save_for_seconds=0)


# --- replayed provider items are counted ----------------------------------------------------


def test_raw_items_replayed_to_the_provider_count_as_tokens():
    thinking = [{"type": "thinking", "thinking": "t" * 4000, "signature": "s"}]
    plain = Message(role="assistant", content="ok")
    replayed = plain.model_copy(update={"raw": ProviderRaw(provider="anthropic", items=thinking)})
    assert estimate_tokens([replayed]) > estimate_tokens([plain]) + 900


def test_converted_thinking_blocks_count_as_tokens():
    items = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t" * 4000, "signature": "s"},
                {"type": "text", "text": "ok"},
            ],
        },
    ]
    gemini = [{"role": "model", "parts": [{"text": "t" * 4000, "thought": True}, {"text": "ok"}]}]
    for native, fmt in ((items, "anthropic"), (gemini, "gemini")):
        messages = convert(native, fmt).messages
        assert estimate_tokens(messages) > 1000, fmt
