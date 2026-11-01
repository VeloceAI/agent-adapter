"""Which cache holds summaries: the step's, else the policy's, else none (opt-in only)."""

import pytest

from agent_adapter.context import CompressionPolicy, Summarize, SummarizeLongMessage, compress
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError
from tests.context.ai_fakes import FakeSummariser
from tests.context.conversations import assistant, system, user
from tests.fakes.cache import DictCache


def chars(messages):
    return sum(len(m.content) for m in messages)


def chat():
    return [system(), user("u" * 300), assistant("a" * 300), user("v" * 300), user("q")]


def summaries(cache):
    return [key for key in cache.data if key.startswith("aa:ctxsum:")]


def make_policy(steps=(), **extra):
    return CompressionPolicy(
        start_when_tokens_over=200, shrink_to_tokens=100, steps=steps, count_tokens=chars, **extra
    )


async def test_the_step_cache_wins_over_the_policy_and_the_run():
    step_cache, policy_cache, run_cache = DictCache(), DictCache(), DictCache()
    step = Summarize(FakeSummariser(), keep_recent_messages=1, cache=step_cache)
    p = make_policy([step], cache=policy_cache)
    await compress(chat(), p, ctx=ToolContext(run_id="r", cache=run_cache))
    assert len(summaries(step_cache)) == 1
    assert summaries(policy_cache) == [] and summaries(run_cache) == []


async def test_the_policy_cache_wins_over_the_run():
    policy_cache, run_cache = DictCache(), DictCache()
    model = FakeSummariser()
    p = make_policy([Summarize(model, keep_recent_messages=1)], cache=policy_cache)
    ctx = ToolContext(run_id="r", cache=run_cache)
    await compress(chat(), p, ctx=ctx)
    await compress(chat(), p, ctx=ctx)
    assert len(summaries(policy_cache)) == 1 and summaries(run_cache) == []
    assert len(model.requests) == 1  # the second compression hit the policy cache


async def test_the_run_cache_is_never_used_without_opting_in():
    run_cache = DictCache()
    model = FakeSummariser()
    p = make_policy([Summarize(model, keep_recent_messages=1)])
    ctx = ToolContext(run_id="r", cache=run_cache)
    await compress(chat(), p, ctx=ctx)
    await compress(chat(), p, ctx=ctx)
    assert run_cache.data == {} and len(model.requests) == 2


async def test_cached_summaries_expire_after_cache_for_seconds():
    cache = DictCache()
    model = FakeSummariser()
    step = Summarize(model, keep_recent_messages=1, cache=cache, cache_for_seconds=60)
    await compress(chat(), make_policy([step]))
    ((_, expires),) = cache.data.values()
    assert expires == cache.clock() + 60
    forever = Summarize(model, keep_recent_messages=1, cache=DictCache(), cache_for_seconds=None)
    await compress(chat(), make_policy([forever]))
    assert [expires for _, expires in forever.cache.data.values()] == [None]
    for bad in (0, -1, float("inf"), "1h", True):
        with pytest.raises(ConfigError, match="cache_for_seconds"):
            Summarize(model, cache_for_seconds=bad)


async def test_no_cache_anywhere_still_summarises_every_time():
    model = FakeSummariser()
    p = make_policy([Summarize(model, keep_recent_messages=1)])
    first, _ = await compress(chat(), p)
    second, _ = await compress(chat(), p)
    assert first == second and len(model.requests) == 2


async def test_the_default_summarize_step_uses_the_policy_cache():
    policy_cache = DictCache()
    model = FakeSummariser()
    p = make_policy(summary_model=model, cache=policy_cache)
    long_chat = [system()] + [user(f"{n} " + "x" * 300) for n in range(12)]
    _, report = await compress(long_chat, p)
    assert report.steps_used[-1] == "summarize"
    assert len(summaries(policy_cache)) == 1


async def test_summarize_long_message_follows_the_same_order():
    step_cache, policy_cache = DictCache(), DictCache()
    messages = [user("a" * 400), assistant("b"), user("c")]
    p = make_policy(
        [SummarizeLongMessage(FakeSummariser(), bigger_than_tokens=50)], cache=policy_cache
    )
    await compress(messages, p)
    assert len(summaries(policy_cache)) == 1
    step = SummarizeLongMessage(FakeSummariser(), bigger_than_tokens=50, cache=step_cache)
    await compress(messages, make_policy([step], cache=policy_cache))
    assert len(summaries(step_cache)) == 1 and len(summaries(policy_cache)) == 1


async def test_the_cache_object_is_not_part_of_the_summary_key():
    one, two = DictCache(), DictCache()
    await compress(
        chat(), make_policy([Summarize(FakeSummariser(), keep_recent_messages=1, cache=one)])
    )
    await compress(
        chat(), make_policy([Summarize(FakeSummariser(), keep_recent_messages=1, cache=two)])
    )
    assert summaries(one) == summaries(two)


def test_caches_must_be_cache_ports():
    with pytest.raises(ConfigError, match="cache"):
        make_policy(cache=object())
    with pytest.raises(ConfigError, match="cache"):
        Summarize(FakeSummariser(), cache="redis://")
    with pytest.raises(ConfigError, match="cache"):
        SummarizeLongMessage(FakeSummariser(), cache=1)
    assert make_policy(cache=DictCache()).cache is not None
