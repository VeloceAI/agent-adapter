import asyncio
import copy
import logging
from types import SimpleNamespace

import pytest

from agent_adapter.cache import decode, digest, encode, gcache_key
from agent_adapter.cache.providers import GeminiExplicitCache, GeminiImplicitCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.ports import PromptCachePlugin
from agent_adapter.core.types import ModelCaps
from agent_adapter.runtime import run_sync, use_context
from tests.fakes.cache import BrokenCache, DictCache
from tests.fakes.sinks import RecordingLedger

MODEL = "gemini-2.5-pro"
STATIC = "S" * 8200  # ~2050 estimated tokens: above a 2048 minimum
ON = ModelCaps(explicit_cache=True, min_cache_tokens=2048)
REQUEST = LLMRequest(
    messages=(Message(role="system", content=STATIC), Message(role="user", content="q"))
)


def kwargs(**config):
    base = {"system_instruction": STATIC, "temperature": 0.1}
    base.update(config)
    return {
        "model": MODEL,
        "contents": [{"role": "user", "parts": [{"text": "q"}]}],
        "config": base,
    }


def caps(value: ModelCaps):
    return lambda model: value


class FakeCaches:
    def __init__(self, *, error: Exception | None = None, tokens=2100, name="cachedContents/c"):
        self.calls: list[dict] = []
        self.error = error
        self.tokens = tokens
        self.name = name

    async def create(self, *, model, config):
        self.calls.append({"model": model, "config": config})
        await asyncio.sleep(0)  # let concurrent callers interleave
        if self.error is not None:
            raise self.error
        usage = SimpleNamespace(total_token_count=self.tokens)
        return SimpleNamespace(name=f"{self.name}{len(self.calls)}", usage_metadata=usage)


def client(caches: FakeCaches):
    return SimpleNamespace(aio=SimpleNamespace(caches=caches))


async def apply(plugin, kw, fake, request=REQUEST, model=MODEL):
    before = copy.deepcopy(kw)
    out = await plugin.apply(request, kw, model=model, client=client(fake))
    assert kw == before
    return out


def test_plugins_satisfy_the_protocol():
    for plugin in (GeminiExplicitCache(), GeminiImplicitCache()):
        assert isinstance(plugin, PromptCachePlugin)
        assert plugin.providers == frozenset({"gemini"})
    assert GeminiExplicitCache(ttl_s=600).key_fields() == {"ttl_s": 600.0}
    assert GeminiImplicitCache().key_fields() == {}


async def test_implicit_cache_changes_nothing():
    kw = kwargs()
    assert await GeminiImplicitCache().apply(REQUEST, kw, model=MODEL, client=None) == kw


async def test_gate_off_for_unknown_models_and_models_without_explicit_caching():
    fake = FakeCaches()
    assert await apply(GeminiExplicitCache(), kwargs(), fake, model="unknown") == kwargs()
    argon = ModelCaps(explicit_cache=False, min_cache_tokens=4096)
    assert await apply(GeminiExplicitCache(caps=caps(argon)), kwargs(), fake) == kwargs()
    assert fake.calls == []


async def test_small_prefixes_are_not_cached(caplog):
    fake = FakeCaches()
    big_min = ModelCaps(explicit_cache=True, min_cache_tokens=4096)
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.cache.providers.gemini"):
        out = await apply(GeminiExplicitCache(caps=caps(big_min)), kwargs(), fake)
    assert out == kwargs() and fake.calls == []
    assert "< 4096" in caplog.text


async def test_creates_once_per_prefix_and_reuses_the_name():
    fake = FakeCaches()
    store = DictCache()
    plugin = GeminiExplicitCache(ttl_s=3600, store=store, caps=caps(ON))
    first = await apply(plugin, kwargs(), fake)
    second = await apply(plugin, kwargs(temperature=0.9), fake)
    assert first["config"] == {"temperature": 0.1, "cached_content": "cachedContents/c1"}
    assert second["config"]["cached_content"] == "cachedContents/c1"
    assert first["contents"] == kwargs()["contents"]
    assert fake.calls == [
        {
            "model": MODEL,
            "config": {
                "system_instruction": STATIC,
                "ttl": "3600s",
                "display_name": gcache_key("aa", MODEL, digest(STATIC)),
            },
        }
    ]
    key = gcache_key("aa", MODEL, digest(STATIC))
    envelope = decode(store.data[key][0], "gcache")
    assert envelope.payload["name"] == "cachedContents/c1"
    assert store.data[key][1] == pytest.approx(3600 - 60)  # forgotten before Gemini expires it
    other = await apply(plugin, kwargs(system_instruction=STATIC + "x"), fake)
    assert other["config"]["cached_content"] == "cachedContents/c2"


async def test_concurrent_requests_create_one_cache():
    fake = FakeCaches()
    plugin = GeminiExplicitCache(caps=caps(ON))
    outs = await asyncio.gather(*(apply(plugin, kwargs(), fake) for _ in range(5)))
    assert len(fake.calls) == 1
    assert {out["config"]["cached_content"] for out in outs} == {"cachedContents/c1"}


def test_cache_names_are_shared_across_event_loops():
    fake = FakeCaches()
    plugin = GeminiExplicitCache(caps=caps(ON))
    names = [
        run_sync(lambda: apply(plugin, kwargs(), fake))["config"]["cached_content"] for _ in "ab"
    ]
    assert names == ["cachedContents/c1"] * 2
    assert len(fake.calls) == 1


async def test_tools_and_tool_config_move_into_the_cache():
    fake = FakeCaches()
    tools = [{"function_declarations": [{"name": "f", "description": "d"}]}]
    tool_config = {"function_calling_config": {"mode": "AUTO"}}
    kw = kwargs(tools=tools, tool_config=tool_config)
    out = await apply(GeminiExplicitCache(ttl_s=90.5, caps=caps(ON)), kw, fake)
    assert out["config"] == {"temperature": 0.1, "cached_content": "cachedContents/c1"}
    sent = fake.calls[0]["config"]
    assert sent["tools"] == tools and sent["tool_config"] == tool_config
    assert sent["ttl"] == "90.5s"
    prefix = {"system_instruction": STATIC, "tools": tools, "tool_config": tool_config}
    assert sent["display_name"] == gcache_key("aa", MODEL, digest(prefix))


async def test_creation_records_storage_in_the_context_ledger():
    fake = FakeCaches(tokens=2_000_000)
    ledger = RecordingLedger()
    ctx = ToolContext(run_id="run_1", call_id="call_1", thread_id="th", ledger=ledger)
    plugin = GeminiExplicitCache(ttl_s=1800, caps=caps(ON))
    with use_context(ctx):
        await apply(plugin, kwargs(), fake)
        await apply(plugin, kwargs(), fake)  # a reuse costs nothing new
    [entry] = ledger.entries()
    assert entry.kind == "api" and entry.provider == "gemini"
    assert entry.model == f"{MODEL}:cache_storage"
    assert entry.units == pytest.approx(1.0)  # 2M tokens x 0.5 h = 1M token-hours
    assert (entry.run_id, entry.call_id, entry.thread_id) == ("run_1", "call_1", "th")
    assert entry.entry_id == "gcache:cachedContents/c1"
    assert entry.metadata == {
        "cache_name": "cachedContents/c1",
        "ttl_s": "1800",
        "tokens": "2000000",
        "tokens_estimated": "false",
    }


async def test_storage_uses_the_estimate_when_usage_is_missing():
    fake = FakeCaches(tokens=None)
    ledger = RecordingLedger()
    with use_context(ToolContext(run_id="r", call_id="c", ledger=ledger)):
        await apply(GeminiExplicitCache(caps=caps(ON)), kwargs(), fake)
    [entry] = ledger.entries()
    assert entry.metadata["tokens_estimated"] == "true"
    assert int(entry.metadata["tokens"]) >= 2048


async def test_creation_failure_sends_the_request_unchanged(caplog):
    fake = FakeCaches(error=RuntimeError("quota"))
    plugin = GeminiExplicitCache(caps=caps(ON))
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.providers.gemini"):
        assert await apply(plugin, kwargs(), fake) == kwargs()
    assert "caches.create failed" in caplog.text


async def test_cancellation_propagates():
    fake = FakeCaches(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await apply(GeminiExplicitCache(caps=caps(ON)), kwargs(), fake)


async def test_failed_creation_is_shared_and_remembered(caplog):
    fake = FakeCaches(error=RuntimeError("quota"))
    plugin = GeminiExplicitCache(ttl_s=3600, caps=caps(ON))
    now = [1000.0]
    plugin._clock = lambda: now[0]
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.providers.gemini"):
        outs = await asyncio.gather(*(apply(plugin, kwargs(), fake) for _ in range(10)))
        assert outs == [kwargs()] * 10
        assert len(fake.calls) == 1  # waiters share the one failed attempt
        now[0] += 299
        assert await apply(plugin, kwargs(), fake) == kwargs()
        assert len(fake.calls) == 1  # still inside the failure window
    assert caplog.text.count("caches.create failed") == 1
    now[0] += 2  # window min(ttl_s, 300 s) is over
    fake.error = None
    out = await apply(plugin, kwargs(), fake)
    assert out["config"]["cached_content"] == "cachedContents/c2"


async def test_failure_window_is_capped_by_the_ttl():
    fake = FakeCaches(error=RuntimeError("quota"))
    plugin = GeminiExplicitCache(ttl_s=10, caps=caps(ON))
    now = [0.0]
    plugin._clock = lambda: now[0]
    await apply(plugin, kwargs(), fake)
    now[0] += 10
    await apply(plugin, kwargs(), fake)
    assert len(fake.calls) == 2


async def test_different_prefixes_create_concurrently():
    started: list[str] = []
    release = asyncio.Event()

    class Slow(FakeCaches):
        async def create(self, *, model, config):
            started.append(config["system_instruction"][-1])
            await release.wait()
            return await super().create(model=model, config=config)

    fake = Slow()
    plugin = GeminiExplicitCache(caps=caps(ON))
    tasks = [
        asyncio.ensure_future(apply(plugin, kwargs(system_instruction=STATIC + tail), fake))
        for tail in "ab"
    ]
    for _ in range(5):
        await asyncio.sleep(0)
    assert sorted(started) == ["a", "b"]  # the second prefix did not queue behind the first
    release.set()
    outs = await asyncio.gather(*tasks)
    assert len(fake.calls) == 2
    assert all(o["config"]["cached_content"].startswith("cachedContents/") for o in outs)


async def test_a_cancelled_waiter_does_not_cancel_the_shared_creation():
    release = asyncio.Event()

    class Slow(FakeCaches):
        async def create(self, *, model, config):
            await release.wait()
            return await super().create(model=model, config=config)

    fake = Slow()
    plugin = GeminiExplicitCache(caps=caps(ON))
    first = asyncio.ensure_future(apply(plugin, kwargs(), fake))
    second = asyncio.ensure_future(apply(plugin, kwargs(), fake))
    await asyncio.sleep(0)
    first.cancel()
    release.set()
    out = await second
    assert out["config"]["cached_content"] == "cachedContents/c1"
    with pytest.raises(asyncio.CancelledError):
        await first


async def test_a_nameless_result_is_not_used(caplog):
    fake = FakeCaches(name="")

    class Nameless(FakeCaches):
        async def create(self, *, model, config):
            return SimpleNamespace(name=None)

    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.providers.gemini"):
        out = await apply(GeminiExplicitCache(caps=caps(ON)), kwargs(), Nameless())
    assert out == kwargs() and fake.calls == []
    assert "returned no name" in caplog.text


async def test_broken_store_still_creates_and_uses_the_cache(caplog):
    fake = FakeCaches()
    plugin = GeminiExplicitCache(store=BrokenCache(), caps=caps(ON))
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.providers.gemini"):
        out = await apply(plugin, kwargs(), fake)
    assert out["config"]["cached_content"] == "cachedContents/c1"
    assert "store get failed" in caplog.text and "store set failed" in caplog.text


async def test_bad_store_entries_are_ignored():
    fake = FakeCaches()
    store = DictCache()
    key = gcache_key("aa", MODEL, digest(STATIC))
    await store.set(key, encode("gcache", ["not", "a", "dict"]))
    out = await apply(GeminiExplicitCache(store=store, caps=caps(ON)), kwargs(), fake)
    assert out["config"]["cached_content"] == "cachedContents/c1"


@pytest.mark.parametrize(
    "kw",
    [
        {"model": MODEL, "contents": []},
        kwargs(system_instruction=None),
        kwargs(cached_content="cachedContents/host"),
        kwargs(system_instruction=float("nan")),
    ],
)
async def test_ineligible_configs_are_unchanged(kw):
    fake = FakeCaches()
    assert await apply(GeminiExplicitCache(caps=caps(ON)), kw, fake) == kw
    assert fake.calls == []


async def test_sdk_config_objects_are_accepted():
    types = pytest.importorskip("google.genai.types")
    config = types.GenerateContentConfig(system_instruction=STATIC, temperature=0.2)
    kw = {"model": MODEL, "contents": [], "config": config}
    fake = FakeCaches()
    out = await GeminiExplicitCache(caps=caps(ON)).apply(
        REQUEST, kw, model=MODEL, client=client(fake)
    )
    assert out["config"] == {"temperature": 0.2, "cached_content": "cachedContents/c1"}
    assert kw["config"] is config


def test_ttl_must_be_positive():
    with pytest.raises(ConfigError, match="ttl_s"):
        GeminiExplicitCache(ttl_s=0)


async def test_a_name_stored_meanwhile_by_another_loop_is_reused():
    class LateStore(DictCache):
        async def get(self, key):
            value = await super().get(key)
            if sum(1 for op, _ in self.calls if op == "get") == 1:
                payload = {"name": "cachedContents/other-loop"}
                await self.set(key, encode("gcache", payload))
            return value

    fake = FakeCaches()
    out = await apply(GeminiExplicitCache(store=LateStore(), caps=caps(ON)), kwargs(), fake)
    assert out["config"]["cached_content"] == "cachedContents/other-loop"
    assert fake.calls == []


async def test_a_finished_creation_only_forgets_its_own_entry():
    from agent_adapter.cache.providers.gemini import _forget

    loop = asyncio.get_running_loop()
    old, new = loop.create_future(), loop.create_future()
    inflight = {"k": new}
    _forget(inflight, "k", old)
    assert inflight == {"k": new}
    _forget(inflight, "k", new)
    assert inflight == {}


@pytest.mark.parametrize("ttl", [float("nan"), float("inf"), float("-inf"), -5])
def test_ttl_must_be_finite_and_positive(ttl):
    with pytest.raises(ConfigError, match="ttl_s"):
        GeminiExplicitCache(ttl_s=ttl)


async def test_expired_failure_markers_are_pruned():
    fake = FakeCaches(error=RuntimeError("quota"))
    plugin = GeminiExplicitCache(ttl_s=60, caps=caps(ON))
    now = [0.0]
    plugin._clock = lambda: now[0]
    for i in range(50):
        await apply(plugin, kwargs(system_instruction=f"{STATIC}{i}"), fake)
        now[0] += 61  # each earlier marker has expired by the next failure
    assert len(plugin._failed) == 1
    assert len(fake.calls) == 50


async def test_failure_markers_are_bounded(monkeypatch):
    import agent_adapter.cache.providers.gemini as gemini

    monkeypatch.setattr(gemini, "_MAX_FAILURE_MARKERS", 5)
    fake = FakeCaches(error=RuntimeError("quota"))
    plugin = GeminiExplicitCache(caps=caps(ON))
    for i in range(12):
        await apply(plugin, kwargs(system_instruction=f"{STATIC}{i}"), fake)
    assert len(plugin._failed) == 5
    newest = gcache_key("aa", MODEL, digest(f"{STATIC}11"))
    assert newest in plugin._failed
