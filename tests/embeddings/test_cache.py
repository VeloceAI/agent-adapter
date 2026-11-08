import json
import logging

import pytest

from agent_adapter.cache import MemoryCache, digest
from agent_adapter.cache.codec import encode
from agent_adapter.core.errors import ModelOutputInvalid
from agent_adapter.embeddings import CallableEmbedder, EmbeddingCache, embed_key
from tests.embeddings.support import Recorder, vector
from tests.fakes.cache import BrokenCache, DictCache
from tests.fakes.clock import FakeClock


def test_embed_key_names_namespace_provider_model_and_text_digest():
    key = embed_key("aa", "hello", provider="local", model="m1")
    assert key == f"aa:embed:v1:local:m1:{digest(['local', 'm1', 'hello'])}"
    assert "hello" not in key
    assert key != embed_key("aa", "hello", provider="local", model="m2")
    assert key != embed_key("aa", "hello", provider="other", model="m1")
    assert key != embed_key("ns", "hello", provider="local", model="m1")


async def test_second_call_is_served_from_the_cache():
    fn = Recorder()
    embedder = CallableEmbedder(fn.run, model="m", cache=MemoryCache())
    first = await embedder.embed(["a", "bb"])
    second = await embedder.embed(["bb", "a"])
    assert fn.batches == [["a", "bb"]]
    assert first.cache_hits == 0 and second.cache_hits == 2
    assert second.vectors == (vector("bb"), vector("a"))


async def test_partial_hit_sends_only_missing_texts():
    fn = Recorder()
    embedder = CallableEmbedder(fn.run, model="m", cache=MemoryCache(), batch_size=10)
    await embedder.embed(["a"])
    result = await embedder.embed(["a", "b", "a", "c"])
    assert fn.batches == [["a"], ["b", "c"]]
    assert result.cache_hits == 2  # both "a" positions
    assert result.vectors == tuple(vector(t) for t in ["a", "b", "a", "c"])


async def test_another_model_name_does_not_share_entries():
    cache = MemoryCache()
    fn = Recorder()
    await CallableEmbedder(fn.run, model="m1", cache=cache).embed(["a"])
    await CallableEmbedder(fn.run, model="m2", cache=cache).embed(["a"])
    assert fn.batches == [["a"], ["a"]]


async def test_ttl_and_namespace_reach_the_backend():
    clock = FakeClock()
    cache = DictCache(clock)
    fn = Recorder()
    embedder = CallableEmbedder(fn.run, model="m", cache=cache, cache_ttl=10, namespace="app")
    await embedder.embed(["a"])
    assert list(cache.data) == [embed_key("app", "a", provider="callable", model="m")]
    clock.advance(11)
    await embedder.embed(["a"])
    assert fn.batches == [["a"], ["a"]]


async def test_broken_backend_is_a_miss_and_a_warning(caplog):
    fn = Recorder()
    embedder = CallableEmbedder(fn.run, model="m", cache=BrokenCache(ConnectionError("down")))
    with caplog.at_level(logging.WARNING, logger="agent_adapter.embeddings"):
        result = await embedder.embed(["a", "b"])
        await embedder.embed(["a", "b"])
    assert result.vectors == (vector("a"), vector("b"))
    assert fn.batches == [["a", "b"], ["a", "b"]]
    messages = [r.getMessage() for r in caplog.records]
    assert any("get failed" in m for m in messages) and any("set failed" in m for m in messages)


@pytest.mark.parametrize(
    "stored",
    [
        b"not json",
        encode("llm", [1.0, 2.0, 3.0]),
        encode("embedding", "text"),
        encode("embedding", []),
        json.dumps({"v": 1, "kind": "embedding", "stored_at": 0, "payload": [None]}).encode(),
    ],
    ids=["not-json", "llm-kind", "text-payload", "empty-payload", "null-vector"],  # stable ids
)
async def test_malformed_entries_are_misses(stored):
    cache = MemoryCache()
    await cache.set(embed_key("aa", "a", provider="callable", model="m"), stored)
    fn = Recorder()
    result = await CallableEmbedder(fn.run, model="m", cache=cache).embed(["a"])
    assert fn.batches == [["a"]] and result.cache_hits == 0
    assert result.vectors == (vector("a"),)


def test_embed_key_is_unambiguous_when_names_contain_colons():
    one = embed_key("aa", "t", provider="a:b", model="c")
    two = embed_key("aa", "t", provider="a", model="b:c")
    assert one != two


async def test_colon_names_do_not_share_vectors():
    cache = MemoryCache()
    fn = Recorder()
    await CallableEmbedder(fn.run, provider="a:b", model="c", cache=cache).embed(["t"])
    await CallableEmbedder(fn.run, provider="a", model="b:c", cache=cache).embed(["t"])
    assert fn.batches == [["t"], ["t"]]


async def test_stale_vectors_of_another_dimension_are_re_embedded_and_replaced(caplog):
    cache = MemoryCache()
    await EmbeddingCache(cache).set_many({"a": (1.0,)}, provider="callable", model="m")
    fn = Recorder()
    embedder = CallableEmbedder(fn.run, model="m", cache=cache)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.embeddings"):
        result = await embedder.embed(["a", "b"])
    assert result.vectors == (vector("a"), vector("b")) and result.cache_hits == 0
    assert fn.batches == [["b"], ["a"]]
    assert any("another dimension" in r.getMessage() for r in caplog.records)
    again = await embedder.embed(["a", "b"])
    assert again.cache_hits == 2 and fn.batches == [["b"], ["a"]]


async def test_cached_vectors_of_mixed_dimensions_without_a_fresh_batch_raise():
    cache = MemoryCache()
    store = EmbeddingCache(cache)
    await store.set_many({"a": (1.0,), "b": (1.0, 2.0)}, provider="callable", model="m")
    with pytest.raises(ModelOutputInvalid, match="dimension"):
        await CallableEmbedder(Recorder().run, model="m", cache=cache).embed(["a", "b"])


async def test_a_batch_of_another_dimension_is_never_cached():
    cache = MemoryCache()

    async def fn(texts):
        return [[1.0] * len(texts[0]) for _ in texts]

    embedder = CallableEmbedder(fn, model="m", cache=cache, batch_size=1)
    with pytest.raises(ModelOutputInvalid, match="dimension"):
        await embedder.embed(["a", "bb"])
    found = await EmbeddingCache(cache).get_many(["a", "bb"], provider="callable", model="m")
    assert found == {"a": (1.0,)}


async def test_vectors_of_finished_batches_stay_cached_when_a_later_batch_fails():
    cache = MemoryCache()
    calls: list[list[str]] = []

    async def fn(texts):
        calls.append(list(texts))
        if "boom" in texts:
            raise RuntimeError("model crashed")
        return [vector(t) for t in texts]

    embedder = CallableEmbedder(fn, model="m", cache=cache, batch_size=1)
    with pytest.raises(Exception, match="RuntimeError"):
        await embedder.embed(["a", "boom"])
    result = await embedder.embed(["a"])
    assert result.cache_hits == 1 and calls == [["a"], ["boom"]]


async def test_embedding_cache_round_trip_and_counts():
    store = EmbeddingCache(MemoryCache(), namespace="x", ttl=None)
    assert await store.set_many({"a": (1.0, 2.0), "b": (3.0, 4.0)}, provider="p", model="m") == 2
    found = await store.get_many(["a", "b", "c"], provider="p", model="m")
    assert found == {"a": (1.0, 2.0), "b": (3.0, 4.0)}


async def test_embedding_cache_set_failure_counts_zero():
    store = EmbeddingCache(BrokenCache())
    assert await store.set_many({"a": (1.0,)}, provider="p", model="m") == 0
    assert await store.get_many(["a"], provider="p", model="m") == {}
