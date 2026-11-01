import asyncio
import logging
import math
import threading
import time

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ConfigError,
    ModelError,
    ModelOutputInvalid,
    ModelRateLimited,
    ModelTimeout,
    ToolUsageError,
)
from agent_adapter.core.types import Usage
from agent_adapter.embeddings import CallableEmbedder, EmbeddingResult
from agent_adapter.runtime.current import current_context, use_context
from tests.embeddings.support import Recorder, vector


async def test_async_fn_gets_a_list_and_vectors_keep_input_order():
    fn = Recorder()
    embedder = CallableEmbedder(fn.run, model="local-v1", provider="local")
    result = await embedder.embed(("b c", "a"))
    assert result.vectors == (vector("b c"), vector("a"))
    assert (result.model, result.usage, result.cache_hits) == ("local-v1", Usage(), 0)
    assert fn.batches == [["b c", "a"]]


async def test_sync_fn_runs_in_a_worker_thread_and_sees_the_current_context():
    seen: list[ToolContext | None] = []
    threads: list[int] = []

    def fn(texts):
        seen.append(current_context())
        threads.append(threading.get_ident())
        return [vector(t) for t in texts]

    ctx = ToolContext(run_id="r1", call_id="c1")
    with use_context(ctx):
        result = await CallableEmbedder(fn, model="m").embed(["x"])
    assert result.vectors == (vector("x"),)
    assert threads[0] != threading.get_ident()
    assert seen[0] is not None and (seen[0].run_id, seen[0].call_id) == ("r1", "c1")


async def test_explicit_ctx_is_current_inside_the_fn():
    seen: list[ToolContext | None] = []

    async def fn(texts):
        seen.append(current_context())
        return [[1.0] for _ in texts]

    ctx = ToolContext(run_id="r2", call_id="c2")
    await CallableEmbedder(fn, model="m").embed(["x"], ctx=ctx)
    assert seen == [ctx]


async def test_sync_fn_returning_an_awaitable_is_awaited():
    async def inner(texts):
        return [[2.0] for _ in texts]

    embedder = CallableEmbedder(lambda texts: inner(texts), model="m")
    assert (await embedder.embed(["a", "b"])).vectors == ((2.0,), (2.0,))


async def test_callable_object_with_async_call_is_awaited_on_the_loop():
    class Client:
        def __init__(self):
            self.thread = 0

        async def __call__(self, texts):
            self.thread = threading.get_ident()
            return [[1.0] for _ in texts]

    client = Client()
    await CallableEmbedder(client, model="m").embed(["a"])
    assert client.thread == threading.get_ident()


async def test_texts_are_sent_in_batches_and_usage_is_summed():
    fn = Recorder(tokens_per_text=3, model="served-name")
    embedder = CallableEmbedder(fn.sync, model="m", batch_size=2)
    result = await embedder.embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert fn.batches == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert result.vectors == tuple(vector(t) for t in ["a", "bb", "ccc", "dddd", "eeeee"])
    assert result.usage == Usage(input_tokens=15)
    assert result.model == "served-name"


async def test_duplicate_texts_are_embedded_once():
    fn = Recorder()
    result = await CallableEmbedder(fn.run, model="m").embed(["a", "b", "a", "a"])
    assert fn.batches == [["a", "b"]]
    assert result.vectors == (vector("a"), vector("b"), vector("a"), vector("a"))


async def test_no_texts_means_no_call():
    fn = Recorder()
    result = await CallableEmbedder(fn.run, model="m").embed([])
    assert result == EmbeddingResult(vectors=(), model="m")
    assert fn.batches == []


async def test_fn_mutating_its_batch_does_not_change_the_result_order():
    async def fn(texts):
        out = [vector(t) for t in texts]
        texts.clear()
        return out

    result = await CallableEmbedder(fn, model="m").embed(["a", "bb"])
    assert result.vectors == (vector("a"), vector("bb"))


@pytest.mark.parametrize("texts", ["one string", b"bytes", [1, "a"], ["a", None], 7, {"a": 1}])
async def test_texts_must_be_a_sequence_of_strings(texts):
    fn = Recorder()
    with pytest.raises(ToolUsageError, match="texts"):
        await CallableEmbedder(fn.run, model="m").embed(texts)
    assert fn.batches == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"batch_size": 2.0},
        {"timeout_s": 0},
        {"timeout_s": -1.0},
        {"timeout_s": math.nan},
        {"timeout_s": math.inf},
        {"model": ""},
        {"provider": ""},
    ],
)
def test_bad_settings_raise_config_error(kwargs):
    settings = {"model": "m", **kwargs}
    with pytest.raises(ConfigError):
        CallableEmbedder(Recorder().run, **settings)


def test_fn_must_be_callable():
    with pytest.raises(ToolUsageError, match="callable"):
        CallableEmbedder("not a function", model="m")  # type: ignore[arg-type]


async def test_slow_async_fn_times_out_as_retryable_model_timeout():
    async def fn(texts):
        await asyncio.sleep(5)
        return [[1.0]]

    embedder = CallableEmbedder(fn, model="m", provider="p", timeout_s=0.01)
    with pytest.raises(ModelTimeout) as info:
        await embedder.embed(["a"])
    assert info.value.retryable and (info.value.provider, info.value.model) == ("p", "m")
    assert "timed out" in str(info.value)


async def test_slow_sync_fn_times_out_too():
    release = threading.Event()

    def fn(texts):
        release.wait(5)
        return [[1.0]]

    try:
        with pytest.raises(ModelTimeout):
            await CallableEmbedder(fn, model="m", timeout_s=0.01).embed(["a"])
    finally:
        release.set()


async def test_timeout_applies_per_batch():
    async def fn(texts):
        await asyncio.sleep(0.03)
        return [[1.0] for _ in texts]

    embedder = CallableEmbedder(fn, model="m", batch_size=1, timeout_s=0.5)
    start = time.perf_counter()
    assert len((await embedder.embed(["a", "b", "c"])).vectors) == 3
    assert time.perf_counter() - start >= 0.08


async def test_foreign_fn_error_becomes_model_error_without_its_text(caplog):
    async def fn(texts):
        raise ValueError(f"cannot embed {texts[0]}")

    embedder = CallableEmbedder(fn, model="m", provider="p")
    with (
        caplog.at_level(logging.INFO, logger="agent_adapter.embeddings"),
        pytest.raises(ModelError) as info,
    ):
        await embedder.embed(["my secret text"])
    assert type(info.value) is ModelError
    assert isinstance(info.value.__cause__, ValueError)
    assert "secret" not in str(info.value) and "ValueError" in str(info.value)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings and all("secret" not in r.getMessage() for r in caplog.records)


async def test_adapter_errors_from_fn_propagate_unchanged():
    error = ModelRateLimited("slow down", provider="p", model="m", retry_after=2.0)

    async def fn(texts):
        raise error

    with pytest.raises(ModelRateLimited) as info:
        await CallableEmbedder(fn, model="m").embed(["a"])
    assert info.value is error


async def test_cancellation_propagates():
    started = asyncio.Event()

    async def fn(texts):
        started.set()
        await asyncio.sleep(5)

    task = asyncio.create_task(CallableEmbedder(fn, model="m").embed(["a"]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "returned",
    [
        [[1.0]],  # one vector for two texts
        [[1.0], [2.0], [3.0]],
        [[1.0], [2.0, 3.0]],  # mixed dimensions
        [[1.0], [math.nan]],
        [[1.0], ["1.5"]],  # a string is not a number
        [[1.0], [True]],  # nor is a bool
        [[1.0], [math.inf]],
        "vectors",
        None,
        EmbeddingResult(vectors=[[1.0]]),
    ],
)
async def test_bad_fn_output_raises_model_output_invalid(returned):
    async def fn(texts):
        return returned

    with pytest.raises(ModelOutputInvalid) as info:
        await CallableEmbedder(fn, model="m", provider="p").embed(["a", "b"])
    assert (info.value.provider, info.value.model) == ("p", "m")


async def test_dimensions_must_match_across_batches():
    async def fn(texts):
        return [[1.0] * len(texts[0]) for _ in texts]

    embedder = CallableEmbedder(fn, model="m", batch_size=1)
    with pytest.raises(ModelOutputInvalid, match="dimension"):
        await embedder.embed(["a", "bb"])


async def test_array_like_output_is_accepted():
    class Array:
        def __init__(self, rows):
            self.rows = rows

        def tolist(self):
            return self.rows

    def fn(texts):
        return Array([[float(len(t))] for t in texts])

    result = await CallableEmbedder(fn, model="m").embed(["ab", "c"])
    assert result.vectors == ((2.0,), (1.0,))


@pytest.mark.parametrize("timeout_s", [None, 5.0])
async def test_fn_raising_its_own_timeout_error_is_not_reported_as_ours(timeout_s):
    async def fn(texts):
        raise TimeoutError("socket read timed out")

    embedder = CallableEmbedder(fn, model="m", provider="p", timeout_s=timeout_s)
    with pytest.raises(ModelTimeout) as info:
        await embedder.embed(["a"])
    message = str(info.value)
    assert "after" not in message and "None" not in message
    assert message.endswith("embedding fn raised TimeoutError") and info.value.retryable
    assert isinstance(info.value.__cause__, TimeoutError)
