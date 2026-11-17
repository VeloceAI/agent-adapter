"""Concurrent identical model requests share one provider call."""

import asyncio

import pytest

from agent_adapter.cache._inflight import _FLIGHTS
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.errors import ModelUnavailable
from agent_adapter.hooks import Hook, ShortCircuit
from agent_adapter.models import CallableModel
from tests.fakes.cache import DictCache
from tests.fakes.models import text
from tests.models.instrument_support import HI, check_required, instrumented


class GatedModel(CallableModel):
    """Counts provider calls; each waits for `gate`, then answers or raises `fail`."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.calls = 0
        self.gate = asyncio.Event()
        self.fail = fail
        super().__init__(self._answer, provider="openai", model="scripted")

    async def _answer(self, request: LLMRequest) -> str:
        self.calls += 1
        await self.gate.wait()
        if self.fail is not None:
            raise self.fail
        return "shared answer"


async def _ticks(n: int = 30) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def _together(model, port, requests):
    tasks = [asyncio.ensure_future(model.generate(r)) for r in requests]
    await _ticks()
    port.gate.set()
    return await asyncio.gather(*tasks, return_exceptions=True)


async def test_five_concurrent_requests_make_one_provider_call():
    port = GatedModel()
    model = instrumented(port, cache=DictCache(), cache_ttl=60)
    replies = await _together(model, port, [HI] * 5)
    assert port.calls == 1
    assert {r.text for r in replies} == {"shared answer"}
    assert [r.cached for r in replies].count(False) == 1
    events = model.sink.events
    check_required(events)
    shared = [e for e in events if e.type == "inflight.shared"]
    assert len(shared) == 4 and {e.data["kind"] for e in shared} == {"llm"}
    assert [e.type for e in events].count("llm.cached") == 4
    entries = model.ledger.entries()
    assert [e.cache_hit for e in entries].count(True) == 4
    assert sum(e.cost_usd for e in entries if e.cache_hit) == 0
    assert not _FLIGHTS.get()


async def test_different_requests_are_not_shared():
    port = GatedModel()
    model = instrumented(port, cache=DictCache(), cache_ttl=60)
    other = LLMRequest(messages=(Message(role="user", content="bye"),))
    await _together(model, port, [HI, other])
    assert port.calls == 2


async def test_no_cache_or_opted_out_means_no_sharing():
    port = GatedModel()
    await _together(instrumented(port), port, [HI] * 3)
    assert port.calls == 3
    port = GatedModel()
    opted_out = HI.model_copy(update={"response_cache": False})
    await _together(instrumented(port, cache=DictCache()), port, [opted_out] * 3)
    assert port.calls == 3


async def test_streams_are_not_shared():
    port = GatedModel()
    model = instrumented(port, cache=DictCache(), cache_ttl=60)

    async def drain() -> None:
        async for _ in model.stream(HI):
            pass

    tasks = [asyncio.ensure_future(drain()) for _ in range(3)]
    await _ticks()
    port.gate.set()
    await asyncio.gather(*tasks)
    assert port.calls == 3


async def test_provider_failure_reaches_every_waiter():
    port = GatedModel(fail=ModelUnavailable("down", provider="openai", model="scripted"))
    model = instrumented(port, cache=DictCache(), cache_ttl=60)
    outcomes = await _together(model, port, [HI] * 4)
    assert port.calls == 1
    assert all(isinstance(o, ModelUnavailable) for o in outcomes)
    assert [e.type for e in model.sink.events].count("llm.failed") == 4
    assert not _FLIGHTS.get()


class _Recover(Hook):
    def __init__(self) -> None:
        self.seen = 0

    async def on_llm_error(self, inv, error):
        self.seen += 1
        return ShortCircuit(text("fallback"))


async def test_each_waiter_runs_its_own_error_hooks():
    port = GatedModel(fail=ModelUnavailable("down", provider="openai", model="scripted"))
    hook = _Recover()
    model = instrumented(port, cache=DictCache(), cache_ttl=60, hooks=[hook])
    replies = await _together(model, port, [HI] * 3)
    assert port.calls == 1 and hook.seen == 3
    assert {r.text for r in replies} == {"fallback"}


async def test_cancelled_leader_hands_the_call_to_one_waiter():
    port = GatedModel()
    model = instrumented(port, cache=DictCache(), cache_ttl=60)
    leader = asyncio.ensure_future(model.generate(HI))
    await _ticks()
    waiters = [asyncio.ensure_future(model.generate(HI)) for _ in range(3)]
    await _ticks()
    leader.cancel()
    await _ticks()
    port.gate.set()
    replies = await asyncio.gather(*waiters)
    with pytest.raises(asyncio.CancelledError):
        await leader
    assert port.calls == 2
    assert {r.text for r in replies} == {"shared answer"}
    assert [r.cached for r in replies].count(False) == 1
    assert not _FLIGHTS.get()
