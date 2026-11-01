"""CallableModel with a sync host function (worker thread) and host-reported usage."""

import asyncio
import contextvars
import threading

import pytest

from agent_adapter.core.errors import ModelOutputInvalid
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ToolCall, Usage
from agent_adapter.cost import Ledger, Prices
from agent_adapter.exec import Executor, run_tool_loop
from agent_adapter.models.callable import CallableModel

REQUEST = LLMRequest(messages=(Message(role="user", content="hi"),))
TENANT: contextvars.ContextVar[str] = contextvars.ContextVar("tenant", default="none")
PRICES = Prices.load(
    {"version": "t", "models": {"local-7b": {"provider": "local", "input": 1.0, "output": 2.0}}}
)


async def test_sync_fn_runs_off_the_event_loop_thread():
    seen: list[int] = []

    def host(request):
        seen.append(threading.get_ident())
        return "sync"

    assert (await CallableModel(host).generate(REQUEST)).text == "sync"
    assert seen and seen[0] != threading.get_ident()


async def test_async_fn_runs_on_the_loop_thread():
    seen: list[int] = []

    async def host(request):
        seen.append(threading.get_ident())
        return "async"

    class Router:
        async def __call__(self, request):
            seen.append(threading.get_ident())
            return "object"

    assert (await CallableModel(host).generate(REQUEST)).text == "async"
    assert (await CallableModel(Router()).generate(REQUEST)).text == "object"
    assert seen == [threading.get_ident()] * 2


async def test_sync_fn_sees_the_callers_context_variables():
    def host(request):
        return TENANT.get()

    token = TENANT.set("acme")
    try:
        assert (await CallableModel(host).generate(REQUEST)).text == "acme"
    finally:
        TENANT.reset(token)


async def test_sync_fn_does_not_block_other_tasks():
    started, release = threading.Event(), threading.Event()

    def blocking(request):
        started.set()
        assert release.wait(5)
        return "done"

    call = asyncio.create_task(CallableModel(blocking).generate(REQUEST))
    await asyncio.to_thread(started.wait, 5)
    assert not call.done()  # the loop is free while the host function blocks
    release.set()
    assert (await call).text == "done"


async def test_sync_callable_object_and_sync_fn_returning_awaitable():
    class Local:
        def __call__(self, request):
            return {"text": "obj"}

    async def later():
        return "awaited"

    assert (await CallableModel(Local()).generate(REQUEST)).text == "obj"
    model = CallableModel(lambda request: later())
    assert (await model.generate(REQUEST)).text == "awaited"


async def test_sync_fn_errors_propagate_unchanged():
    def host(request):
        raise ConnectionError("local server down")

    with pytest.raises(ConnectionError, match="local server down"):
        await CallableModel(host).generate(REQUEST)


async def test_usage_as_mapping_or_usage_object():
    def mapped(request):
        return {"text": "a", "usage": {"input_tokens": 7, "output_tokens": 2}}

    def typed(request):
        return {"text": "b", "usage": Usage(input_tokens=1)}

    assert (await CallableModel(mapped).generate(REQUEST)).usage == Usage(
        input_tokens=7, output_tokens=2
    )
    assert (await CallableModel(typed).generate(REQUEST)).usage == Usage(input_tokens=1)


async def test_provider_shaped_usage_keys_are_rejected(caplog: pytest.LogCaptureFixture):
    def host(request):
        return {"text": "secret reply", "usage": {"prompt_tokens": 7, "completion_tokens": 2}}

    with pytest.raises(ModelOutputInvalid, match=r"\['completion_tokens', 'prompt_tokens'\]"):
        await CallableModel(host, provider="local").generate(REQUEST)
    assert "secret reply" not in caplog.text
    assert "unknown keys" in caplog.text


async def test_invalid_usage_values_are_rejected():
    def host(request):
        return {"text": "x", "usage": {"input_tokens": -1}}

    with pytest.raises(ModelOutputInvalid, match="invalid response mapping"):
        await CallableModel(host).generate(REQUEST)


async def test_host_usage_reaches_the_ledger_priced():
    def local(request):
        return {"text": "ok", "usage": {"input_tokens": 1_000_000, "output_tokens": 500_000}}

    ledger = Ledger(prices=PRICES)
    executor = Executor(Registry(), ledger=ledger)
    llm = executor.model(CallableModel(local, provider="local", model="local-7b"))
    async with executor.run_scope() as ctx:
        await llm.bind(ctx).generate(REQUEST)
    [entry] = ledger.entries(ctx.run_id)
    assert (entry.kind, entry.provider, entry.model) == ("llm", "local", "local-7b")
    assert (entry.input_tokens, entry.output_tokens, entry.cost_usd) == (1_000_000, 500_000, 2.0)


async def test_tool_loop_runs_on_a_sync_local_model():
    registry = Registry()

    @registry.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def local(request: LLMRequest) -> LLMResponse | dict[str, object]:
        last = request.messages[-1]
        usage = {"input_tokens": 100, "output_tokens": 10}
        if last.role == "tool":
            return {"text": f"sum is {last.content}", "usage": usage}
        call = ToolCall(name="add", arguments={"a": 2, "b": 3}, id="c1")
        return {"tool_calls": [call], "usage": usage}

    ledger = Ledger(prices=PRICES)
    executor = Executor(registry, ledger=ledger)
    model = CallableModel(local, provider="local", model="local-7b")
    result = await run_tool_loop(model, registry, REQUEST.messages, executor=executor)
    assert (result.finish, result.steps, result.final.text) == ("stop", 2, "sum is 5")
    assert result.usage == Usage(input_tokens=200, output_tokens=20)
    llm_entries = [e for e in ledger.entries() if e.kind == "llm"]
    assert len(llm_entries) == 2
    assert ledger.total_usd() == pytest.approx(2 * (100 * 1.0 + 10 * 2.0) / 1_000_000)
