"""ProviderPort without any SDK: a fake port over a fake client (core session coverage)."""

import logging

import pytest

from agent_adapter.core.errors import ModelError, ModelOutputInvalid, ModelRateLimited
from agent_adapter.core.messages import (
    LLMRequest,
    LLMResponse,
    Message,
    OutputSchema,
    StreamDone,
    TextDelta,
)
from agent_adapter.core.ports import LLMPort
from agent_adapter.models._port import ProviderPort
from agent_adapter.runtime.sync import run_sync
from tests.fakes.models import FakePromptCachePlugin, text

REQUEST = LLMRequest(messages=(Message(role="user", content="hi"),))


class FakeClient:
    def __init__(self, replies=(), chunks=()):
        self.replies = list(replies)
        self.chunks = list(chunks)
        self.calls: list[dict] = []
        self.closed = False
        self.stream_closed = False

    async def create(self, kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def stream(self, kwargs):
        self.calls.append(kwargs)
        client = self

        class Stream:
            def __init__(self):
                self.items = list(client.chunks)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.items:
                    raise StopAsyncIteration
                item = self.items.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

            def close(self):  # a sync close is accepted too
                client.stream_closed = True

        return Stream()


class Assembler:
    def __init__(self):
        self.parts = []

    def feed(self, chunk):
        self.parts.append(chunk)
        return [TextDelta(text=chunk)]

    def response(self):
        return text("".join(self.parts), provider="fake", model="")


class Mapped(Exception):
    pass


class FakePort(ProviderPort):
    provider = "fake"
    api = "fake-api"

    def __init__(self, model="m1", *, client=None, made=None, **kwargs):
        self.made = made if made is not None else []
        super().__init__(
            model,
            client=client,
            client_kwargs=kwargs.get("client_kwargs"),
            cache_plugin=kwargs.get("cache_plugin"),
            defaults=kwargs.get("defaults"),
        )

    def _require_sdk(self):
        return None

    def _new_client(self):
        client = FakeClient(replies=[text("loop", provider="fake")])
        self.made.append(client)
        return client

    async def _close_client(self, client):
        client.closed = True

    def _build(self, request, model):
        out = {"model": model, "n": len(request.messages)}
        if request.temperature is not None:
            out["temperature"] = request.temperature
        return out

    async def _create(self, client, kwargs):
        return await client.create(kwargs)

    async def _open_stream(self, client, kwargs):
        return await client.stream(kwargs)

    def _parse(self, raw):
        return raw

    def _assembler(self, model):
        return Assembler()

    def _map_error(self, exc, model):
        if isinstance(exc, Mapped):
            return ModelRateLimited("slow", provider="fake", model=model, status_code=429)
        return None


async def test_generate_merges_defaults_plugin_then_extra():
    client = FakeClient(replies=[text("ok", provider="fake", model="")])
    plugin = FakePromptCachePlugin({"cache": "on", "temperature": 0.9}, providers=("fake",))
    port = FakePort(
        client=client, cache_plugin=plugin, defaults={"temperature": 0.1, "store": False}
    )
    assert isinstance(port, LLMPort)
    request = REQUEST.model_copy(update={"temperature": 0.5, "extra": {"cache": "off"}})
    response = await port.generate(request)
    assert client.calls == [
        {"model": "m1", "n": 1, "temperature": 0.9, "store": False, "cache": "off"}
    ]
    assert plugin.calls[0]["kwargs"]["temperature"] == 0.5  # request field beats the default
    assert response.model == "m1" and response.latency_ms >= 0
    assert port.cache_key_fields() == {"api": "fake-api", "plugin": plugin.key_fields()}
    assert port.client() is client


async def test_model_override_and_structured_output():
    client = FakeClient(replies=[text('{"a": 1}'), text("bad")])
    port = FakePort(client=client)
    schema = OutputSchema(name="A", json_schema={"type": "object"})
    request = REQUEST.model_copy(update={"model": "m2", "output_schema": schema})
    assert (await port.generate(request)).parsed == {"a": 1}
    assert client.calls[0]["model"] == "m2"
    with pytest.raises(ModelOutputInvalid):
        await port.generate(request)


async def test_errors_mapped_from_cause_or_propagated(caplog):
    client = FakeClient(replies=[Mapped(), KeyError("x")])
    port = FakePort(client=client)
    with caplog.at_level(logging.WARNING), pytest.raises(ModelRateLimited) as info:
        await port.generate(REQUEST)
    assert isinstance(info.value.__cause__, Mapped)
    assert "model-rate-limited" in caplog.text
    with pytest.raises(KeyError):
        await port.generate(REQUEST)


async def test_stream_yields_deltas_then_done_and_closes():
    client = FakeClient(chunks=["a", "b"])
    events = [e async for e in FakePort(client=client).stream(REQUEST)]
    assert events[:2] == [TextDelta(text="a"), TextDelta(text="b")]
    assert isinstance(events[2], StreamDone) and events[2].response.text == "ab"
    assert events[2].response.model == "m1"
    assert client.stream_closed


async def test_stream_errors_mapped_and_stream_closed():
    client = FakeClient(chunks=["a", Mapped()])
    with pytest.raises(ModelRateLimited):
        _ = [e async for e in FakePort(client=client).stream(REQUEST)]
    assert client.stream_closed
    with pytest.raises(ValueError):
        _ = [e async for e in FakePort(client=FakeClient(chunks=[ValueError()])).stream(REQUEST)]


async def test_generate_with_stream_true_collects():
    client = FakeClient(chunks=["x", "y"])
    response = await FakePort(client=client).generate(REQUEST.model_copy(update={"stream": True}))
    assert isinstance(response, LLMResponse) and response.text == "xy"


async def test_streams_without_close_are_fine():
    class Bare:
        def __init__(self):
            self.items = ["z"]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop()

    class BareClient(FakeClient):
        async def stream(self, kwargs):
            return Bare()

    events = [e async for e in FakePort(client=BareClient()).stream(REQUEST)]
    assert events[0] == TextDelta(text="z")


def test_port_creates_one_client_per_loop_and_closes_it():
    made: list[FakeClient] = []
    port = FakePort(made=made)
    for _ in range(2):
        assert run_sync(port.generate(REQUEST)).text == "loop"
    assert len(made) == 2 and made[0] is not made[1]
    assert all(client.closed for client in made)


def test_plugin_errors_propagate():
    class Boom(Exception):
        pass

    def explode(kwargs):
        raise Boom()

    plugin = FakePromptCachePlugin({}, providers=("fake",), edit=explode)
    port = FakePort(client=FakeClient(replies=[text("x")]), cache_plugin=plugin)
    with pytest.raises(Boom):
        run_sync(port.generate(REQUEST))


def test_model_error_attributes_survive():
    err = ModelError("x", provider="fake", model="m")
    assert err.kind == "model-error"


async def test_assembler_errors_are_logged_like_mapped_errors(caplog):
    class Failing(Assembler):
        def feed(self, chunk):
            raise ModelRateLimited("mid-stream", provider="fake", model="m1")

    class FailingPort(FakePort):
        def _assembler(self, model):
            return Failing()

    port = FailingPort(client=FakeClient(chunks=["a"]))
    with caplog.at_level(logging.WARNING), pytest.raises(ModelRateLimited) as info:
        _ = [e async for e in port.stream(REQUEST)]
    assert info.value.__cause__ is None
    assert "model-rate-limited" in caplog.text


def test_client_and_plugin_errors_are_mapped(caplog):
    class BrokenClientPort(FakePort):
        def _new_client(self):
            raise Mapped()

    with caplog.at_level(logging.WARNING), pytest.raises(ModelRateLimited) as info:
        run_sync(BrokenClientPort().generate(REQUEST))
    assert isinstance(info.value.__cause__, Mapped)

    def sdk_failure(kwargs):
        raise Mapped()

    plugin = FakePromptCachePlugin({}, providers=("fake",), edit=sdk_failure)
    port = FakePort(client=FakeClient(), cache_plugin=plugin)
    with pytest.raises(ModelRateLimited):
        run_sync(port.generate(REQUEST))

    class TypoPort(FakePort):
        def _new_client(self):
            raise TypeError("bogus kwarg")

    with pytest.raises(TypeError, match="bogus"):
        run_sync(TypoPort().generate(REQUEST))
