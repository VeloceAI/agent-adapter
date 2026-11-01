import pytest

from agent_adapter.core.errors import ModelUnavailable
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.types import ToolCall, Usage
from tests.fakes.models import FakePromptCachePlugin, ScriptedModel, calls, text

REQUEST = LLMRequest(messages=(Message(role="user", content="hi"),))


def test_text_and_calls_helpers():
    reply = text("hi", usage=Usage(input_tokens=2), parsed={"a": 1})
    assert (reply.text, reply.message.content, reply.parsed) == ("hi", "hi", {"a": 1})
    assert reply.usage.input_tokens == 2
    asked = calls(("search", {"q": "x"}), ToolCall(name="lookup", id="c9"), content="Let me look.")
    assert [(c.name, c.id) for c in asked.tool_calls] == [("search", "call_0"), ("lookup", "c9")]
    assert asked.finish_reason == "tool_calls" and asked.message.tool_calls == asked.tool_calls


async def test_scripted_model_replays_in_order_and_records():
    model = ScriptedModel([text("one"), "two", ModelUnavailable("down", provider="scripted")])
    assert (await model.generate(REQUEST)).text == "one"
    assert (await model.generate(REQUEST)).text == "two"
    with pytest.raises(ModelUnavailable):
        await model.generate(REQUEST)
    with pytest.raises(AssertionError, match="no response left"):
        await model.generate(REQUEST)
    assert len(model.requests) == 4


async def test_fake_plugin_merges_and_records():
    seen = []
    plugin = FakePromptCachePlugin({"k": 1}, providers=("openai",), edit=lambda kw: seen.append(kw))
    out = await plugin.apply(REQUEST, {"model": "m"}, model="m", client="c")
    assert out == {"model": "m", "k": 1} and seen == [out]
    assert plugin.calls[0]["client"] == "c" and plugin.key_fields() == {
        "name": "fake",
        "sets": ["k"],
    }
    assert plugin.providers == frozenset({"openai"})
