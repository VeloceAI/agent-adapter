"""The real WP05 prompt-cache plug-ins applied to the real WP06 ports' wire requests."""

import json

import pytest

from agent_adapter.cache.providers import AnthropicPromptCache, OpenAIPromptCache
from agent_adapter.core.messages import LLMRequest, Message, ToolDef
from agent_adapter.core.types import ModelCaps
from tests.models.conftest import fixture

httpx2 = pytest.importorskip("httpx2")

TOOL = ToolDef(
    name="get_weather",
    description="Weather.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
EPHEMERAL = {"type": "ephemeral", "ttl": "5m"}


class Wire:
    def __init__(self, reply):
        self.reply = reply
        self.bodies: list[dict] = []

    def __call__(self, request):
        self.bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=self.reply)


def _http(wire):
    return httpx2.AsyncClient(transport=httpx2.MockTransport(wire))


async def test_anthropic_prompt_cache_marks_the_port_request():
    anthropic = pytest.importorskip("anthropic")
    from agent_adapter.models.anthropic import AnthropicModel

    wire = Wire(fixture("anthropic/tool_use"))
    client = anthropic.AsyncAnthropic(api_key="test", http_client=_http(wire), max_retries=0)
    plugin = AnthropicPromptCache(
        on=("system", "tools", "breakpoints"), caps=lambda model: ModelCaps(cache_control=True)
    )
    model = AnthropicModel("claude-test", client=client, cache_plugin=plugin, max_tokens=64)
    request = LLMRequest(
        messages=(
            Message(role="system", content="Be brief."),
            Message(role="user", content="Long document.", cache_breakpoint=True),
            Message(role="assistant", content="Read it."),
            Message(role="user", content="Summarise."),
        ),
        tools=(TOOL,),
    )
    await model.generate(request)
    [body] = wire.bodies
    assert body["system"][-1]["cache_control"] == EPHEMERAL
    assert body["tools"][-1]["cache_control"] == EPHEMERAL
    first, *rest = body["messages"]
    assert first["content"][-1] == {
        "type": "text",
        "text": "Long document.",
        "cache_control": EPHEMERAL,
    }
    assert "cache_control" not in json.dumps(rest)


async def test_openai_prompt_cache_sets_the_key_on_the_port_request():
    openai = pytest.importorskip("openai")
    from agent_adapter.models.openai import OpenAIModel

    wire = Wire(fixture("openai/responses_text"))
    client = openai.AsyncOpenAI(api_key="test", http_client=_http(wire), max_retries=0)
    plugin = OpenAIPromptCache(
        key="tax-assistant", caps=lambda model: ModelCaps(prompt_cache_key=True)
    )
    model = OpenAIModel("gpt-test", client=client, cache_plugin=plugin)
    await model.generate(LLMRequest(messages=(Message(role="user", content="hi"),)))
    [body] = wire.bodies
    assert body["prompt_cache_key"] == "tax-assistant"
