"""Real prompt-cache plug-ins with default caps (`Prices.default()`) on real ports' wire bodies.

Each port runs through `executor.model(...)`, so the ledger entry (incl. `cache_write_by_ttl`)
is checked too. Transport path only: httpx2 / httpx MockTransport, no network.
"""

import json

import pytest

from agent_adapter.cache import digest
from agent_adapter.cache.providers import (
    AnthropicPromptCache,
    GeminiExplicitCache,
    OpenAIPromptCache,
    OpenAIPromptCacheV2,
)
from agent_adapter.core.messages import LLMRequest, Message, ToolDef
from agent_adapter.cost import Ledger
from agent_adapter.exec import Executor
from tests.models.conftest import fixture

SYSTEM = "You answer questions about the 2026 tax code. " * 200  # ~9600 chars, ~2400 tokens
TOOL = ToolDef(
    name="get_rate",
    description="Tax rate.",
    parameters={
        "type": "object",
        "properties": {"year": {"type": "integer"}},
        "required": ["year"],
    },
)
REQUEST = LLMRequest(
    messages=(
        Message(role="system", content=SYSTEM, cache_breakpoint=True),
        Message(role="user", content="What is the rate?"),
    ),
    tools=(TOOL,),
    stage="answer",
)


class Wire:
    """MockTransport handler: records (path, body) and answers with the next reply."""

    def __init__(self, response_type, *replies):
        self.response_type = response_type
        self.replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, request):
        self.calls.append((request.url.path, json.loads(request.content)))
        return self.response_type(200, json=self.replies.pop(0))

    @property
    def bodies(self):
        return [body for _, body in self.calls]


def instrumented(port):
    ledger = Ledger()  # the packaged price table
    return Executor(ledger=ledger).model(port), ledger


def openai_client(wire):
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(wire))
    return openai.AsyncOpenAI(api_key="test", http_client=http, max_retries=0)


async def test_openai_prompt_cache_key_and_24h_retention_on_gpt_4_1():
    httpx2 = pytest.importorskip("httpx2")
    from agent_adapter.models.openai import OpenAIModel

    reply = {**fixture("openai/responses_text"), "model": "gpt-4.1"}
    wire = Wire(httpx2.Response, reply)
    port = OpenAIModel(
        "gpt-4.1", client=openai_client(wire), cache_plugin=OpenAIPromptCache(retention="24h")
    )
    model, ledger = instrumented(port)
    await model.generate(REQUEST)
    [body] = wire.bodies
    assert body["prompt_cache_key"] == digest(SYSTEM)
    assert body["prompt_cache_retention"] == "24h"
    [entry] = ledger.entries()
    assert (entry.model, entry.cached_input_tokens, entry.cache_write_input_tokens) == (
        "gpt-4.1",
        4000,
        7000,
    )
    assert entry.cost_usd is not None and entry.price_version


async def test_openai_unknown_models_get_no_cache_parameters():
    httpx2 = pytest.importorskip("httpx2")
    from agent_adapter.models.openai import OpenAIModel

    wire = Wire(httpx2.Response, fixture("openai/responses_text"))
    port = OpenAIModel(
        "my-finetune", client=openai_client(wire), cache_plugin=OpenAIPromptCache(retention="24h")
    )
    await port.generate(REQUEST)
    [body] = wire.bodies
    assert "prompt_cache_key" not in body and "prompt_cache_retention" not in body


async def test_openai_v2_prompt_cache_options_on_gpt_5_6_responses():
    httpx2 = pytest.importorskip("httpx2")
    from agent_adapter.models.openai import OpenAIModel

    wire = Wire(httpx2.Response, fixture("openai/responses_text"))
    plugin = OpenAIPromptCacheV2(ttl="12h", key="tax")
    port = OpenAIModel("gpt-5.6", client=openai_client(wire), cache_plugin=plugin)
    model, ledger = instrumented(port)
    await model.generate(REQUEST)
    [body] = wire.bodies
    assert body["prompt_cache_options"] == {"mode": "implicit", "ttl": "12h"}
    assert body["prompt_cache_key"] == "tax"
    [entry] = ledger.entries()
    assert entry.cache_write_by_ttl == {"30m": 3000, "12h": 4000}
    assert entry.cost_usd is not None


async def test_openai_v2_explicit_breakpoints_on_chat_and_ttl_filled_writes():
    httpx2 = pytest.importorskip("httpx2")
    from agent_adapter.models.openai import OpenAIModel

    reply = {
        **fixture("openai/chat_text"),
        "model": "gpt-5.6",
        "usage": {
            "prompt_tokens": 3000,
            "completion_tokens": 10,
            "total_tokens": 3010,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 2500},
        },
    }
    wire = Wire(httpx2.Response, reply)
    plugin = OpenAIPromptCacheV2(mode="explicit", ttl="12h")
    port = OpenAIModel("gpt-5.6", api="chat", client=openai_client(wire), cache_plugin=plugin)
    model, ledger = instrumented(port)
    await model.generate(REQUEST)
    [body] = wire.bodies
    assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "12h"}
    system = body["messages"][0]
    assert system["content"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    [entry] = ledger.entries()
    assert (entry.input_tokens, entry.cache_write_input_tokens) == (500, 2500)
    assert entry.cache_write_by_ttl == {"12h": 2500}


async def test_gemini_explicit_cache_creates_once_then_sends_cached_content():
    pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    from google.genai import Client, types

    from agent_adapter.models.gemini import GeminiModel

    created = {
        "name": "cachedContents/tax-2026",
        "model": "models/gemini-2.5-pro",
        "usageMetadata": {"totalTokenCount": 2400},
    }
    answer = {**fixture("gemini/text"), "model_version": "gemini-2.5-pro"}
    wire = Wire(httpx.Response, created, answer, answer)
    options = types.HttpOptions(
        httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(wire)),
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    client = Client(api_key="test", http_options=options)
    port = GeminiModel("gemini-2.5-pro", client=client, cache_plugin=GeminiExplicitCache(ttl_s=600))
    model, ledger = instrumented(port)
    await model.generate(REQUEST)
    await model.generate(REQUEST)
    paths = [path for path, _ in wire.calls]
    assert paths[0].endswith("/cachedContents")
    assert all(path.endswith(":generateContent") for path in paths[1:])
    assert len(paths) == 3  # one creation for both calls
    create, *generates = wire.bodies
    assert create["systemInstruction"]["parts"][0]["text"] == SYSTEM
    assert create["ttl"] == "600s" and create["tools"]
    for body in generates:
        assert body["cachedContent"] == "cachedContents/tax-2026"
        assert "systemInstruction" not in body and "tools" not in body
    storage, first, second = ledger.entries()
    assert (storage.kind, storage.model) == ("api", "gemini-2.5-pro:cache_storage")
    assert storage.cost_usd == pytest.approx(2400 * (600 / 3600) / 1e6 * 4.50)
    assert storage.run_id == first.run_id  # recorded through the instrumented call's context
    assert (first.kind, second.kind) == ("llm", "llm")


async def test_anthropic_cache_control_on_system_and_tools_for_the_claude_family():
    httpx2 = pytest.importorskip("httpx2")
    anthropic = pytest.importorskip("anthropic")
    from agent_adapter.models.anthropic import AnthropicModel

    wire = Wire(httpx2.Response, fixture("anthropic/tool_use"))
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(wire))
    client = anthropic.AsyncAnthropic(api_key="test", http_client=http, max_retries=0)
    port = AnthropicModel(
        "claude-opus-4-7", client=client, cache_plugin=AnthropicPromptCache(ttl="1h")
    )
    model, ledger = instrumented(port)
    await model.generate(REQUEST)
    [body] = wire.bodies
    marker = {"type": "ephemeral", "ttl": "1h"}
    assert body["system"][-1]["cache_control"] == marker
    assert body["tools"][-1]["cache_control"] == marker
    assert "cache_control" not in json.dumps(body["messages"])
    [entry] = ledger.entries()
    assert entry.cache_write_by_ttl == {"5m": 1000, "1h": 500}
    assert (entry.cached_input_tokens, entry.cache_write_input_tokens) == (2000, 1500)
    assert entry.cost_usd is None  # caps-only row: no cited Anthropic prices
