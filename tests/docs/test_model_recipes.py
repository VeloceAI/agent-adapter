"""Model recipes: Bedrock, Azure OpenAI, Groq, OpenRouter and Mistral over fake SDK modules."""

import asyncio
import importlib
import runpy
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from agent_adapter import LLMRequest, Message, ToolCall, ToolDef
from agent_adapter.errors import (
    ModelBadRequest,
    ModelContextOverflow,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
PAGE = ROOT / "docs" / "how-to" / "model-recipes.md"
ASK = (Message(role="user", content="Where is ticket T-1?"),)
TOOL = ToolDef(name="ticket_status", description="Status.", parameters={"type": "object"})


@pytest.fixture
def load(monkeypatch: pytest.MonkeyPatch):
    """Import an example as a module, with its sibling `models_common` importable."""
    monkeypatch.syspath_prepend(str(EXAMPLES))

    def _load(name: str) -> Any:
        sys.modules.pop(name, None)
        return importlib.import_module(name)

    yield _load
    for name in ("models_common", "models_bedrock", "models_mistral", "models_openai_compatible"):
        sys.modules.pop(name, None)


def fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> None:
    """Put a fake SDK module in `sys.modules` (the real one need not be installed)."""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(
    ("example", "expected"),
    [
        (
            "models_bedrock.py",
            [
                "bedrock stop 2 Ticket T-1: waiting for parts.",
                "amazon.nova-pro-v1:0 ['0.000800', '0.000800']",
            ],
        ),
        (
            "models_mistral.py",
            [
                "mistral stop 2 Ticket T-1: waiting for parts.",
                "mistral-large-latest ['0.001780', '0.001780']",
            ],
        ),
        (
            "models_openai_compatible.py",
            [
                "azure stop 2 Ticket T-1: waiting for parts.",
                "shop-gpt ['0.001900', '0.001900']",
                "groq stop 2 Ticket T-1: waiting for parts.",
                "llama-3.3-70b-versatile ['0.000616', '0.000616']",
                "openrouter stop 2 Ticket T-1: waiting for parts.",
                "openai/gpt-5.4 ['0.002800', '0.002800']",
            ],
        ),
    ],
)
def test_each_recipe_runs_offline_and_prices_its_usage(load, capsys, example, expected):
    load("models_common")
    runpy.run_path(str(EXAMPLES / example), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == expected


def test_bedrock_factory_uses_boto3_bedrock_runtime(load, monkeypatch):
    bedrock = load("models_bedrock")
    made: list[tuple[str, str]] = []

    def client(service: str, *, region_name: str) -> Any:
        made.append((service, region_name))
        return bedrock.FakeBedrock()

    fake_module(monkeypatch, "boto3", client=client)
    port = bedrock.bedrock_model("amazon.nova-pro-v1:0", region="eu-west-1")
    reply = asyncio.run(port.generate(LLMRequest(messages=ASK, tools=(TOOL,))))
    assert made == [("bedrock-runtime", "eu-west-1")]
    assert reply.finish_reason == "tool_calls" and reply.tool_calls[0].name == "ticket_status"
    assert reply.usage.input_tokens == 900 and reply.provider == "bedrock"


def test_to_converse_maps_system_tools_results_and_config(load):
    bedrock = load("models_bedrock")
    calls = (
        ToolCall(id="u1", name="ticket_status", arguments={"ticket": "T-1"}),
        ToolCall(id="u2", name="ticket_status", arguments={"ticket": "T-2"}),
    )
    messages = (
        Message(role="system", content="Be brief."),
        *ASK,
        Message(role="assistant", content="Checking.", tool_calls=calls),
        Message(role="tool", content="open", tool_call_id="u1"),
        Message(role="tool", content="closed", tool_call_id="u2"),
    )
    request = LLMRequest(messages=messages, tools=(TOOL,), max_output_tokens=64, stop=("END",))
    kwargs = bedrock.to_converse(request, "m")
    assert kwargs["system"] == [{"text": "Be brief."}]
    assert kwargs["inferenceConfig"] == {"maxTokens": 64, "stopSequences": ["END"]}
    turns = kwargs["messages"]
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    assert turns[1]["content"][1] == {
        "toolUse": {"toolUseId": "u1", "name": "ticket_status", "input": {"ticket": "T-1"}}
    }
    assert [b["toolResult"]["toolUseId"] for b in turns[2]["content"]] == ["u1", "u2"]
    assert kwargs["toolConfig"]["toolChoice"] == {"auto": {}}
    spec = kwargs["toolConfig"]["tools"][0]["toolSpec"]
    assert spec == {
        "name": "ticket_status",
        "description": "Status.",
        "inputSchema": {"json": {"type": "object"}},
    }


@pytest.mark.parametrize(
    ("choice", "expected"),
    [("required", {"any": {}}), ("ticket_status", {"tool": {"name": "ticket_status"}})],
)
def test_to_converse_tool_choice(load, choice, expected):
    bedrock = load("models_bedrock")
    kwargs = bedrock.to_converse(LLMRequest(messages=ASK, tools=(TOOL,), tool_choice=choice), "m")
    assert kwargs["toolConfig"]["toolChoice"] == expected


def test_to_converse_sends_no_tools_for_none_and_no_config_by_default(load):
    bedrock = load("models_bedrock")
    kwargs = bedrock.to_converse(LLMRequest(messages=ASK, tools=(TOOL,), tool_choice="none"), "m")
    assert set(kwargs) == {"modelId", "messages"}


@pytest.mark.parametrize(
    ("reason", "finish"),
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("guardrail_intervened", "content_filter"),
        ("something_new", "other"),
    ],
)
def test_from_converse_reads_text_stop_reason_and_cache_usage(load, reason, finish):
    bedrock = load("models_bedrock")
    raw = {
        "output": {"message": {"content": [{"text": "a"}, {"text": "b"}]}},
        "stopReason": reason,
        "usage": {
            "inputTokens": 10,
            "outputTokens": 5,
            "cacheReadInputTokens": 70,
            "cacheWriteInputTokens": 20,
        },
    }
    reply = bedrock.from_converse(raw)
    assert reply["text"] == "ab" and reply["finish_reason"] == finish
    assert reply["raw_finish_reason"] == reason
    usage = reply["usage"]
    assert (usage.input_tokens, usage.cached_input_tokens, usage.cache_write_input_tokens) == (
        10,
        70,
        20,
    )
    assert usage.prompt_tokens == 100
    assert bedrock.from_converse({})["text"] == ""  # an empty response is not a crash


def test_azure_and_openrouter_factories_build_the_openai_sdk_clients(load, monkeypatch):
    compat = load("models_openai_compatible")
    built: list[tuple[str, dict[str, Any]]] = []

    def client_class(kind: str) -> type:
        class Client:
            def __init__(self, **kwargs: Any) -> None:
                built.append((kind, kwargs))
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=compat.fake_create)
                )

        return Client

    fake_module(
        monkeypatch, "openai", AzureOpenAI=client_class("azure"), OpenAI=client_class("openai")
    )
    azure = compat.azure_model("https://shop.openai.azure.com", "key-1", "shop-gpt")
    router = compat.openrouter_model("key-2")
    assert built[0] == (
        "azure",
        {
            "azure_endpoint": "https://shop.openai.azure.com",
            "api_key": "key-1",
            "api_version": "2024-10-21",
        },
    )
    assert built[1] == ("openai", {"base_url": "https://openrouter.ai/api/v1", "api_key": "key-2"})
    reply = asyncio.run(azure.generate(LLMRequest(messages=ASK)))
    assert reply.provider == "azure" and reply.model == "shop-gpt"
    assert reply.usage.cached_input_tokens == 400 and reply.usage.input_tokens == 600
    assert asyncio.run(router.generate(LLMRequest(messages=ASK))).model == "openai/gpt-5.4"


def test_groq_factory_sends_plain_tool_schemas(load, monkeypatch):
    compat = load("models_openai_compatible")
    sent: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> dict[str, Any]:
        sent.append(kwargs)
        return compat.fake_create(**kwargs)

    class Groq:
        def __init__(self, *, api_key: str) -> None:
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    fake_module(monkeypatch, "groq", Groq=Groq)
    port = compat.groq_model("key-3")
    asyncio.run(port.generate(LLMRequest(messages=ASK, tools=(TOOL,))))
    assert sent[0]["model"] == "llama-3.3-70b-versatile"
    assert "strict" not in sent[0]["tools"][0]["function"]


def test_mistral_factory_and_conversions(load, monkeypatch):
    mistral = load("models_mistral")
    keys: list[str] = []

    class Mistral(mistral.FakeMistral):
        def __init__(self, *, api_key: str) -> None:
            keys.append(api_key)

    fake_module(monkeypatch, "mistralai", Mistral=Mistral)
    port = mistral.mistral_model("key-4")
    reply = asyncio.run(port.generate(LLMRequest(messages=ASK, tools=(TOOL,))))
    assert keys == ["key-4"]
    assert reply.tool_calls[0].arguments == {"ticket": "T-1"}  # an object, sent back as JSON
    request = LLMRequest(
        messages=ASK,
        tools=(TOOL,),
        tool_choice="required",
        max_output_tokens=9,
        seed=7,
        reasoning="low",
    )
    kwargs = mistral.to_mistral(request, "m")
    assert kwargs["tool_choice"] == "any" and kwargs["max_tokens"] == 9
    assert kwargs["random_seed"] == 7
    assert not {"max_completion_tokens", "seed", "reasoning_effort"} & set(kwargs)


def test_from_mistral_accepts_an_sdk_object(load):
    mistral = load("models_mistral")

    class Response:
        def model_dump(self) -> dict[str, Any]:
            message = {
                "role": "assistant",
                "content": [{"type": "thinking"}, {"type": "text", "text": "hi"}],
            }
            return {
                "id": "c",
                "model": "m",
                "choices": [{"message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }

    reply = mistral.from_mistral(Response())
    assert reply["text"] == "hi" and reply["usage"].input_tokens == 3


class StatusError(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status_code = status


class ClientError(Exception):
    """botocore's shape: the status sits in a response dict."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.response = {
            "Error": {"Code": "ValidationException"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (StatusError("slow down", 429), ModelRateLimited),
        (StatusError("bad field", 400), ModelBadRequest),
        (StatusError("too many tokens in the prompt", 400), ModelContextOverflow),
        (ClientError("Throttled", 503), ModelUnavailable),
        (APITimeoutError("timed out"), ModelTimeout),
        (APIConnectionError("reset"), ModelUnavailable),
    ],
)
def test_typed_error_maps_sdk_errors(load, exc, expected):
    common = load("models_common")
    error = common.typed_error(exc, provider="groq", model="m")
    assert type(error) is expected and error.provider == "groq" and error.model == "m"


def test_a_non_provider_error_propagates_unchanged(load):
    compat = load("models_openai_compatible")

    def broken(**kwargs: Any) -> Any:
        raise ValueError("a bug in the host")

    port = compat.chat_model(broken, provider="groq", model="m")
    with pytest.raises(ValueError, match="a bug in the host"):
        asyncio.run(port.generate(LLMRequest(messages=ASK)))

    def limited(**kwargs: Any) -> Any:
        raise StatusError("slow down", 429)

    with pytest.raises(ModelRateLimited) as caught:
        asyncio.run(
            compat.chat_model(limited, provider="groq", model="m").generate(
                LLMRequest(messages=ASK)
            )
        )
    assert isinstance(caught.value.__cause__, StatusError)


@pytest.mark.parametrize("name", ["models_bedrock", "models_mistral"])
def test_bedrock_and_mistral_wrap_provider_errors_too(load, name):
    module = load(name)

    class Client:
        def converse(self, **kwargs: Any) -> Any:
            raise ClientError("Too many requests", 429)

        chat = types.SimpleNamespace(
            complete=lambda **kwargs: (_ for _ in ()).throw(StatusError("x", 429))
        )

    factory = module.converse_model if name == "models_bedrock" else module.complete_model
    with pytest.raises(ModelRateLimited):
        asyncio.run(factory(Client(), "m").generate(LLMRequest(messages=ASK)))
    plain = types.SimpleNamespace(
        converse=lambda **kwargs: (_ for _ in ()).throw(KeyError("k")),
        chat=types.SimpleNamespace(complete=lambda **kwargs: (_ for _ in ()).throw(KeyError("k"))),
    )
    with pytest.raises(KeyError):
        asyncio.run(factory(plain, "m").generate(LLMRequest(messages=ASK)))


def test_the_page_uses_every_section_and_stays_short():
    text = PAGE.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 150
    for line in text.splitlines():
        if line.startswith('--8<-- "examples/'):
            path, _, section = line.split('"')[1].partition(":")
            source = (ROOT / path).read_text(encoding="utf-8")
            assert f"[start:{section}]" in source and f"[end:{section}]" in source, line
    for example in (
        "models_bedrock",
        "models_mistral",
        "models_openai_compatible",
        "models_common",
    ):
        assert f"examples/{example}.py" in text
