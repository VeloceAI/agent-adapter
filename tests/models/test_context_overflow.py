"""`ModelContextOverflow`: the typed error each provider port maps a too-long request to."""

import pickle

import pytest

from agent_adapter import errors as public
from agent_adapter.core.errors import ModelBadRequest, ModelContextOverflow, ModelUnavailable
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.models.errors import class_for_error_type, is_context_overflow, with_overflow

REQUEST = LLMRequest(messages=(Message(role="user", content="hi"),))
OPENAI = "This model's maximum context length is 128000 tokens. However, you requested 130000."
ANTHROPIC = "prompt is too long: 210000 tokens > 200000 maximum"
ANTHROPIC_MAX = "input length and `max_tokens` exceed context limit: 190000 + 21333 > 200000"
GEMINI = "The input token count (2000000) exceeds the maximum number of tokens allowed (1048576)."


def test_the_error_is_a_bad_request_with_its_own_kind():
    error = ModelContextOverflow("too long", provider="openai", model="m", status_code=400)
    assert isinstance(error, ModelBadRequest) and error.kind == "model-context-overflow"
    assert error.retryable is False
    assert public.ModelContextOverflow is ModelContextOverflow
    assert "ModelContextOverflow" in public.__all__
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is ModelContextOverflow and copy.__dict__ == error.__dict__


@pytest.mark.parametrize(
    "text", [OPENAI, ANTHROPIC, ANTHROPIC_MAX, GEMINI, "context_length_exceeded"]
)
def test_provider_overflow_messages_are_recognised(text):
    assert is_context_overflow(text)


def test_other_bad_requests_are_not_overflows():
    assert not is_context_overflow("max_tokens is too large: 50000")
    assert with_overflow(ModelBadRequest, ValueError("bad schema")) is ModelBadRequest
    assert with_overflow(ModelUnavailable, ValueError(OPENAI)) is ModelUnavailable
    assert with_overflow(ModelBadRequest, ValueError(ANTHROPIC)) is ModelContextOverflow
    body = {"error": {"type": "invalid_request_error", "code": "context_length_exceeded"}}
    assert with_overflow(ModelBadRequest, ValueError("x"), body) is ModelContextOverflow
    assert with_overflow(ModelContextOverflow, ValueError("x")) is ModelContextOverflow
    assert class_for_error_type("invalid_request_error context_length_exceeded") is (
        ModelContextOverflow
    )


def test_gemini_maps_a_token_count_error():
    errors = pytest.importorskip("google.genai.errors")
    from agent_adapter.models.gemini import map_gemini_error

    body = {"error": {"code": 400, "message": GEMINI, "status": "INVALID_ARGUMENT"}}
    mapped = map_gemini_error(errors.ClientError(400, body), model="gemini-2.5-flash")
    assert type(mapped) is ModelContextOverflow and mapped.status_code == 400
    other = {"error": {"code": 400, "message": "bad schema", "status": "INVALID_ARGUMENT"}}
    assert type(map_gemini_error(errors.ClientError(400, other), model="m")) is ModelBadRequest


async def test_openai_maps_context_length_exceeded():
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    from agent_adapter.models.openai import OpenAIModel

    def handler(request):
        body = {
            "error": {
                "message": OPENAI,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        }
        return httpx2.Response(400, json=body)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = openai.AsyncOpenAI(api_key="t", http_client=http, max_retries=0)
    with pytest.raises(ModelContextOverflow) as info:
        await OpenAIModel("gpt-4.1", client=client).generate(REQUEST)
    assert info.value.status_code == 400 and info.value.provider == "openai"


async def test_anthropic_maps_prompt_is_too_long():
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    from agent_adapter.models.anthropic import AnthropicModel

    def handler(request):
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": ANTHROPIC}}
        return httpx2.Response(400, json=body)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = anthropic.AsyncAnthropic(api_key="t", http_client=http, max_retries=0)
    with pytest.raises(ModelContextOverflow) as info:
        await AnthropicModel("claude-x", client=client).generate(REQUEST)
    assert info.value.provider == "anthropic"
