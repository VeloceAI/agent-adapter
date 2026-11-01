"""Model ports: wrap a host LLM function, fall back on transient errors, normalise usage."""

import asyncio
import importlib.util

from agent_adapter import CallableModel, FallbackModel, LLMRequest, Message
from agent_adapter.errors import ModelRateLimited
from agent_adapter.models import normalise_usage


# --8<-- [start:callable]
async def call_llm_model(request: LLMRequest) -> str:
    """Stands in for the host's own LLM router (keys, routing and stats stay there)."""
    question = request.messages[-1].content
    return f"answer to {question!r}"


host = CallableModel(call_llm_model, provider="host", model="router-v2")
# --8<-- [end:callable]


# --8<-- [start:fallback]
async def overloaded(request: LLMRequest) -> str:
    raise ModelRateLimited("slow down", provider="primary", model="big", retry_after=2.0)


primary = CallableModel(overloaded, provider="primary", model="big")
model = FallbackModel([primary, host])  # falls over on rate limits, 5xx and timeouts only
# --8<-- [end:fallback]


# --8<-- [start:providers]
def provider_ports() -> list[object]:
    """Ports for the three SDKs; clients are created lazily, one per event loop."""
    from agent_adapter.models import AnthropicModel, GeminiModel, OpenAIModel

    return [
        OpenAIModel("gpt-5", client_kwargs={"api_key": "sk-...", "max_retries": 0}),
        OpenAIModel("gpt-4.1", api="chat", strict_tools=True),
        GeminiModel("gemini-2.5-flash", thinking={"include_thoughts": False}),
        AnthropicModel("claude-opus-4-1", max_tokens=2048),
    ]


# --8<-- [end:providers]


async def main() -> None:
    request = LLMRequest(messages=(Message(role="user", content="Return window?"),))
    reply = await model.generate(request)
    print(reply.provider, reply.text)
    # --8<-- [start:usage]
    usage = normalise_usage(
        "openai",
        {
            "input_tokens": 12000,
            "input_tokens_details": {"cached_tokens": 4000, "cache_write_tokens": 7000},
            "output_tokens": 40,
        },
        api="responses",
    )
    print(usage.input_tokens, usage.cached_input_tokens, usage.cache_write_input_tokens)
    # --8<-- [end:usage]
    sdks = ("openai", "google.genai", "anthropic")
    if all(importlib.util.find_spec(name) is not None for name in sdks):
        print([type(port).__name__ for port in provider_ports()])


if __name__ == "__main__":
    asyncio.run(main())
