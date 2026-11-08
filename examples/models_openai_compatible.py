"""Azure OpenAI, Groq and OpenRouter: each vendor's own SDK wrapped as a CallableModel."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

from models_common import demo, tool_turn, typed_error

from agent_adapter import CallableModel, LLMRequest, Prices
from agent_adapter.models.openai import build_chat_request, parse_chat_completion


# --8<-- [start:chat]
def chat_model(
    create: Callable[..., Any], *, provider: str, model: str, plain_tools: bool = False
) -> CallableModel:
    """A model port over any OpenAI-shaped `chat.completions.create` of a vendor SDK."""

    def call(request: LLMRequest) -> dict[str, Any]:
        kwargs = build_chat_request(request, model=request.model or model, strict_tools=False)
        if plain_tools:  # send tool schemas without the OpenAI-only "strict" flag
            for tool in kwargs.get("tools", []):
                tool["function"].pop("strict", None)
        try:
            raw = create(**kwargs)  # a sync SDK call: CallableModel runs it in a worker thread
        except Exception as exc:
            error = typed_error(exc, provider=provider, model=model)
            if error is None:
                raise
            raise error from exc
        reply = parse_chat_completion(raw)  # SDK object or dict -> text, tool calls, usage
        return {
            "id": reply.id,
            "text": reply.text,
            "tool_calls": reply.tool_calls,
            "finish_reason": reply.finish_reason,
            "raw_finish_reason": reply.raw_finish_reason,
            "usage": reply.usage,  # prompt tokens split into uncached, cached and written
        }

    return CallableModel(call, provider=provider, model=model)


# --8<-- [end:chat]


# --8<-- [start:azure]
def azure_model(endpoint: str, api_key: str, deployment: str) -> CallableModel:
    """Azure OpenAI: `model` is your deployment name; the SDK is imported on first use."""
    from openai import AzureOpenAI

    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2024-10-21")
    return chat_model(client.chat.completions.create, provider="azure", model=deployment)


# --8<-- [end:azure]


# --8<-- [start:groq]
def groq_model(api_key: str, model: str = "llama-3.3-70b-versatile") -> CallableModel:
    """Groq's own SDK (`pip install groq`)."""
    from groq import Groq

    client = Groq(api_key=api_key)
    return chat_model(
        client.chat.completions.create, provider="groq", model=model, plain_tools=True
    )


# --8<-- [end:groq]


# --8<-- [start:openrouter]
OPENROUTER = "https://openrouter.ai/api/v1"


def openrouter_model(api_key: str, model: str = "openai/gpt-5.4") -> CallableModel:
    """OpenRouter through the openai SDK; `model` is OpenRouter's "vendor/model" id."""
    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER, api_key=api_key)
    return chat_model(client.chat.completions.create, provider="openrouter", model=model)


# --8<-- [end:openrouter]


# --8<-- [start:prices]
PRICES = Prices.load(  # example rates: copy yours from each vendor's price page
    {
        "version": "mine-2026-11",
        "models": {
            "shop-gpt": {"provider": "azure", "input": 2.50, "cached_input": 0.25, "output": 15.0},
            "llama-3.3-70b-versatile": {"provider": "groq", "input": 0.60, "output": 0.80},
            "openai/*": {"provider": "openrouter", "input": 2.50, "output": 15.00},
        },
    }
)
# --8<-- [end:prices]


def fake_create(**kwargs: Any) -> dict[str, Any]:
    """Stands in for `client.chat.completions.create`: asks for one tool, then answers."""
    message: dict[str, Any]
    if tool_turn(kwargs["messages"]):
        message = {
            "role": "assistant",
            "content": f"Ticket T-1: {kwargs['messages'][-1]['content']}.",
        }
        finish = "stop"
    else:
        arguments = json.dumps({"ticket": "T-1"})
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "ticket_status", "arguments": arguments},
        }
        message, finish = {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
    usage = {
        "prompt_tokens": 1_000,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 400},
    }
    return {
        "id": "chatcmpl-1",
        "model": kwargs["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    }


async def main() -> None:
    await demo(chat_model(fake_create, provider="azure", model="shop-gpt"), PRICES)
    groq = chat_model(
        fake_create, provider="groq", model="llama-3.3-70b-versatile", plain_tools=True
    )
    await demo(groq, PRICES)
    await demo(chat_model(fake_create, provider="openrouter", model="openai/gpt-5.4"), PRICES)


if __name__ == "__main__":
    asyncio.run(main())
