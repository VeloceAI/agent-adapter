"""Mistral's own SDK (`mistralai`) wrapped as a CallableModel, with usage and prices."""

import asyncio
import json
from typing import Any

from models_common import demo, tool_turn, typed_error

from agent_adapter import CallableModel, LLMRequest, Prices
from agent_adapter.models.openai import build_chat_request, parse_chat_completion


# --8<-- [start:convert]
def to_mistral(request: LLMRequest, model: str) -> dict[str, Any]:
    """`chat.complete(**kwargs)`: Chat Completions kwargs with Mistral's own names."""
    kwargs = build_chat_request(request, model=model, strict_tools=False)
    if "max_completion_tokens" in kwargs:
        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
    if "seed" in kwargs:
        kwargs["random_seed"] = kwargs.pop("seed")
    kwargs.pop("reasoning_effort", None)  # not a Mistral chat parameter
    if kwargs.get("tool_choice") == "required":
        kwargs["tool_choice"] = "any"
    return kwargs


def _text(content: Any) -> str | None:
    """Mistral may send content as a list of chunks; keep the text ones."""
    if isinstance(content, list):
        return "".join(str(c.get("text", "")) for c in content if c.get("type") == "text")
    return None if content is None else str(content)


def from_mistral(raw: Any) -> dict[str, Any]:
    """A `CallableModel` reply mapping from a `ChatCompletionResponse` (or its dict)."""
    data = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
    for choice in data.get("choices") or []:
        message = choice.get("message") or {}
        message["content"] = _text(message.get("content"))
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments")
            if isinstance(arguments, dict):  # Mistral may send arguments as an object
                call["function"]["arguments"] = json.dumps(arguments)
    reply = parse_chat_completion(data)
    return {
        "id": reply.id,
        "text": reply.text,
        "tool_calls": reply.tool_calls,
        "finish_reason": reply.finish_reason,
        "raw_finish_reason": reply.raw_finish_reason,
        "usage": reply.usage,
    }


# --8<-- [end:convert]


# --8<-- [start:model]
def complete_model(client: Any, model: str) -> CallableModel:
    """A model port over a `Mistral` client's `chat.complete`."""

    def call(request: LLMRequest) -> dict[str, Any]:
        name = request.model or model
        try:
            raw = client.chat.complete(**to_mistral(request, name))  # sync: a worker thread
        except Exception as exc:
            error = typed_error(exc, provider="mistral", model=name)
            if error is None:
                raise
            raise error from exc
        return from_mistral(raw)

    return CallableModel(call, provider="mistral", model=model)


def mistral_model(api_key: str, model: str = "mistral-large-latest") -> CallableModel:
    """Mistral's API; the SDK is imported on first use."""
    from mistralai import Mistral

    return complete_model(Mistral(api_key=api_key), model)


# --8<-- [end:model]

# --8<-- [start:prices]
PRICES = Prices.load(  # example rates: copy yours from Mistral's price page
    {
        "version": "mine-2026-11",
        "models": {"mistral-large-latest": {"provider": "mistral", "input": 2.00, "output": 6.00}},
    }
)
# --8<-- [end:prices]


class FakeChat:
    """Stands in for `Mistral(...).chat`: asks for one tool, then answers."""

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        if tool_turn(kwargs["messages"]):
            text = [{"type": "text", "text": f"Ticket T-1: {kwargs['messages'][-1]['content']}."}]
            message: dict[str, Any] = {"role": "assistant", "content": text}
            finish = "stop"
        else:
            function = {"name": "ticket_status", "arguments": {"ticket": "T-1"}}
            call = {"id": "abc123xyz", "type": "function", "function": function}
            message, finish = (
                {"role": "assistant", "content": "", "tool_calls": [call]},
                "tool_calls",
            )
        return {
            "id": "cmpl-1",
            "model": kwargs["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 800, "completion_tokens": 30, "total_tokens": 830},
        }


class FakeMistral:
    """Stands in for `mistralai.Mistral`."""

    chat = FakeChat()


async def main() -> None:
    await demo(complete_model(FakeMistral(), "mistral-large-latest"), PRICES)


if __name__ == "__main__":
    asyncio.run(main())
