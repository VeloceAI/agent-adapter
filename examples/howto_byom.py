"""Bring your own model: wrap any local or remote model function; the adapter adds the rest."""

import asyncio
import importlib.util
import threading
from typing import Any

from agent_adapter import (
    CallableModel,
    Executor,
    Ledger,
    LLMRequest,
    Message,
    Prices,
    Registry,
    run_tool_loop,
)
from agent_adapter.models import OpenAIModel
from agent_adapter.models.openai import build_chat_request, parse_chat_completion

registry = Registry()


@registry.tool
def exchange_rate(currency: str) -> float:
    """Today's EUR rate for a currency."""
    return {"USD": 0.92, "GBP": 1.17}.get(currency, 1.0)


class FakeLlama:
    """Stands in for `llama_cpp.Llama`: blocking, not thread-safe, OpenAI-shaped replies."""

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        last = kwargs["messages"][-1]
        message: dict[str, Any]
        if last["role"] == "tool":
            message = {"role": "assistant", "content": f"The rate is {last['content']}."}
            finish = "stop"
        else:
            call = {"name": "exchange_rate", "arguments": '{"currency": "USD"}'}
            tool_call = {"id": "call_1", "type": "function", "function": call}
            message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
            finish = "tool_calls"
        return {
            "model": "qwen2.5-7b-instruct-q4.gguf",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 180, "completion_tokens": 12},
        }


# --8<-- [start:local]
# real: Llama(model_path="qwen2.5-7b.gguf", chat_format="chatml-function-calling")
llm = FakeLlama()
llm_lock = threading.Lock()  # one llama.cpp context must not run two calls at once


def local_chat(request: LLMRequest) -> dict[str, Any]:
    """Sync and blocking: CallableModel runs it in a worker thread."""
    kwargs = build_chat_request(request, model="qwen2.5-7b", strict_tools=False)
    if "max_completion_tokens" in kwargs:  # llama.cpp still calls it max_tokens
        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
    with llm_lock:
        raw = llm.create_chat_completion(**kwargs)
    reply = parse_chat_completion(raw)  # OpenAI-shaped dict -> text, tool calls, usage
    return {
        "text": reply.text,
        "tool_calls": reply.tool_calls,
        "finish_reason": reply.finish_reason,
        "usage": reply.usage,  # the ledger prices this
    }


local = CallableModel(local_chat, provider="local", model="qwen2.5-7b")
# --8<-- [end:local]


def post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """Stands in for `requests.post(url, json=body, timeout=30).json()`."""
    return {"output": f"echo: {body['prompt']}", "tokens_in": 9, "tokens_out": 3}


# --8<-- [start:http]
GATEWAY = "https://llm.internal.example/v1/generate"  # any API, any JSON shape


def gateway(request: LLMRequest) -> dict[str, Any]:
    """Your HTTP API: map the request in and the reply out; usage uses `Usage` field names."""
    body = {"prompt": request.messages[-1].content, "max_tokens": request.max_output_tokens or 256}
    data = post_json(GATEWAY, body)
    usage = {"input_tokens": data["tokens_in"], "output_tokens": data["tokens_out"]}
    return {"text": data["output"], "usage": usage}


remote = CallableModel(gateway, provider="gateway", model="house-13b")
# --8<-- [end:http]


# --8<-- [start:servers]
SERVERS = {  # OpenAI-compatible servers, default ports
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lm-studio": "http://localhost:1234/v1",
    "llama.cpp": "http://localhost:8080/v1",
}


def local_server(name: str, model: str) -> OpenAIModel:
    """An OpenAIModel aimed at a local server; the client is created on first use."""
    return OpenAIModel(
        model,
        api="chat",  # local servers speak Chat Completions; the Responses API is optional
        strict_tools=False,  # send plain tool schemas, not OpenAI strict-mode ones
        client_kwargs={"base_url": SERVERS[name], "api_key": "local", "max_retries": 0},
    )


# --8<-- [end:servers]


async def main() -> None:
    # --8<-- [start:loop]
    prices = Prices.load(
        {
            "version": "mine",
            "models": {
                "qwen2.5-7b": {"provider": "local", "free": True},  # costs 0, never warns
                "house-13b": {"provider": "gateway", "input": 0.5, "output": 1.5},
            },
        }
    )
    ledger = Ledger(prices=prices)
    executor = Executor(registry, ledger=ledger)
    ask = (Message(role="user", content="What is the USD rate?"),)
    result = await run_tool_loop(local, registry, ask, executor=executor)
    print(result.finish, result.steps, result.final.text)
    print([(e.kind, e.model, e.input_tokens, e.cost_usd) for e in ledger.entries()])

    async with executor.run_scope() as ctx:  # any port gets hooks, cache, ledger, events
        reply = await executor.model(remote).bind(ctx).generate(LLMRequest(messages=ask))
    [entry] = ledger.entries(ctx.run_id)
    print(reply.text, entry.model, f"{entry.cost_usd:.7f}")
    # --8<-- [end:loop]
    if importlib.util.find_spec("openai") is not None:
        port = local_server("ollama", "llama3.1:8b")
        print(port.model, port.api, port.strict_tools)


if __name__ == "__main__":
    asyncio.run(main())
