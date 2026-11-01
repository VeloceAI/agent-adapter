"""OpenAIModel against an OpenAI-compatible local server (Ollama-shaped, MockTransport)."""

import pytest

from agent_adapter.core.messages import Message
from agent_adapter.core.registry import Registry
from agent_adapter.cost import Ledger
from agent_adapter.exec import Executor, run_tool_loop
from tests.models.wire import RecordingWire

openai = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")

from agent_adapter.models.openai import OpenAIModel  # noqa: E402

BASE_URL = "http://localhost:11434/v1"
ASK = (Message(role="user", content="Weather in Pune?"),)


class Wire(RecordingWire):
    response_type = httpx2.Response


def _reply(message: dict[str, object], finish: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 1,
        "model": "llama3.1:8b",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 6, "total_tokens": 46},
    }


TOOL_CALL = _reply(
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city": "Pune"}'},
            }
        ],
    },
    "tool_calls",
)
ANSWER = _reply({"role": "assistant", "content": "Pune: 31C"}, "stop")


def _registry() -> Registry:
    registry = Registry()

    @registry.tool
    def weather(city: str, units: str = "C") -> str:
        """Current weather for a city."""
        return f"{city}: 31{units}"

    return registry


def _local(wire: Wire, *, strict_tools: bool) -> OpenAIModel:
    return OpenAIModel(
        "llama3.1:8b",
        api="chat",
        strict_tools=strict_tools,
        client_kwargs={
            "base_url": BASE_URL,
            "api_key": "ollama",
            "max_retries": 0,
            "http_client": httpx2.AsyncClient(transport=httpx2.MockTransport(wire)),
        },
    )


async def test_tool_loop_runs_on_a_local_openai_compatible_server():
    wire = Wire(TOOL_CALL, ANSWER)
    registry, ledger = _registry(), Ledger()
    executor = Executor(registry, ledger=ledger)
    model = _local(wire, strict_tools=False)
    result = await run_tool_loop(model, registry, ASK, executor=executor)
    assert (result.finish, result.steps, result.final.text) == ("stop", 2, "Pune: 31C")
    assert wire.urls == [f"{BASE_URL}/chat/completions"] * 2
    [tool] = wire.bodies[0]["tools"]
    assert tool["function"]["strict"] is False
    assert tool["function"]["parameters"]["required"] == ["city"]  # optional stays optional
    assert wire.bodies[1]["messages"][-1]["role"] == "tool"
    llm_entries = [e for e in ledger.entries() if e.kind == "llm"]
    assert [e.input_tokens for e in llm_entries] == [40, 40]


async def test_strict_tools_default_sends_strict_schemas():
    wire = Wire(ANSWER)
    registry = _registry()
    await run_tool_loop(_local(wire, strict_tools=True), registry, ASK)
    [tool] = wire.bodies[0]["tools"]
    assert tool["function"]["strict"] is True
    assert sorted(tool["function"]["parameters"]["required"]) == ["city", "units"]


async def test_server_error_surfaces_as_a_model_error():
    from agent_adapter.core.errors import ModelError

    wire = Wire(httpx2.Response(500, json={"error": {"message": "model not loaded"}}))
    with pytest.raises(ModelError):
        await run_tool_loop(_local(wire, strict_tools=False), _registry(), ASK)
