"""Migration: from hand-written provider tool calling to a registry and the executor."""

import asyncio
import json
from typing import Any

from agent_adapter import (
    CallableModel,
    Executor,
    LLMRequest,
    Message,
    Registry,
    ToolCall,
    run_tool_loop,
)
from agent_adapter.integrations.anthropic import anthropic_tools, tool_result_blocks
from agent_adapter.integrations.openai import chat_tools


# --8<-- [start:before]
def exchange_rate(currency: str) -> float:
    return {"USD": 0.92, "GBP": 1.17}[currency]


TOOLS = [  # written by hand, kept in step with the function by hand
    {
        "type": "function",
        "function": {
            "name": "exchange_rate",
            "description": "Today's EUR rate for a currency.",
            "parameters": {
                "type": "object",
                "properties": {"currency": {"type": "string"}},
                "required": ["currency"],
            },
        },
    }
]
DISPATCH = {"exchange_rate": exchange_rate}


def answer_tool_call_before(name: str, arguments: str) -> str:
    return json.dumps(DISPATCH[name](**json.loads(arguments)))  # KeyError, TypeError leak out


# --8<-- [end:before]

# --8<-- [start:after]
registry = Registry()


@registry.tool(timeout=5, retries=1)
def rate(currency: str) -> float:
    """Today's EUR rate for a currency."""
    return {"USD": 0.92, "GBP": 1.17}[currency]


executor = Executor(registry)


async def answer_tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """One OpenAI tool call in, one `role="tool"` message out; failures become text."""
    result = await executor.run_call(
        ToolCall(name=name, arguments=json.loads(arguments), id=call_id)
    )
    content = (
        json.dumps(result.value)
        if result.ok
        else json.dumps(result.error.model_dump() if result.error else {})
    )
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# --8<-- [end:after]


async def main() -> None:
    print(answer_tool_call_before("exchange_rate", '{"currency": "USD"}'))
    # --8<-- [start:compare]
    print(chat_tools(registry)[0]["function"]["parameters"]["required"])  # generated
    print(await answer_tool_call("c1", "rate", '{"currency": "USD"}'))
    bad = await answer_tool_call("c2", "rate", '{"currency": 7}')
    print(json.loads(bad["content"])["kind"])  # the model reads the error and retries
    print([tool["name"] for tool in anthropic_tools(registry)])  # the same tool, Anthropic shape
    blocks = tool_result_blocks(
        [await executor.run_call(ToolCall(name="rate", arguments={"currency": "GBP"}, id="tu_1"))]
    )
    print(blocks[0]["type"], blocks[0]["tool_use_id"])
    # --8<-- [end:compare]
    # --8<-- [start:loop]
    replies: list[Any] = [
        {"tool_calls": [ToolCall(name="rate", arguments={"currency": "USD"}, id="c3")]},
        "1 USD is 0.92 EUR.",
    ]

    async def fake_model(request: LLMRequest) -> Any:  # in production: OpenAIModel(...)
        return replies.pop(0)

    question = (Message(role="user", content="USD in EUR?"),)
    done = await run_tool_loop(CallableModel(fake_model), registry, question, executor=executor)
    print(done.finish, done.final.text)
    # --8<-- [end:loop]


if __name__ == "__main__":
    asyncio.run(main())
