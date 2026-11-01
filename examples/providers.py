"""One tool, three providers: render it for each and run the model's tool calls."""

# --8<-- [start:define]
import asyncio
import json
from typing import Any, Literal

from agent_adapter import Registry

registry = Registry()


@registry.tool
def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Current weather for a city.

    Args:
        city: City name, e.g. Paris.
        unit: Temperature unit.
    """
    return f"{city}: 31 degrees {unit.upper()}"


# --8<-- [end:define]


async def openai_round_trip(response: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render for OpenAI Chat Completions and answer each tool call through the executor."""
    # --8<-- [start:openai]
    from agent_adapter import Executor, ToolCall
    from agent_adapter.integrations.openai import chat_tools

    executor = Executor(registry)  # once per process; no approval port, so confirm tools are denied
    tools = chat_tools(registry)  # pass as tools= to chat.completions.create
    replies = []
    for call in response.choices[0].message.tool_calls:
        # arguments is a JSON string; strict mode sends null for "use the default".
        args = json.loads(call.function.arguments)
        result = await executor.run_call(
            ToolCall(name=call.function.name, arguments=args, id=call.id)
        )
        data = result.to_dict()  # a denial, bad arguments or an unknown tool land in "error"
        payload = data["value"] if result.ok else {"error": data["error"]}
        replies.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(payload)})
    # --8<-- [end:openai]
    return tools, replies


async def gemini_round_trip() -> tuple[list[dict[str, Any]], Any]:
    """Render Gemini function declarations and run a function call through the executor."""
    # --8<-- [start:gemini]
    from agent_adapter import Executor
    from agent_adapter.integrations.gemini import function_declarations

    executor = Executor(registry)  # once per process
    declarations = function_declarations(registry)  # plain dicts
    # The model's function call. Gemini sends args as a dict.
    result = await executor.run("get_weather", {"city": "Paris"})  # the value; failures raise
    # --8<-- [end:gemini]
    return declarations, result


async def anthropic_round_trip(message: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render Anthropic Messages tools and answer the tool_use blocks through the executor."""
    # --8<-- [start:anthropic]
    from agent_adapter import Executor, ToolCall
    from agent_adapter.integrations.anthropic import anthropic_tools, tool_result_blocks

    executor = Executor(registry)  # once per process; no approval port, so confirm tools are denied
    tools = anthropic_tools(registry)  # pass as tools= to messages.create
    calls = [
        ToolCall(name=block.name, arguments=block.input, id=block.id)  # input is a dict
        for block in message.content
        if block.type == "tool_use"
    ]
    results = await executor.run_many(calls)  # one ToolResult per call, in order
    reply = {"role": "user", "content": tool_result_blocks(results)}
    # --8<-- [end:anthropic]
    return tools, reply


def fake_chat_completion() -> Any:
    """A Chat Completions response shaped like the SDK's, holding one tool call."""
    from types import SimpleNamespace as Fake

    function = Fake(name="get_weather", arguments='{"city": "Paris", "unit": null}')
    call = Fake(id="call_1", type="function", function=function)
    return Fake(choices=[Fake(message=Fake(tool_calls=[call]))])


def fake_message() -> Any:
    """A Messages API response shaped like the SDK's, holding one tool_use block."""
    from types import SimpleNamespace as Fake

    block = Fake(
        type="tool_use", id="toolu_1", name="get_weather", input={"city": "Paris", "unit": "f"}
    )
    return Fake(content=[Fake(type="text", text="Checking."), block])


async def main() -> None:
    """Print each provider's first tool definition and the answer to the model's call."""
    rounds = [
        await openai_round_trip(fake_chat_completion()),
        await gemini_round_trip(),
        await anthropic_round_trip(fake_message()),
    ]
    for tools, answer in rounds:
        print(json.dumps(tools[0], indent=2))
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
