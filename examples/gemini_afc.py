"""Run Gemini function calls through the executor, by hand or with genai AFC."""

import asyncio
import os

from google import genai
from google.genai import types

from agent_adapter import Executor, Registry

registry = Registry()


@registry.tool
def shipping_rate(item: str, region: str = "us") -> float:
    """Shipping rate for an item, in percent of the order value (by region)."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


async def manual() -> list[types.Part]:
    # --8<-- [start:manual]
    from agent_adapter.integrations.gemini import genai_tool, run_function_calls

    executor = Executor(registry)
    config = types.GenerateContentConfig(
        tools=[genai_tool(registry)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    # response = await client.aio.models.generate_content(model=..., contents=..., config=config)
    # calls = response.function_calls
    calls = [types.FunctionCall(id="fc_1", name="shipping_rate", args={"item": "laptop"})]
    parts = await run_function_calls(calls, executor=executor)  # one function response each
    # --8<-- [end:manual]
    assert config.tools is not None
    return parts


async def automatic(client: genai.Client) -> str | None:
    # --8<-- [start:afc]
    from agent_adapter.integrations.gemini import afc_callables

    executor = Executor(registry)
    config = types.GenerateContentConfig(tools=afc_callables(registry, executor=executor))
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash", contents="Shipping a laptop to the US?", config=config
    )
    # --8<-- [end:afc]
    return response.text


async def main() -> None:
    parts = await manual()
    response = parts[0].function_response
    assert response is not None
    print(response.name, response.response)
    if os.environ.get("GEMINI_API_KEY"):  # the AFC round trip needs a real model
        print(await automatic(genai.Client()))


if __name__ == "__main__":
    asyncio.run(main())
