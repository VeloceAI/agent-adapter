"""Run registry tools in DSPy: `acall` directly, then inside `dspy.ReAct` (offline LM)."""

import asyncio

import dspy
from dspy.utils.dummies import DummyLM

from agent_adapter import Executor, Registry, ToolError
from agent_adapter.integrations.dspy import dspy_tools

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    if region == "mars":
        raise ToolError(f"no rate for {region!r}")
    return 18.0


def offline_lm() -> DummyLM:
    """Scripted ReAct turns: call shipping_rate, finish, then answer."""
    return DummyLM(
        [
            {
                "next_thought": "Look it up.",
                "next_tool_name": "shipping_rate",
                "next_tool_args": {"region": "us"},
            },
            {"next_thought": "Done.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "The tool said 18.", "answer": "18.0"},
        ]
    )


async def main() -> None:
    # --8<-- [start:tools]
    executor = Executor(registry)  # hooks, approval, cache, ledger apply to every call
    tools = dspy_tools(registry, executor=executor)
    rate = tools[0]
    print(rate.name, await rate.acall(region="us"))
    print(await rate.acall(region="mars"))  # ToolError comes back as a model-visible error
    # --8<-- [end:tools]
    # --8<-- [start:react]
    react = dspy.ReAct("question -> answer", tools=tools)
    with dspy.context(lm=offline_lm()):
        prediction = await react.acall(question="Shipping to the US?")
    print("answer:", prediction.answer)
    # --8<-- [end:react]


if __name__ == "__main__":
    asyncio.run(main())
