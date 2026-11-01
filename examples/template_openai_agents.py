"""Starter OpenAI Agents SDK agent: registry tools via attach(). Run: python agent.py"""

import os

from agents import Agent, Runner

from agent_adapter import Executor, Registry, attach

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


agent = Agent(
    name="shop helper",
    instructions="Answer with the tools.",
    tools=attach("openai-agents", registry, executor=Executor(registry)),
)

if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        print(Runner.run_sync(agent, "What is the shipping rate to the US?").final_output)
    else:
        print("tools:", [tool.name for tool in agent.tools], "- set OPENAI_API_KEY to run")
