"""Compress what Pydantic AI, Agno, Strands, smolagents, AutoGen and AG2 send to the model.

Each hook takes a `CompressionPolicy` and works on that framework's own message type.
`main()` runs the Strands hook offline; the other hooks are run by their contract tests.
"""

import asyncio
import dataclasses
from typing import Any

from agent_adapter.context import CompressionPolicy, RemoveOldMessages
from agent_adapter.integrations.ag2 import ag2_compress_policy
from agent_adapter.integrations.agno import agno_compressed_model
from agent_adapter.integrations.autogen import autogen_model_context
from agent_adapter.integrations.pydantic_ai import pydantic_ai_history_processor
from agent_adapter.integrations.smolagents import smolagents_compressed_model
from agent_adapter.integrations.strands import strands_compress_middleware

OLD = "an old turn with a lot of detail " * 30
policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)


# --8<-- [start:hooks]
def pydantic_ai_agent(model: Any) -> Any:
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import ProcessHistory

    return Agent(model, capabilities=[ProcessHistory(pydantic_ai_history_processor(policy))])


def agno_agent(model: Any) -> Any:  # model: any Agno model, e.g. OpenAIChat(id="gpt-4.1")
    from agno.agent import Agent

    return Agent(model=agno_compressed_model(model, policy))


def strands_agent(agent: Any) -> Any:  # agent: a strands.Agent
    from strands.middleware import InvokeModelStage

    agent.add_middleware(InvokeModelStage.Input, strands_compress_middleware(policy))
    return agent


def smolagents_agent(model: Any, tools: list[Any]) -> Any:
    from smolagents import ToolCallingAgent

    return ToolCallingAgent(tools=tools, model=smolagents_compressed_model(model, policy))


def autogen_agent(model_client: Any) -> Any:
    from autogen_agentchat.agents import AssistantAgent

    context = autogen_model_context(policy)
    return AssistantAgent("assistant", model_client=model_client, model_context=context)


def ag2_agent(config: Any) -> Any:  # config: an AG2 ModelConfig
    from ag2 import Agent

    return Agent("assistant", config=config, assembly=[ag2_compress_policy(policy)])


# --8<-- [end:hooks]


@dataclasses.dataclass
class _InvokeModelContext:  # the fields of strands.middleware.InvokeModelContext used here
    messages: list[dict[str, Any]]
    system_prompt: str

    def replace(self, **changes: Any) -> "_InvokeModelContext":
        return dataclasses.replace(self, **changes)


def main() -> None:
    history = [{"role": role, "content": [{"text": OLD}]} for role in ("user", "assistant") * 3]
    history.append({"role": "user", "content": [{"text": "And now?"}]})
    context = _InvokeModelContext(messages=history, system_prompt="Be brief.")
    sent = asyncio.run(strands_compress_middleware(policy)(context))
    print("strands:", len(history), "->", len(sent.messages), sent.system_prompt)


if __name__ == "__main__":
    main()
