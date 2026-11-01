"""Run registry tools in smolagents: a direct call, then a `ToolCallingAgent` (offline model)."""

from typing import Any

from smolagents import ChatMessage, ChatMessageToolCall, MessageRole, ToolCallingAgent
from smolagents.models import ChatMessageToolCallFunction

from agent_adapter import Executor, Registry, ToolError
from agent_adapter.integrations.smolagents import smolagents_tools

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    if region == "mars":
        raise ToolError(f"no rate for {region!r}")
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


class OfflineModel:
    """Scripted turns: call shipping_rate, then give the final answer."""

    model_id = "offline"

    def __init__(self) -> None:
        self.turns = [("shipping_rate", {"region": "us"}), ("final_answer", {"answer": "18.0"})]

    def generate(self, messages: Any, stop_sequences: Any = None, **kwargs: Any) -> ChatMessage:
        name, arguments = self.turns.pop(0)
        function = ChatMessageToolCallFunction(name=name, arguments=arguments)
        call = ChatMessageToolCall(id=name, type="function", function=function)
        return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[call])

    __call__ = generate


def main() -> None:
    # --8<-- [start:tools]
    executor = Executor(registry)  # hooks, approval, cache, ledger apply to every call
    tools = smolagents_tools(registry, executor=executor)
    rate = tools[0]
    print(rate.name, rate.output_type, rate(region="us"))  # sync: forward uses run_sync
    print(rate(region="mars"))  # ToolError comes back as a model-visible error
    # --8<-- [end:tools]
    # --8<-- [start:agent]
    agent = ToolCallingAgent(tools=tools, model=OfflineModel(), verbosity_level=0)
    print("answer:", agent.run("Shipping rate to the US?"))
    # --8<-- [end:agent]


if __name__ == "__main__":
    main()
