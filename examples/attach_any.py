"""attach(): one call gives any framework its native tools; approval stays on the executor."""

from agent_adapter import AutoDeny, Executor, Registry, attach
from agent_adapter.errors import IntegrationError, ToolUsageError

# --8<-- [start:attach]
registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


executor = Executor(registry, approval=AutoDeny())  # the safety rules, for every framework

tools = attach("openai", registry, executor=executor)  # by name: Chat Completions tools
# tools = attach(agent, registry, executor=executor)   # by object: agents.Agent, StateGraph...
# --8<-- [end:attach]


def main() -> None:
    """Print the attached tool, then the two errors a newcomer meets first."""
    print(tools[0]["function"]["name"])
    # --8<-- [start:errors]
    try:
        attach("openai-agent", registry)  # a typo
    except IntegrationError as error:
        print(str(error).split(" (installed:")[0])  # the rest lists every integration
    try:
        attach("openai", registry, approval=True)  # never: approval is the executor's
    except ToolUsageError as error:
        print(str(error).split(":")[0])
    # --8<-- [end:errors]


if __name__ == "__main__":
    main()
