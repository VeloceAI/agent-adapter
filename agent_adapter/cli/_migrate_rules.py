"""What `agent-adapter migrate` looks for: framework tool APIs by their import path."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bridge:
    """How the registry reaches one framework: the integration module and the call."""

    module: str
    function: str
    call: str


def _bridge(module: str, function: str, *, executor: bool = True) -> Bridge:
    args = "registry, executor=executor" if executor else "registry"
    return Bridge(f"agent_adapter.integrations.{module}", function, f"{function}({args})")


BRIDGES: dict[str, Bridge] = {
    "langchain": _bridge("langchain", "structured_tools"),
    "crewai": _bridge("crewai", "crewai_tools"),
    "openai-agents": _bridge("openai_agents", "function_tools"),
    "adk": _bridge("adk", "adk_tools"),
    "pydantic-ai": _bridge("pydantic_ai", "pydantic_ai_tools"),
    "llamaindex": _bridge("llamaindex", "llamaindex_tools"),
    "smolagents": _bridge("smolagents", "smolagents_tools"),
    "agno": _bridge("agno", "agno_functions"),
    "strands": _bridge("strands", "strands_tools"),
    "openai": _bridge("openai", "chat_tools", executor=False),
    "anthropic": _bridge("anthropic", "anthropic_tools", executor=False),
}
FRAMEWORKS: tuple[str, ...] = tuple(BRIDGES)

# Decorators that turn a function into a framework tool.
DECORATORS: dict[str, str] = {
    "langchain_core.tools.tool": "langchain",
    "langchain_core.tools.convert.tool": "langchain",
    "langchain.tools.tool": "langchain",
    "langchain.agents.tool": "langchain",
    "crewai.tools.tool": "crewai",
    "crewai.tool": "crewai",
    "agents.function_tool": "openai-agents",
    "agents.tool.function_tool": "openai-agents",
    "smolagents.tool": "smolagents",
    "agno.tools.tool": "agno",
    "agno.tools.decorator.tool": "agno",
    "strands.tool": "strands",
}

# Calls that build a framework tool from a function: framework, then the function's keyword.
WRAPPERS: dict[str, tuple[str, str]] = {
    "google.adk.tools.FunctionTool": ("adk", "func"),
    "google.adk.tools.function_tool.FunctionTool": ("adk", "func"),
    "llama_index.core.tools.FunctionTool.from_defaults": ("llamaindex", "fn"),
    "llama_index.core.tools.function_tool.FunctionTool.from_defaults": ("llamaindex", "fn"),
    "langchain_core.tools.StructuredTool.from_function": ("langchain", "func"),
    "langchain.tools.StructuredTool.from_function": ("langchain", "func"),
    "pydantic_ai.Tool": ("pydantic-ai", "function"),
}

# Agents whose `tools=[...]` takes plain functions.
AGENTS: dict[str, str] = {
    "google.adk.agents.Agent": "adk",
    "google.adk.agents.LlmAgent": "adk",
    "google.adk.Agent": "adk",
    "pydantic_ai.Agent": "pydantic-ai",
    "agno.agent.Agent": "agno",
    "strands.Agent": "strands",
}

# Base classes of hand-written tool classes: these need a manual move.
SUBCLASSES: dict[str, str] = {
    "crewai.tools.BaseTool": "crewai",
    "crewai.tools.base_tool.BaseTool": "crewai",
    "langchain_core.tools.BaseTool": "langchain",
    "langchain.tools.BaseTool": "langchain",
    "smolagents.Tool": "smolagents",
}

# Pydantic AI registers tools on an agent instance: `@agent.tool` and `@agent.tool_plain`.
PYDANTIC_AI_AGENT = "pydantic_ai.Agent"
PYDANTIC_AI_RUN_CONTEXT = "pydantic_ai.RunContext"
PYDANTIC_AI_DECORATORS = ("tool", "tool_plain")
