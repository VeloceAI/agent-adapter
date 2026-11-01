"""Make a LangGraph, OpenAI Agents or Pydantic AI agent optimisable with `CallableHost`.

Each `build_*_host(model)` takes that framework's own model object; the agent is rebuilt per
case from the tuned texts. The optimiser's `model` goes unused, so run with `NullModel()`
(CLI: `--model none`). Each builder raises `MissingDependency` without its SDK.
"""

from pathlib import Path
from typing import Any

from harness_agent import SURFACE, lookup_article

from agent_adapter import Executor, Registry
from agent_adapter.models import LLMPort
from agent_adapter.optimiser import (
    CallableHost,
    EvalCase,
    HarnessTree,
    RunCase,
    apply_tree,
    load_cases,
)
from agent_adapter.optimiser.graders import contains

CASES = Path(__file__).with_name("harness_cases.jsonl")
registry = Registry()
registry.wrap(lookup_article)


def system_prompt(tree: HarnessTree) -> str:
    """The tuned system prompt, with the tuned tool hint appended."""
    hint = tree.files.get("tool/lookup_article/prompt_hint", "")
    return tree.files["prompt/system"] + (f"\n\nWhen to use lookup_article: {hint}" if hint else "")


def _host(run_case: RunCase) -> CallableHost:
    return CallableHost(
        run_case, surface=SURFACE, cases=load_cases(CASES), judge=contains(), domain="support"
    )


def build_langgraph_host(chat_model: Any) -> CallableHost:
    """A LangGraph ReAct agent over a LangChain chat model."""
    from agent_adapter.integrations.langgraph import INTEGRATION

    INTEGRATION.require()

    async def run_case(tree: HarnessTree, case: EvalCase, model: LLMPort) -> str:
        # --8<-- [start:langgraph]
        from langgraph.prebuilt import create_react_agent

        from agent_adapter.integrations.langchain import structured_tools

        tuned = apply_tree(registry, tree)  # tuned tool descriptions
        tools = structured_tools(tuned, executor=Executor(tuned))
        agent = create_react_agent(chat_model, tools, prompt=system_prompt(tree))
        state = await agent.ainvoke({"messages": [("user", str(case.input))]})
        return str(state["messages"][-1].content)
        # --8<-- [end:langgraph]

    return _host(run_case)


def build_openai_agents_host(agents_model: Any) -> CallableHost:
    """An OpenAI Agents SDK agent over an `agents.Model`."""
    from agent_adapter.integrations.openai_agents import INTEGRATION

    INTEGRATION.require()

    async def run_case(tree: HarnessTree, case: EvalCase, model: LLMPort) -> str:
        # --8<-- [start:openai_agents]
        from agents import Agent, Runner

        from agent_adapter.integrations.openai_agents import function_tools

        tuned = apply_tree(registry, tree)
        tools = function_tools(tuned, executor=Executor(tuned))
        agent = Agent(
            name="support", instructions=system_prompt(tree), tools=tools, model=agents_model
        )
        result = await Runner.run(agent, str(case.input))
        return str(result.final_output)
        # --8<-- [end:openai_agents]

    return _host(run_case)


def build_pydantic_ai_host(pai_model: Any) -> CallableHost:
    """A Pydantic AI agent over a `pydantic_ai` model."""
    from agent_adapter.integrations.pydantic_ai import INTEGRATION

    INTEGRATION.require()

    async def run_case(tree: HarnessTree, case: EvalCase, model: LLMPort) -> str:
        # --8<-- [start:pydantic_ai]
        from pydantic_ai import Agent

        from agent_adapter.integrations.pydantic_ai import pydantic_ai_tools

        tuned = apply_tree(registry, tree)
        tools = pydantic_ai_tools(tuned, executor=Executor(tuned))
        agent = Agent(pai_model, system_prompt=system_prompt(tree), tools=tools)
        result = await agent.run(str(case.input))
        return str(result.output)
        # --8<-- [end:pydantic_ai]

    return _host(run_case)
