"""Optimise a tool-calling agent's harness offline, then review the patch (nothing is applied)."""

import asyncio
import json
import tempfile
from pathlib import Path

from agent_adapter import CallableModel, LLMRequest, LLMResponse, Message, Registry, ToolCall
from agent_adapter.cache import PromptLayout
from agent_adapter.optimiser import (
    CaseRun,
    EvalCase,
    OptimiserConfig,
    RegistryHost,
    Score,
    optimise,
)

registry = Registry()


@registry.tool(parallel_safe=True)
def find_article(topic: str) -> str:
    """Look up the help-center article that covers a topic."""
    return {"returns": "Article HC-101", "shipping": "Article HC-201"}.get(topic, "not found")


# --8<-- [start:host]
async def judge(case: EvalCase, run: CaseRun) -> Score:
    """Exact match: does the answer name the expected article?"""
    hit = case.expected in run.output
    return Score(value=1.0 if hit else 0.0, passed=hit, notes="" if hit else "no article named")


host = RegistryHost(
    registry,
    layout=PromptLayout(static="You answer customer-support questions."),
    cases=[
        EvalCase(id="returns", input="returns", expected="Article HC-101"),
        EvalCase(id="shipping", input="shipping", expected="Article HC-201"),
    ],
    judge=judge,
    domain="support",
)
# --8<-- [end:host]


async def agent(request: LLMRequest) -> LLMResponse:
    """Stands in for the real model: it looks the topic up only when the harness says so."""
    last = request.messages[-1]
    if last.role == "tool":
        return text_reply(f"That is {last.content}.")
    if "find_article" in request.messages[0].content:
        call = ToolCall(name="find_article", arguments={"topic": last.content}, id="c1")
        return LLMResponse(
            provider="demo",
            model="agent",
            tool_calls=(call,),
            finish_reason="tool_calls",
            message=Message(role="assistant", tool_calls=(call,)),
        )
    return text_reply("I believe a refund applies.")


async def proposer(request: LLMRequest) -> LLMResponse:
    """Stands in for the proposer model: the knowledge editor adds one rule, the rest keep."""
    if request.stage == "propose:know1":
        edit = {
            "path": "prompt/system",
            "content": "You answer customer-support questions. "
            "Call find_article for the topic before you answer, and name the article.",
            "delete": False,
            "reason": "cases returns and shipping: answers name no article and no tool was called",
        }
        reply = {"keep_base": False, "notes": "require a lookup", "edits": [edit]}
    else:
        reply = {"keep_base": True, "notes": "no procedure change needed", "edits": []}
    return text_reply(json.dumps(reply))


def text_reply(content: str) -> LLMResponse:
    return LLMResponse(
        provider="demo",
        model="demo",
        text=content,
        message=Message(role="assistant", content=content),
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as out:
        # --8<-- [start:run]
        config = OptimiserConfig(k=1, out_dir=out, lessons_path="")
        result = await optimise(
            host, config, proposer=CallableModel(proposer), model=CallableModel(agent)
        )
        # --8<-- [end:run]
        print(f"chosen: {result.chosen}")
        print(f"base mean {result.base.mean_score} -> {result.probes['agg1'].mean_score}")
        used = result.budget
        print(f"case runs {used.case_runs}, proposer calls {used.proposer_calls}")
        print(result.patch.splitlines()[0])
        print(sorted(p.name for p in Path(out).iterdir() if p.is_file()))
    print(host.layout.static)  # the host itself is unchanged


if __name__ == "__main__":
    asyncio.run(main())
