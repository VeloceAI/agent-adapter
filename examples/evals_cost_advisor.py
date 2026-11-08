"""Find a cheaper model per flow node that keeps the eval score, offline.

A support desk flow triages a ticket, then writes a reply. Three scripted models with
different prices stand in for real ones: the advisor runs the evals on the expensive one,
tries the cheaper ones node by node, and suggests a mapping. Nothing is applied for you.
"""

from typing import Any

from pydantic import BaseModel

from agent_adapter import (
    CallableModel,
    Executor,
    Flow,
    FlowRunner,
    LLMRequest,
    Message,
    Registry,
    ToolContext,
    run_sync,
)
from agent_adapter.errors import ConfigError
from agent_adapter.evals import EvalCase, FlowTarget
from agent_adapter.evals.advisor import AdvisorReport, advise_sync, apply_mapping
from agent_adapter.evals.graders import code_check
from agent_adapter.flow import node_model


class Ticket(BaseModel):
    text: str
    category: str = ""
    reply: str = ""


async def ask(ctx: ToolContext, stage: str, text: str) -> str:
    """One model call from a node, on whatever model the runner gives this node."""
    model = node_model(ctx)
    if model is None:
        raise ConfigError("this flow needs FlowRunner(model=...)")
    request = LLMRequest(messages=(Message(role="user", content=text),), stage=stage)
    return (await model.generate(request)).text


async def triage(state: Ticket, ctx: ToolContext) -> dict[str, str]:
    return {"category": await ask(ctx, "triage", state.text)}


async def reply(state: Ticket, ctx: ToolContext) -> dict[str, str]:
    return {"reply": await ask(ctx, "reply", f"{state.category}: {state.text}")}


flow = (
    Flow("support_desk", state=Ticket)
    .node("triage", triage)
    .node("reply", reply)
    .edge("triage", "reply")
    .end("reply")
)


def scripted(name: str, *, writes_well: bool) -> CallableModel:
    """A stand-in model: every one triages well; only `writes_well` ones quote the ticket."""

    def answer(request: LLMRequest) -> dict[str, Any]:
        text = str(request.messages[-1].content)
        if request.stage == "triage":
            out = "billing" if "charged" in text else "shipping"
        else:
            ticket = text.split("Ticket ")[-1].split(":")[0]
            out = f"About ticket {ticket}: we are on it." if writes_well else "We are on it."
        return {"text": out, "usage": {"input_tokens": 200, "output_tokens": 50}}

    return CallableModel(answer, provider="local", model=name)


# --8<-- [start:advise]
CANDIDATES = {
    "large": scripted("large", writes_well=True),
    "medium": scripted("medium", writes_well=True),
    "small": scripted("small", writes_well=False),
}
PRICES = {  # USD per 1M tokens: your own table
    "version": "demo",
    "models": {
        "large": {"provider": "local", "input": 10.0, "output": 40.0},
        "medium": {"provider": "local", "input": 2.0, "output": 8.0},
        "small": {"provider": "local", "input": 0.2, "output": 0.8},
    },
}
CASES = [
    EvalCase(id="refund", input="Ticket 17: I was charged twice", expected="billing"),
    EvalCase(id="late", input="Ticket 18: my parcel is late", expected="shipping"),
]


def good_reply(case: EvalCase, run: Any) -> bool:
    """Right category, and the reply names the ticket."""
    ticket = str(case.input).split(":")[0].split()[-1]
    return run.output.category == case.expected and ticket in run.output.reply


def target(model: Any) -> FlowTarget:
    """The eval target around whatever model the advisor hands it."""
    return FlowTarget(flow, initial_state=lambda case: {"text": case.input}, model=model)


def run_advisor() -> AdvisorReport:
    return advise_sync(
        target,
        CASES,
        {"reply": code_check(good_reply)},
        candidates=CANDIDATES,
        baseline="large",
        prices=PRICES,
        max_spend=0.50,
    )


# --8<-- [end:advise]


def main() -> dict[str, Any]:
    """Advise, print the report, then apply the mapping by hand and run the flow once."""
    report = run_advisor()
    print(report.to_markdown())
    # --8<-- [start:apply]
    model = apply_mapping(CANDIDATES["large"], report.mapping, CANDIDATES)
    runner = FlowRunner(Executor(Registry()), model=model)
    # --8<-- [end:apply]
    done = run_sync(runner.run(flow, {"text": "Ticket 19: charged twice again"}))
    print(done.state.category, "|", done.state.reply)
    return {
        "mapping": report.mapping,
        "spent": report.spent_usd,
        "reply": done.state.reply,
        "baseline_cost": report.baseline_trial.cost_usd if report.baseline_trial else 0.0,
        "combined_cost": report.combined.cost_usd if report.combined else 0.0,
    }


if __name__ == "__main__":
    main()
