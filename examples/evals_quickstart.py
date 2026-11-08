"""Evaluate a shop support agent offline: cases, graders, a report, and a regression check.

A scripted model stands in for a real one, so it runs with no network and no API key.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from agent_adapter import (
    CallableModel,
    Executor,
    Ledger,
    LLMRequest,
    LLMResponse,
    Message,
    Prices,
    Registry,
    ToolCall,
    Usage,
)
from agent_adapter.evals import EvalReport, LLMJudge, ToolLoopTarget, load_cases, run_eval_sync
from agent_adapter.evals.graders import contains, max_cost, tools_called

registry = Registry()
ORDERS = {"1042": "shipped", "1043": "packed"}


@registry.tool
def order_status(order_id: str) -> str:
    """Where an order is."""
    return ORDERS.get(order_id, "unknown")


def scripted_agent(request: LLMRequest) -> LLMResponse:
    """Stands in for the agent's model: look the order up, then answer with its status."""
    asked = request.messages[-1].content if request.messages[-1].role == "user" else ""
    results = [m for m in request.messages if m.role == "tool"]
    usage = Usage(input_tokens=120, output_tokens=20)
    order = next((w.strip("?.!") for w in asked.split() if w.strip("?.!").isdigit()), "")
    if order and not results:
        call = ToolCall(name="order_status", arguments={"order_id": order}, id="call_1")
        message = Message(role="assistant", tool_calls=(call,))
        return LLMResponse(
            provider="local", model="scripted", tool_calls=(call,), usage=usage, message=message
        )
    text = f"Your order is {results[0].content}." if results else "Which order do you mean?"
    message = Message(role="assistant", content=text)
    return LLMResponse(provider="local", model="scripted", text=text, usage=usage, message=message)


def scripted_judge(request: LLMRequest) -> str:
    """Stands in for the judge model: polite answers pass."""
    answer = request.messages[-1].content
    polite = "order" in answer.lower()
    return json.dumps({"value": 1.0 if polite else 0.0, "passed": polite, "notes": "tone"})


PRICES = Prices.empty().merged(
    {"version": "demo", "models": {"scripted": {"provider": "local", "input": 1.0, "output": 4.0}}}
)
# The same JSONL the chat page's "save as eval case" writes; extra fields are ignored.
CASES = [
    {"id": "shipped", "input": "Where is order 1042?", "expected": "shipped", "tags": ["orders"]},
    {"id": "packed", "input": "Status of 1043 please", "expected": "packed", "tags": ["orders"]},
    {"id": "refund", "input": "Has 1042 been refunded?", "expected": "shipped", "tags": ["chat"]},
]


def lazy_agent(request: LLMRequest) -> LLMResponse:
    """A worse version of the agent: it never looks the order up."""
    text = "Which order do you mean?"
    message = Message(role="assistant", content=text)
    usage = Usage(input_tokens=120, output_tokens=10)
    return LLMResponse(provider="local", model="scripted", text=text, usage=usage, message=message)


# --8<-- [start:run]
def evaluate(model_fn: Any, cases_file: Path) -> EvalReport:
    """Run every case through an executor with prices, and grade every run."""
    agent = CallableModel(model_fn, provider="local", model="scripted")
    judge = CallableModel(scripted_judge, provider="local", model="scripted")
    return run_eval_sync(
        ToolLoopTarget(agent, registry, system="You answer shop order questions."),
        load_cases(cases_file),
        {
            "answer": contains(),
            "looked_up": tools_called("order_status"),
            "polite": LLMJudge(judge, "The reply is polite and mentions the order."),
            "budget": max_cost(0.01),
        },
        executor=Executor(registry, ledger=Ledger(prices=PRICES)),
        name="shop-support",
    )


# --8<-- [end:run]


def main() -> dict[str, Any]:
    """Evaluate the agent, print the report, then catch a worse version as a regression."""
    with tempfile.TemporaryDirectory() as folder:
        cases_file = Path(folder) / "eval_cases.jsonl"
        cases_file.write_text("".join(json.dumps(c) + "\n" for c in CASES), encoding="utf-8")
        baseline = evaluate(scripted_agent, cases_file)
        print(baseline.to_markdown())
        # --8<-- [start:compare]
        candidate = evaluate(lazy_agent, cases_file)
        comparison = candidate.compare(baseline)
        # --8<-- [end:compare]
    regressed = [change.case_id for change in comparison.regressions]
    print("regressions:", regressed)
    print(f"JUnit: {baseline.to_junit().count('<testcase ')} test cases")
    return {
        "passed": baseline.n_passed,
        "cases": baseline.n_cases,
        "cost_usd": baseline.cost_usd,
        "regressions": regressed,
    }


if __name__ == "__main__":
    main()
