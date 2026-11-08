"""Production runs become eval cases: record, save the cases, evaluate in CI, fix, evaluate again.

The runs come from a shop support app's OpenTelemetry export (content captured), so this runs
offline with no model and no API key.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from agent_adapter.evals import cases_from_runs, load_cases, run_eval_sync, write_cases
from agent_adapter.evals.graders import contains
from agent_adapter.viewer import import_otlp

NS = 1_000_000_000  # nanoseconds per second


def attr(key: str, value: object) -> dict[str, object]:
    """One OTLP JSON attribute; lists of messages travel as JSON text."""
    text = value if isinstance(value, str) else json.dumps(value)
    return {"key": key, "value": {"stringValue": text}}


def chat_span(trace: str, start: float, asked: str, answer: str | None) -> dict[str, Any]:
    """One model call of trace `trace`; no answer means the call failed."""
    attrs = [
        attr("gen_ai.operation.name", "chat"),
        attr("gen_ai.request.model", "shop-model"),
        attr(
            "gen_ai.input.messages",
            [{"role": "user", "parts": [{"type": "text", "content": asked}]}],
        ),
    ]
    if answer is not None:
        output = [{"role": "assistant", "parts": [{"type": "text", "content": answer}]}]
        attrs.append(attr("gen_ai.output.messages", output))
    return {
        "traceId": trace,
        "spanId": f"{trace}-s1",
        "name": "chat shop-model",
        "startTimeUnixNano": str(int(start * NS)),
        "endTimeUnixNano": str(int((start + 1) * NS)),
        "attributes": attrs,
        "status": {"code": 1 if answer is not None else 2},
    }


# What production recorded: one good answer and one failed call.
TRACES = {
    "resourceSpans": [
        {
            "scopeSpans": [
                {
                    "spans": [
                        chat_span(
                            "t1", 1_800_000_000.0, "Where is order 1042?", "Order 1042 shipped."
                        ),
                        chat_span("t2", 1_800_000_010.0, "Cancel order 1043", None),
                    ]
                }
            ]
        }
    ]
}


def old_agent(question: str) -> str:
    """The agent that failed in production: it cannot cancel."""
    order = next((w for w in question.split() if w.strip("?").isdigit()), "").strip("?")
    if question.lower().startswith("cancel"):
        raise RuntimeError("no cancel tool")
    return f"Order {order} shipped."


def fixed_agent(question: str) -> str:
    """The fix: cancelling works."""
    if question.lower().startswith("cancel"):
        return f"Order {question.split()[-1]} is cancelled."
    return old_agent(question)


# --8<-- [start:loop]
def main(folder: Path) -> list[str]:
    """Turn the recorded runs into cases, evaluate the old and the fixed agent on them."""
    source = folder / "traces.json"
    source.write_text(json.dumps(TRACES), encoding="utf-8")
    runs = folder / "runs"
    import_otlp(source, runs)  # or runs your app recorded with capture_content=True

    found = cases_from_runs(runs)  # the failed run's case has no expected value yet
    cases_file = folder / "eval_cases.jsonl"
    write_cases(cases_file, found.cases)

    # A person fills in the expected answer of the regression case (here: in code).
    cases = [
        case if case.expected is not None else case.model_copy(update={"expected": "cancelled"})
        for case in load_cases(cases_file)
    ]
    lines = [f"{case.id}: {case.input!r} -> {case.expected!r} {list(case.tags)}" for case in cases]
    for agent in (old_agent, fixed_agent):
        report = run_eval_sync(agent, cases, [contains()])
        lines.append(f"{agent.__name__}: {report.n_passed}/{report.n_cases} passed")
    return lines


# --8<-- [end:loop]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        for line in main(Path(tmp)):
            print(line)
