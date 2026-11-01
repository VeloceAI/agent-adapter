"""DSPy builds OpenAI chat dicts from its History: the openai format, in an adapter subclass."""

from typing import Any

import pytest

pytest.importorskip("dspy")

import dspy
from dspy.utils.dummies import DummyLM

from tests.integrations.compression_oracle import PAD, STEPS, check_round_trip, check_safe, example

pytestmark = pytest.mark.contract


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


def history(turns: int) -> dspy.History:
    entries = [{"question": f"q{n} {PAD}", "answer": f"a{n} {PAD}"} for n in range(turns)]
    return dspy.History(messages=entries)


def formatted(turns: int) -> list[dict[str, Any]]:
    inputs = {"question": "new question", "history": history(turns)}
    return dspy.ChatAdapter().format(QA, [], inputs)


def test_dspy_messages_round_trip_as_openai():
    check_round_trip(formatted(3), "openai")


@pytest.mark.parametrize("step", list(STEPS))
def test_no_step_breaks_what_dspy_sends(step):
    items = formatted(3)
    smaller = check_safe(items, STEPS[step]())
    assert smaller[0] is items[0] and smaller[-1] is items[-1]


def test_docs_adapter_sends_fewer_messages_and_dspy_still_parses_the_answer():
    adapter = example("compression_dspy")["CompressingChatAdapter"]()
    lm = DummyLM([{"answer": "ok"}])
    with dspy.context(lm=lm, adapter=adapter):
        result = dspy.Predict(QA)(question="new question", history=history(6))
    assert result.answer == "ok"
    sent = lm.history[-1]["messages"]
    full = formatted(6)
    assert len(sent) < len(full) and sent[0] == full[0] and sent[-1] == full[-1]
