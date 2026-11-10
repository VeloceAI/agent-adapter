"""DSPy prompts: lm15 messages in a `Prompt` (3.5+, "dspy") or OpenAI chat dicts (3.4)."""

import dataclasses
from typing import Any

import pytest

pytest.importorskip("dspy")

import dspy
from dspy.utils.dummies import DummyLM

from agent_adapter.integrations.dspy import dspy_compress_prompt
from tests.integrations.compression_oracle import (
    PAD,
    STEPS,
    check_round_trip,
    check_safe,
    example,
    policy,
)

pytestmark = pytest.mark.contract
PROMPT = hasattr(dspy.adapters, "Prompt")  # DSPy 3.5 adapters return a Prompt of lm15 messages
FORMAT = "dspy" if PROMPT else "openai"


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


def history(turns: int) -> dspy.History:
    entries = [{"question": f"q{n} {PAD}", "answer": f"a{n} {PAD}"} for n in range(turns)]
    return dspy.History(messages=entries)


def formatted(turns: int) -> Any:
    inputs = {"question": "new question", "history": history(turns)}
    return dspy.ChatAdapter().format(QA, [], inputs)


def natives(prompt: Any) -> list[Any]:
    """What DSPy sends as one list: the system text first, as in DSPy 3.4's chat dicts."""
    if not PROMPT:
        return list(prompt)
    from dspy.lm15 import Message

    return [Message.developer(prompt.system), *prompt.messages]


def as_dicts(prompt: Any) -> list[dict[str, Any]]:
    if not PROMPT:
        return list(prompt)
    from dspy.adapters.base import prompt_to_openai_messages

    return prompt_to_openai_messages(prompt)


def test_dspy_messages_round_trip():
    check_round_trip(natives(formatted(3)), FORMAT)


@pytest.mark.parametrize("step", list(STEPS))
def test_no_step_breaks_what_dspy_sends(step):
    items = natives(formatted(3))
    smaller = check_safe(items, STEPS[step]())
    assert smaller[0] is items[0] and smaller[-1] is items[-1]


@pytest.mark.parametrize("step", ["drop_old", "summary", "own_wipe"])
def test_compress_prompt_keeps_the_system_text_and_the_new_input(step):
    prompt = formatted(3)
    smaller, report = dspy_compress_prompt(prompt, policy(STEPS[step]()))
    assert type(smaller) is type(prompt)
    if PROMPT:
        assert smaller.system == prompt.system
        assert smaller.messages[-1] is prompt.messages[-1]
    else:
        assert smaller[0] is prompt[0] and smaller[-1] is prompt[-1]
    assert report.tokens_after <= report.tokens_before


def test_docs_adapter_sends_fewer_messages_and_dspy_still_parses_the_answer():
    adapter = example("compression_dspy")["CompressingChatAdapter"]()
    lm = DummyLM([{"answer": "ok"}])
    with dspy.context(lm=lm, adapter=adapter):
        result = dspy.Predict(QA)(question="new question", history=history(6))
    assert result.answer == "ok"
    sent = lm.history[-1]["messages"]
    full = as_dicts(formatted(6))
    assert len(sent) < len(full) and sent[0] == full[0] and sent[-1] == full[-1]


def tool_round() -> list[Any]:
    """A native tool-calling turn: a call, then two results in one tool message."""
    from dspy.lm15 import Message, TextPart, ToolCallPart, ToolResultPart

    calls = [ToolCallPart(id=f"c{n}", name="lookup", input={"n": n}) for n in range(2)]
    results = [
        ToolResultPart(id=f"c{n}", name="lookup", content=(TextPart(f"r{n} {PAD}"),))
        for n in range(2)
    ]
    return [
        Message.developer("Be brief."),
        Message.user(f"old {PAD}"),
        Message(role="assistant", parts=[TextPart("Looking."), *calls]),
        Message(role="tool", parts=results),
        Message.user("new question"),
    ]


needs_prompt = pytest.mark.skipif(not PROMPT, reason="lm15 messages arrive with DSPy 3.5")


@needs_prompt
def test_tool_rounds_round_trip():
    check_round_trip(tool_round(), "dspy")


@needs_prompt
@pytest.mark.parametrize("step", list(STEPS))
def test_no_step_breaks_a_tool_round(step):
    items = tool_round()
    smaller = check_safe(items, STEPS[step]())
    assert smaller[0] is items[0] and smaller[-1] is items[-1]


@needs_prompt
def test_shrunk_tool_results_keep_their_ids_and_names():
    smaller = check_safe(tool_round(), STEPS["shrink"]())
    results = [part for m in smaller if m.role == "tool" for part in m.parts]
    assert [(r.id, r.name) for r in results] == [("c0", "lookup"), ("c1", "lookup")]
    assert all(len(r.content[0].text) < len(PAD) for r in results)


@needs_prompt
def test_messages_a_step_writes_become_lm15_messages():
    from dspy.lm15 import Message as Lm15Message

    from agent_adapter.core.messages import Message
    from agent_adapter.core.types import ToolCall
    from agent_adapter.integrations.dspy_compress import DSPyConverter

    call = ToolCall(id="c9", name="lookup", arguments={"n": 9})
    rendered = DSPyConverter([]).render(
        [
            Message(role="system", content="rules"),
            Message(role="user", content="summary"),
            Message(role="assistant", content="", tool_calls=(call,)),
            Message(role="tool", tool_call_id="c9", content="9"),
        ]
    )
    assert all(isinstance(m, Lm15Message) for m in rendered)
    assert [m.role for m in rendered] == ["developer", "user", "assistant", "tool"]
    assert [p.type for p in rendered[2].parts] == ["tool_call"]
    assert rendered[3].parts[0].id == "c9"


@needs_prompt
def test_compress_prompt_honours_pinned_without_system_text():
    bare = dataclasses.replace(formatted(3), system=None)
    first = bare.messages[0]
    pinned = policy(STEPS["drop_old"]())
    smaller, _ = dspy_compress_prompt(bare, pinned, pinned=lambda message: message is first)
    assert smaller.system is None
    assert smaller.messages[0] is first and smaller.messages[-1] is bare.messages[-1]
    assert len(smaller.messages) < len(bare.messages)
