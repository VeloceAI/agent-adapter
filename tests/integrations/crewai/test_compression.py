"""CrewAI's messages are OpenAI chat dicts: the openai format, through CrewAI's own hook path."""

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("crewai")

from crewai.utilities.agent_utils import (
    _setup_before_llm_call_hooks,
    build_tool_calls_assistant_message,
)
from crewai.utilities.types import LLMMessage
from crewai_core.printer import Printer

from agent_adapter.context.formats import convert
from tests.integrations.compression_oracle import (
    DENIED,
    PAD,
    STEPS,
    check_round_trip,
    check_safe,
    example,
)

pytestmark = pytest.mark.contract
GEMINI_PART = {"thought_signature": "c2ln"}  # CrewAI keeps Gemini raw parts beside the calls


def calls(*ids: str) -> LLMMessage:
    raw = [{"id": i, "function": {"name": "lookup", "arguments": '{"q": 1}'}} for i in ids]
    message, _ = build_tool_calls_assistant_message(raw)
    assert message is not None
    return message


def tool(call_id: str, text: str) -> LLMMessage:
    return {"role": "tool", "tool_call_id": call_id, "name": "lookup", "content": text}


def conversation() -> list[LLMMessage]:
    gemini = calls("c3")
    gemini["raw_tool_call_parts"] = [GEMINI_PART]
    return [
        {"role": "system", "content": "You are a researcher."},
        {"role": "user", "content": f"old task {PAD}", "files": {}},
        calls("c1", "c2"),
        tool("c1", f"found {PAD}"),
        tool("c2", DENIED),
        {"role": "assistant", "content": f"Thought: old {PAD}"},
        {"role": "user", "content": f"middle {PAD}"},
        gemini,
        tool("c3", f"more {PAD}"),
        {"role": "assistant", "content": "Final Answer: ok"},
        {"role": "user", "content": "new task"},
    ]


def test_round_trip_gives_equal_values_and_the_same_objects():
    check_round_trip(conversation(), "openai")


@pytest.mark.parametrize("step", list(STEPS))
def test_no_step_orphans_a_tool_round_or_touches_protected_messages(step):
    items = conversation()
    smaller = check_safe(items, STEPS[step](), pinned=lambda m: "files" in m)
    assert items[1] in smaller and items[2] in smaller and items[4] in smaller


def test_cut_keeps_crewai_keys():
    items = conversation()
    smaller = check_safe(items, STEPS["cut"]())
    assert smaller[1]["files"] == {} and "characters cut" in smaller[1]["content"]
    assert smaller[7]["raw_tool_call_parts"] == [GEMINI_PART]
    assert smaller[8]["name"] == "lookup" and smaller[8]["tool_call_id"] == "c3"
    assert [c.id for m in convert(smaller).messages for c in m.tool_calls] == ["c1", "c2", "c3"]


def test_docs_hook_shortens_the_executor_list_in_place():
    hook = example("compression_crewai")["compress_crewai_messages"]
    messages: list[Any] = [m for _ in range(8) for m in conversation()[5:7]]
    messages.append({"role": "user", "content": "new task"})
    executor = SimpleNamespace(
        before_llm_call_hooks=[hook], messages=messages, llm=None, iterations=0, agent=None
    )
    executor.task = executor.crew = None
    assert _setup_before_llm_call_hooks(executor, Printer(), verbose=False)
    assert executor.messages is messages and len(messages) < 17
    assert messages[-1] == {"role": "user", "content": "new task"}
