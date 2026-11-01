"""The camel compression format on real `MemoryRecord`s, and the context-creator example."""

import json
from typing import Any

import pytest

pytest.importorskip("camel")

from camel.agents import ChatAgent
from camel.memories import ChatHistoryMemory, MemoryRecord
from camel.messages import BaseMessage, FunctionCallingMessage
from camel.types import OpenAIBackendRole, RoleType
from camel.utils import OpenAITokenCounter

from agent_adapter.context.formats import convert
from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.camel import CamelConverter
from tests.context.conversations import invariant_problems
from tests.integrations.compression_oracle import (
    DENIED,
    PAD,
    STEPS,
    check_round_trip,
    check_safe,
    example,
)
from tests.integrations.test_camel_contract import ScriptedModel

pytestmark = pytest.mark.contract
ASSISTANT = OpenAIBackendRole.ASSISTANT
SYSTEM = OpenAIBackendRole.SYSTEM
_clock = iter(range(1, 10_000))


def record(message: Any, role: OpenAIBackendRole, **fields: Any) -> MemoryRecord:
    return MemoryRecord(message=message, role_at_backend=role, timestamp=next(_clock), **fields)


def user(text: str, **fields: Any) -> MemoryRecord:
    return record(BaseMessage.make_user_message("User", text), OpenAIBackendRole.USER, **fields)


def assistant(text: str) -> MemoryRecord:
    return record(BaseMessage.make_assistant_message("Assistant", text), ASSISTANT)


def calls(*ids: str) -> MemoryRecord:
    found = [
        {"id": i, "type": "function", "function": {"name": "lookup", "arguments": '{"q": 1}'}}
        for i in ids
    ]
    message = BaseMessage("Assistant", RoleType.ASSISTANT, {"tool_calls": found}, "")
    return record(message, ASSISTANT)


def result(call_id: str, value: Any, **fields: Any) -> MemoryRecord:
    message = FunctionCallingMessage(
        role_name="Assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="",
        func_name="lookup",
        result=value,
        tool_call_id=call_id,
        **fields,
    )
    return record(message, OpenAIBackendRole.FUNCTION)


def conversation() -> list[MemoryRecord]:
    legacy = FunctionCallingMessage(
        role_name="Assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="",
        func_name="lookup",
        args={"q": "c4"},
        tool_call_id="c4",
    )
    return [
        record(BaseMessage.make_system_message("Be brief."), OpenAIBackendRole.SYSTEM),
        user(f"old question {PAD}", extra_info={"keep": "1"}),
        calls("c1"),
        result("c1", {"rows": PAD}),
        calls("c2", "c3"),
        result("c2", DENIED),
        result("c3", f"two {PAD}", mask_output=True),
        assistant(f"old answer {PAD}"),
        user(f"middle {PAD}"),
        record(legacy, ASSISTANT),
        result("c4", f"more {PAD}"),
        assistant("answer"),
        user("new question"),
    ]


def test_round_trip_gives_equal_values_and_the_same_objects():
    check_round_trip(conversation(), "camel")


def test_calls_results_and_masked_output():
    messages = convert(conversation()).messages
    assert [c.id for m in messages for c in m.tool_calls] == ["c1", "c2", "c3", "c4"]
    assert [m.tool_call_id for m in messages if m.role == "tool"] == ["c1", "c2", "c3", "c4"]
    assert messages[3].content == json.dumps({"rows": PAD}) and messages[6].content == "[MASKED]"
    assert messages[9].tool_calls[0].arguments == {"q": "c4"}


@pytest.mark.parametrize("step", list(STEPS))
def test_no_step_orphans_a_tool_round_or_touches_protected_messages(step):
    items = conversation()
    keep = lambda r: "keep" in r.extra_info  # noqa: E731
    smaller = check_safe(items, STEPS[step](), pinned=keep)
    assert items[1] in smaller and items[5] in smaller
    # CAMEL puts the system record first and sorts the rest by time: that order is safe too.
    by_time = sorted(smaller, key=lambda r: (r.role_at_backend != SYSTEM, r.timestamp))
    before = convert(items, pinned=keep).messages
    assert invariant_problems(before, convert(by_time, pinned=keep).messages) == []


def test_cut_keeps_record_fields():
    items = conversation()
    smaller = check_safe(items, STEPS["cut"]())
    assert smaller[1].uuid == items[1].uuid and smaller[1].extra_info == {"keep": "1"}
    assert "characters cut" in smaller[1].message.content
    assert "characters cut" in smaller[10].message.result and smaller[10].message.content == ""
    assert smaller[6] is items[6]  # masked: the model sees "[MASKED]", nothing to cut


def test_summaries_get_a_timestamp_between_their_neighbours():
    items = conversation()
    smaller = check_safe(items, STEPS["summary"]())
    (summary,) = [r for r in smaller if r.message.content.startswith("summary")]
    assert summary.role_at_backend == OpenAIBackendRole.USER
    assert summary.message.role_name == "User" and summary.agent_id == items[0].agent_id
    assert smaller[0].timestamp < summary.timestamp < smaller[-1].timestamp
    rendered = CamelConverter(items).render(convert(items).messages)
    assert rendered[2].to_openai_message()["tool_calls"][0]["id"] == "c1"
    assert rendered[3].to_openai_message() == {
        "role": "tool",
        "content": json.dumps({"rows": PAD}),
        "tool_call_id": "c1",
    }
    lone = CamelConverter([]).render(convert(items[-1:]).messages)
    assert CamelConverter([]).finish(lone) == lone  # no neighbours: the stamp stays


def test_a_dict_is_not_a_camel_record():
    with pytest.raises(ConfigError, match="not a camel message"):
        convert([{"role": "user", "content": "a"}], format="camel")


def test_docs_context_creator_shortens_what_the_agent_sends():
    creator_class = example("compression_camel")["CompressingContextCreator"]
    model = ScriptedModel()
    sent: list[Any] = []
    run = model._run
    model._run = lambda messages, *args, **kwargs: (sent.append(messages), run(messages))[1]
    memory = ChatHistoryMemory(creator_class(OpenAITokenCounter(model.model_type), 100_000))
    agent = ChatAgent(model=model, memory=memory)
    for _ in range(12):
        agent.update_memory(BaseMessage.make_user_message("User", PAD), OpenAIBackendRole.USER)
    response = agent.step("new question")
    assert response.msgs[0].content == "done"
    assert len(sent[-1]) < 13 and "new question" in str(sent[-1][-1]["content"])
    assert len(agent.memory.retrieve()) == 14  # the memory keeps every record
