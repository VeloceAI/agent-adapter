"""compress_for: the result is in the input's format, keeps natives, and never orphans a call."""

import asyncio

import pytest

from agent_adapter.context import CompressionPolicy, CutLongMessages, MyOwnStep, RemoveOldMessages
from agent_adapter.context.formats import compress_for, compress_for_sync, convert
from agent_adapter.core.messages import Message
from tests.context.format_cases import CASES, PAD

DROP_OLD = CompressionPolicy(
    start_when_tokens_over=50,
    shrink_to_tokens=10,
    steps=[RemoveOldMessages(keep_first=0, keep_last=1)],
)
CUT = CompressionPolicy(
    start_when_tokens_over=50,
    shrink_to_tokens=10,
    steps=[CutLongMessages(max_tokens_per_message=20)],
)


def _calls_and_results(items, case):
    messages = convert(items).messages
    calls = {c.id for m in messages for c in m.tool_calls}
    results = {m.tool_call_id for m in messages if m.role == "tool"}
    return calls, results


@pytest.mark.parametrize("case", list(CASES))
def test_dropping_old_turns_keeps_whole_rounds_and_native_objects(case):
    items = CASES[case]()
    smaller, report = compress_for_sync(items, DROP_OLD)
    assert report.tokens_after < report.tokens_before
    assert len(smaller) < len(items)
    assert all(any(s is i for i in items) for s in smaller)  # whole groups: the same objects
    calls, results = _calls_and_results(smaller, case)
    assert calls == results
    assert smaller[-1] is items[-1]


@pytest.mark.parametrize("case", list(CASES))
def test_cut_text_keeps_every_other_field(case):
    items = CASES[case]()
    smaller, report = compress_for_sync(items, CUT)
    assert "cut_long_messages" in report.steps_used
    assert type(smaller) is list and len(smaller) == len(items)
    assert all(isinstance(s, dict) for s in smaller)
    assert smaller[-1] is items[-1]
    assert sum(len(str(s)) for s in smaller) < sum(len(str(i)) for i in items)


def test_cut_openai_chat_keeps_name_and_tool_ids():
    items = CASES["openai"]()
    smaller, _ = compress_for_sync(items, CUT)
    assert smaller[1]["name"] == "ann" and "characters cut" in smaller[1]["content"]
    assert smaller[3] is items[3]  # the call itself has no long text
    assert smaller[4]["tool_call_id"] == "c1" and smaller[4]["content"] != items[4]["content"]


def test_cut_anthropic_keeps_thinking_tool_use_and_cache_control():
    items = CASES["anthropic"]()
    smaller, _ = compress_for_sync(items, CUT)
    blocks = smaller[1]["content"]
    assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use"]
    assert blocks[0] == items[1]["content"][0] and blocks[2] == items[1]["content"][2]
    result, text = smaller[2]["content"]
    assert result["tool_use_id"] == "t1" and "characters cut" in result["content"]
    assert text is items[2]["content"][1]  # unchanged part kept as it was


def test_cut_gemini_keeps_thought_and_call_and_makes_the_response_an_object():
    items = CASES["gemini"]()
    smaller, _ = compress_for_sync(items, CUT)
    parts = smaller[1]["parts"]
    assert parts[0] == {"text": "thinking", "thought": True}
    assert parts[2] == items[1]["parts"][2]
    response = smaller[2]["parts"][0]["functionResponse"]
    assert response["name"] == "lookup" and isinstance(response["response"], dict)


def test_cut_responses_keeps_types_and_never_touches_a_round_with_a_denial():
    items = CASES["openai_responses"]()
    smaller, _ = compress_for_sync(items, CUT)
    assert smaller[1]["type"] == "message" and smaller[1]["role"] == "user"
    assert "characters cut" in smaller[1]["content"][0]["text"]
    assert smaller[1]["content"][0]["type"] == "input_text"
    assert all(a is b for a, b in zip(smaller[2:], items[2:], strict=True))


def test_cut_responses_assistant_text_keeps_reasoning_and_calls():
    items = CASES["openai_responses"]()
    items[7] = {"type": "function_call_output", "call_id": "c2", "output": "fine"}
    items.append({"type": "message", "role": "assistant", "content": "ok"})
    items.append({"role": "user", "content": "newest"})
    smaller, _ = compress_for_sync(items, CUT)
    assert smaller[2] is items[2] and smaller[4] is items[4] and smaller[5] is items[5]
    assert "characters cut" in smaller[3]["content"][0]["text"]
    assert smaller[3]["content"][0]["type"] == "output_text"
    assert smaller[6]["call_id"] == "c1" and "characters cut" in smaller[6]["output"]


def test_a_summary_step_renders_new_messages_in_the_input_format():
    def summarise(messages):
        kept = [m for m in messages if m.role == "system"]
        return [*kept, Message(role="user", content="summary of earlier turns"), messages[-1]]

    policy = CompressionPolicy(
        start_when_tokens_over=50, shrink_to_tokens=10, steps=[MyOwnStep(summarise)]
    )
    for case, fresh in (
        ("openai", {"role": "user", "content": "summary of earlier turns"}),
        ("anthropic", {"role": "user", "content": "summary of earlier turns"}),
        ("gemini", {"role": "user", "parts": [{"text": "summary of earlier turns"}]}),
    ):
        items = CASES[case]()
        smaller, _ = compress_for_sync(items, policy)
        assert fresh in smaller, case
        assert smaller[-1] is items[-1]


def test_async_twin_and_tuple_input():
    items = tuple(CASES["openai"]())
    smaller, report = asyncio.run(compress_for(items, DROP_OLD))
    assert isinstance(smaller, list) and report.messages_after < report.messages_before


def test_under_the_threshold_nothing_changes():
    items = CASES["anthropic"]()
    policy = CompressionPolicy(start_when_tokens_over=10_000, shrink_to_tokens=5_000)
    smaller, report = compress_for_sync(items, policy)
    assert report.skipped_reason == "under_threshold"
    assert all(a is b for a, b in zip(smaller, items, strict=True))


def test_message_lists_compress_directly_and_pins_map_back():
    items = [
        Message(role="system", content="s"),
        Message(role="user", content=f"keep {PAD}"),
        Message(role="assistant", content=f"old {PAD}"),
        Message(role="user", content="new"),
    ]
    smaller, _ = compress_for_sync(
        items, DROP_OLD, pinned=lambda message: message.content.startswith("keep")
    )
    assert smaller[1] is items[1] and not smaller[1].pinned
    assert items[2] not in smaller
    plain, _ = compress_for_sync(items, DROP_OLD)
    assert plain[0] is items[0]
    with pytest.raises(Exception, match="pinned must be"):
        compress_for_sync(items, DROP_OLD, pinned=3)  # type: ignore[arg-type]
