"""Every converter round-trips losslessly and keeps tool-call ids, pins and approval markers."""

import copy

import pytest

from agent_adapter.context import CompressionPolicy, RemoveOldMessages
from agent_adapter.context.conversation import Protection, carries_approval_state
from agent_adapter.context.formats import FORMATS, compress_for_sync, convert, detect_format
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from tests.context.format_cases import CASES, DENIED, FORMAT_OF


@pytest.mark.parametrize("case", list(CASES))
def test_convert_then_back_gives_an_equal_value(case):
    items = CASES[case]()
    snapshot = copy.deepcopy(items)
    conversion = convert(items)
    back = conversion.back(conversion.messages)
    assert back == snapshot
    assert all(a is b for a, b in zip(back, items, strict=True))  # the very same objects


@pytest.mark.parametrize("case", list(CASES))
def test_detect_format_names_each_case(case):
    assert detect_format(CASES[case]()) == FORMAT_OF.get(case, case)


def test_tool_call_ids_survive_conversion():
    messages = convert(CASES["openai"]()).messages
    calls = [c.id for m in messages for c in m.tool_calls]
    results = [m.tool_call_id for m in messages if m.role == "tool"]
    assert calls == results == ["c1"]
    responses = convert(CASES["openai_responses"]()).messages
    assert [c.id for m in responses for c in m.tool_calls] == ["c1", "c2"]
    assert [m.tool_call_id for m in responses if m.role == "tool"] == ["c1", "c2"]
    anthropic = convert(CASES["anthropic"]()).messages
    assert [m.tool_call_id for m in anthropic if m.role == "tool"] == ["t1"]


def test_gemini_calls_without_ids_pair_with_their_responses():
    messages = convert(CASES["gemini"]()).messages
    (call,) = [c for m in messages for c in m.tool_calls]
    (result,) = [m for m in messages if m.role == "tool"]
    assert call.id == result.tool_call_id and call.id.startswith("gemini-")
    assert result.name == "lookup"
    assert "thinking" not in messages[1].content  # thought parts are not text


def test_responses_items_form_one_assistant_turn_with_its_reasoning():
    messages = convert(CASES["openai_responses"]()).messages
    assert [m.role for m in messages] == ["system", "user", "assistant", "tool", "tool", "user"]


def test_approval_markers_stay_recognisable():
    messages = convert(CASES["openai_responses"]()).messages
    denied = [m for m in messages if m.role == "tool" and m.content == DENIED]
    assert len(denied) == 1 and carries_approval_state(denied[0])


def test_pinned_predicate_pins_and_keeps_the_native_verbatim():
    items = CASES["openai"]()
    conversion = convert(items, pinned=lambda item: item.get("name") == "ann")
    assert [m.pinned for m in conversion.messages] == [False, True, False, False, False, False]
    assert Protection(conversion.messages).is_protected(conversion.messages[1])
    policy = CompressionPolicy(
        start_when_tokens_over=50, shrink_to_tokens=10, steps=[RemoveOldMessages(keep_last=1)]
    )
    smaller, _ = compress_for_sync(items, policy, pinned=lambda item: item.get("name") == "ann")
    assert items[1] in smaller and smaller[0] is items[0]


def test_the_pinned_marker_is_never_part_of_a_native():
    conversion = convert(CASES["anthropic"](), pinned=lambda item: True)
    assert all("pinned" not in str(item) for item in conversion.back(conversion.messages))


def test_origins_live_outside_the_fields_and_survive_copies():
    for message in convert(CASES["anthropic"]()).messages:
        assert message.raw is None
        plain = Message(**message.__dict__)
        assert message == plain and plain == message
        assert "origin" not in str(message.model_dump())
        copied = message.model_copy(update={"content": "short"})
        assert copied._origin == message._origin  # what a step's copy keeps
        assert copied != message


def test_unknown_and_mixed_inputs_are_config_errors():
    with pytest.raises(ConfigError, match="cannot tell"):
        detect_format([{"role": "user", "content": "a"}, Message(role="user", content="b")])
    with pytest.raises(ConfigError, match="unknown message format"):
        convert([{"role": "user", "content": "a"}], format="xml")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="no conversion"):
        convert([Message(role="user", content="a")])
    with pytest.raises(ConfigError, match="not a anthropic message"):
        convert([{"role": "tool", "content": "a"}], format="anthropic")
    with pytest.raises(ConfigError, match="not a openai message"):
        convert(["text"], format="openai")
    with pytest.raises(ConfigError, match="pinned must be"):
        convert([{"role": "user", "content": "a"}], pinned="yes")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="not a gemini message"):
        convert([{"role": "user"}], format="gemini")
    with pytest.raises(ConfigError, match="not a openai message"):
        convert([{"role": "robot", "content": "a"}], format="openai")


def test_formats_lists_every_format():
    assert FORMATS == ("agent_adapter", "langchain", "openai", "anthropic", "gemini")


def test_plain_dicts_read_as_openai_and_anthropic_blocks_as_anthropic():
    assert detect_format([{"role": "user", "content": "hi"}]) == "openai"
    assert detect_format([]) == "agent_adapter"
    blocks = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t"}]}]
    assert detect_format(blocks) == "anthropic"
