"""The "strands" format (Bedrock Converse dicts, no SDK needed) and the Strands middleware."""

import inspect
import json
from dataclasses import dataclass, replace

import pytest

from agent_adapter.context.formats import compress_for_sync, convert, detect_format
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations.strands import strands_compress_middleware
from agent_adapter.integrations.strands_compress import StrandsConverter, is_strands_message
from tests.integrations.compression_support import (
    CUT,
    DENIED,
    DROP_OLD,
    EXPECTED,
    PAD,
    assert_cut_keeps_types,
    assert_drop_keeps_whole_rounds,
    assert_round_trip,
    assert_summary_is_native,
    render_read_back,
)


def history() -> list[dict]:
    """An old question, a reasoning turn with two tool calls, their results, a new question."""
    return [
        {"role": "user", "content": [{"text": f"old question {PAD}"}], "tracking_id": "t1"},
        {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"reasoningText": {"text": "think", "signature": "s"}}},
                {"text": f"let me look {PAD}"},
                {"toolUse": {"toolUseId": "u1", "name": "lookup", "input": {"q": "a"}}},
                {"toolUse": {"toolUseId": "u2", "name": "lookup", "input": {"q": "b"}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "u1",
                        "status": "success",
                        "content": [{"json": {"value": PAD}}, {"image": {"format": "png"}}],
                    }
                },
                {"toolResult": {"toolUseId": "u2", "status": "error", "content": [{"text": "no"}]}},
            ],
        },
        {"role": "assistant", "content": [{"text": f"answer {PAD}"}], "metadata": {"custom": {}}},
        {
            "role": "user",
            "content": [{"text": "new question"}, {"cachePoint": {"type": "default"}}],
        },
    ]


def test_round_trip_keeps_ids_roles_and_every_block() -> None:
    messages = assert_round_trip(history(), "strands")
    assert [m.role for m in messages] == ["user", "assistant", "tool", "tool", "assistant", "user"]
    assert [c.id for c in messages[1].tool_calls] == ["u1", "u2"]
    assert [m.tool_call_id for m in messages[2:4]] == ["u1", "u2"]
    assert json.loads(messages[2].content) == {"value": PAD} and "think" not in messages[1].content


def test_dropping_old_turns_never_orphans_a_tool_round() -> None:
    smaller = assert_drop_keeps_whole_rounds(history())
    assert smaller[-1]["content"][1] == {"cachePoint": {"type": "default"}}


def test_cut_keeps_reasoning_tool_use_status_media_and_extra_keys() -> None:
    items = history()
    smaller = assert_cut_keeps_types(items)
    assert (
        smaller[0]["tracking_id"] == "t1" and "characters cut" in smaller[0]["content"][0]["text"]
    )
    blocks = smaller[1]["content"]
    assert blocks[0] == items[1]["content"][0] and blocks[2:] == items[1]["content"][2:]
    first, second = smaller[2]["content"]
    assert first["toolResult"]["toolUseId"] == "u1" and first["toolResult"]["status"] == "success"
    assert first["toolResult"]["content"][1] == {"image": {"format": "png"}}
    assert second is items[2]["content"][1]  # the short result is kept as it was
    assert smaller[3]["metadata"] == {"custom": {}}


def test_summary_is_a_user_turn_of_text_blocks() -> None:
    smaller = assert_summary_is_native(history(), dict)
    summary = next(s for s in smaller if "summary of earlier turns" in str(s))
    assert summary == {"role": "user", "content": [{"text": "summary of earlier turns"}]}


def test_render_writes_tool_calls_and_results_as_blocks() -> None:
    call = ToolCall(id="u9", name="f", arguments={"a": 1})
    rendered = StrandsConverter([]).render(
        [
            Message(role="system", content="s"),
            Message(role="assistant", content="", tool_calls=(call,)),
            Message(role="tool", tool_call_id="u9", content="r"),
        ]
    )
    assert rendered == [
        {"role": "user", "content": [{"text": "s"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "u9", "name": "f", "input": {"a": 1}}}],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "u9", "status": "success", "content": [{"text": "r"}]}}
            ],
        },
    ]


def test_an_approval_denial_stays_verbatim() -> None:
    items = history()
    items[2]["content"][1]["toolResult"]["content"] = [{"text": DENIED}]
    conversion = convert(items)
    assert conversion.messages[3].content == DENIED
    smaller, _ = compress_for_sync(items, CUT)
    assert smaller[2]["content"][1] is items[2]["content"][1]


@pytest.mark.parametrize(
    "item",
    [
        {"role": "user", "content": "plain text"},
        {"role": "user", "content": [{"type": "text", "text": "openai"}]},
        {"role": "system", "content": [{"text": "x"}]},
        {"role": "user", "content": []},
        "not a dict",
    ],
)
def test_other_dicts_are_not_strands_messages(item: object) -> None:
    assert not is_strands_message(item)


def test_plain_chat_dicts_still_detect_as_openai() -> None:
    assert detect_format([{"role": "user", "content": "hi"}]) == "openai"


def test_a_bad_item_names_the_format() -> None:
    with pytest.raises(ConfigError, match="not a strands message"):
        convert([{"role": "user", "content": "text"}], "strands")


@dataclass
class Context:
    """The part of `strands.middleware.InvokeModelContext` the middleware uses."""

    messages: list[dict]

    def replace(self, **changes: object) -> "Context":
        return replace(self, **changes)


async def test_middleware_shortens_the_call_and_keeps_the_agent_history() -> None:
    items = history()
    context = Context(messages=items)
    sent = await strands_compress_middleware(DROP_OLD)(context)
    assert len(sent.messages) < len(items) and context.messages is items and len(items) == 5


@pytest.mark.contract
def test_the_real_invoke_model_context_has_what_the_middleware_uses() -> None:
    middleware = pytest.importorskip("strands.middleware")
    context = middleware.InvokeModelContext
    assert "messages" in context.__dataclass_fields__
    assert "messages" in inspect.signature(context.replace).parameters
    assert hasattr(middleware.InvokeModelStage, "Input")


def test_new_messages_render_as_converse_dicts_with_system_text_as_a_user_turn() -> None:
    assert render_read_back(StrandsConverter) == [("user", "rules", [], ""), *EXPECTED[1:]]
