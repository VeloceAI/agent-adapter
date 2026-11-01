"""The llamaindex compression format on real `ChatMessage`s, and the docs example."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)

from agent_adapter.context.formats import compress_for_sync, convert
from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.llamaindex import LlamaIndexConverter
from tests.integrations.compression_oracle import (
    DENIED,
    PAD,
    STEPS,
    check_round_trip,
    check_safe,
    example,
    policy,
)

pytestmark = pytest.mark.contract
LEGACY_CALL = {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}


def tool(call_id: str, text: str) -> ChatMessage:
    return ChatMessage(role="tool", content=text, additional_kwargs={"tool_call_id": call_id})


def call(call_id: str) -> ToolCallBlock:
    return ToolCallBlock(tool_call_id=call_id, tool_name="lookup", tool_kwargs={"q": call_id})


def conversation() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="Be brief."),
        ChatMessage(role="user", content=f"old question {PAD}", additional_kwargs={"keep": 1}),
        ChatMessage(
            role="assistant",
            blocks=[ThinkingBlock(content="hm"), TextBlock(text=f"look {PAD}"), call("c1")],
        ),
        tool("c1", f"found {PAD}"),
        ChatMessage(role="assistant", content="", additional_kwargs={"tool_calls": [LEGACY_CALL]}),
        tool("c2", DENIED),
        ChatMessage(role="assistant", content=f"old answer {PAD}"),
        ChatMessage(role="user", content=f"middle question {PAD}"),
        ChatMessage(role="assistant", blocks=[call("c3")]),
        tool("c3", f"more {PAD}"),
        ChatMessage(role="assistant", content="answer"),
        ChatMessage(role="user", content="new question"),
    ]


def test_round_trip_gives_equal_values_and_the_same_objects():
    check_round_trip(conversation(), "llamaindex")


def test_tool_call_ids_roles_and_legacy_calls_are_read():
    messages = convert(conversation()).messages
    assert [c.id for m in messages for c in m.tool_calls] == ["c1", "c2", "c3"]
    assert [m.tool_call_id for m in messages if m.role == "tool"] == ["c1", "c2", "c3"]
    assert messages[2].content == f"look {PAD}" and messages[2].tool_calls[0].arguments == {
        "q": "c1"
    }
    object_call = SimpleNamespace(id="o1", function=SimpleNamespace(name="f", arguments='{"a": 1}'))
    legacy = ChatMessage(role="assistant", additional_kwargs={"tool_calls": [object_call]})
    (message,) = convert([legacy]).messages
    assert (message.tool_calls[0].id, message.tool_calls[0].arguments) == ("o1", {"a": 1})


@pytest.mark.parametrize("step", list(STEPS))
def test_no_step_orphans_a_tool_round_or_touches_protected_messages(step):
    items = conversation()
    smaller = check_safe(items, STEPS[step](), pinned=lambda m: "keep" in m.additional_kwargs)
    assert items[1] in smaller and items[5] in smaller  # pinned, and the denied round


def test_cut_keeps_blocks_kwargs_and_ids():
    items = conversation()
    smaller = check_safe(items, STEPS["cut"]())
    assert [b.block_type for b in smaller[2].blocks] == ["thinking", "text", "tool_call"]
    assert "characters cut" in smaller[2].blocks[1].text
    assert smaller[3].additional_kwargs == {"tool_call_id": "c1"}
    assert smaller[1].additional_kwargs == {"keep": 1}
    assert smaller[-1] is items[-1]


def test_new_messages_render_as_chat_messages():
    smaller = check_safe(conversation(), STEPS["summary"]())
    assert any(m.role.value == "user" and m.content.startswith("summary") for m in smaller)
    conversion = convert(conversation())
    rendered = LlamaIndexConverter([]).render(conversion.messages)
    assert isinstance(rendered[2].blocks[-1], ToolCallBlock)
    assert rendered[2].blocks[-1].tool_call_id == "c1"
    assert rendered[3].additional_kwargs == {"tool_call_id": "c1"}
    assert rendered[3].content == f"found {PAD}"


def test_a_dict_is_not_a_llamaindex_message():
    with pytest.raises(ConfigError, match="not a llamaindex message"):
        convert([{"role": "user", "content": "a"}], format="llamaindex")


def test_docs_example_sends_a_shorter_chat_message_list():
    sent: list[Any] = []

    class Recorder:
        async def achat(self, messages: list[ChatMessage]) -> ChatResponse:
            sent.append(messages)
            return ChatResponse(message=ChatMessage(role="assistant", content="ok"))

    long = [m for _ in range(8) for m in conversation()[6:8]] + conversation()[-1:]
    response = asyncio.run(example("compression_llamaindex")["ask_llamaindex"](Recorder(), long))
    assert response.message.content == "ok"
    assert all(isinstance(m, ChatMessage) for m in sent[0]) and len(sent[0]) < len(long)
    assert sent[0][-1] is long[-1]
    smaller, _ = compress_for_sync(long, policy(STEPS["drop_old"]()))
    assert len(smaller) < len(long)
