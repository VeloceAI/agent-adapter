"""The semantic_kernel compression format on real `ChatMessageContent`s, and the docs example."""

import asyncio
from typing import Any

import pytest

pytest.importorskip("semantic_kernel")

from semantic_kernel.contents import (
    AuthorRole,
    ChatHistory,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
    ImageContent,
    TextContent,
)

from agent_adapter.context.formats import convert
from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.semantic_kernel import SemanticKernelConverter
from tests.integrations.compression_oracle import (
    DENIED,
    PAD,
    STEPS,
    check_round_trip,
    check_safe,
    example,
)

pytestmark = pytest.mark.contract
IMAGE = ImageContent(uri="https://example.com/a.png")


def call(call_id: str) -> FunctionCallContent:
    return FunctionCallContent(id=call_id, name="tools-lookup", arguments={"q": call_id})


def result(call_id: str, value: Any) -> FunctionResultContent:
    return FunctionResultContent(id=call_id, name="tools-lookup", result=value)


def message(role: AuthorRole, *items: Any, **fields: Any) -> ChatMessageContent:
    return ChatMessageContent(role=role, items=list(items), **fields)


def conversation() -> list[ChatMessageContent]:
    return [
        message(AuthorRole.SYSTEM, TextContent(text="Be brief.")),
        message(AuthorRole.USER, TextContent(text=f"old question {PAD}"), name="ann"),
        message(AuthorRole.ASSISTANT, TextContent(text=f"look {PAD}"), call("c1")),
        message(AuthorRole.TOOL, result("c1", f"found {PAD}")),
        message(AuthorRole.ASSISTANT, call("c2"), call("c3")),
        message(AuthorRole.TOOL, result("c2", DENIED), IMAGE, result("c3", f"two {PAD}")),
        message(AuthorRole.ASSISTANT, TextContent(text=f"old answer {PAD}")),
        message(AuthorRole.USER, TextContent(text=f"middle {PAD}"), IMAGE),
        message(AuthorRole.ASSISTANT, call("c4")),
        message(AuthorRole.TOOL, result("c4", {"rows": PAD})),
        message(AuthorRole.ASSISTANT, TextContent(text="answer")),
        message(AuthorRole.USER, TextContent(text="new question")),
    ]


def test_round_trip_gives_equal_values_and_the_same_objects():
    check_round_trip(conversation(), "semantic_kernel")


def test_each_function_result_is_a_tool_message_with_its_call_id():
    messages = convert(conversation()).messages
    assert [c.id for m in messages for c in m.tool_calls] == ["c1", "c2", "c3", "c4"]
    assert [m.tool_call_id for m in messages if m.role == "tool"] == ["c1", "c2", "c3", "c4"]
    assert messages[2].tool_calls[0].name == "tools-lookup"
    assert messages[2].tool_calls[0].arguments == {"q": "c1"}
    assert messages[-3].content == '{"rows": "' + PAD + '"}'


@pytest.mark.parametrize("step", list(STEPS))
def test_no_step_orphans_a_tool_round_or_touches_protected_messages(step):
    items = conversation()
    smaller = check_safe(items, STEPS[step](), pinned=lambda m: m.name == "ann")
    assert items[1] in smaller and items[4] in smaller and items[5] in smaller


def test_cut_keeps_other_items_and_fields():
    items = conversation()
    smaller = check_safe(items, STEPS["cut"]())
    assert smaller[1].name == "ann" and "characters cut" in smaller[1].content
    assert [i.content_type for i in smaller[5].items] == [
        "function_result",
        "image",
        "function_result",
    ]
    assert smaller[3].items[0].id == "c1" and "characters cut" in smaller[3].items[0].result
    assert smaller[7].items[1] == IMAGE


def test_new_messages_render_as_chat_message_contents():
    smaller = check_safe(conversation(), STEPS["summary"]())
    assert any(m.role == AuthorRole.USER and m.content.startswith("summary") for m in smaller)
    rendered = SemanticKernelConverter([]).render(convert(conversation()).messages)
    assert [i.content_type for i in rendered[2].items] == ["text", "function_call"]
    assert rendered[2].items[1].id == "c1" and rendered[2].items[1].function_name == "lookup"
    assert rendered[3].role == AuthorRole.TOOL and rendered[3].items[0].id == "c1"


def test_a_dict_is_not_a_semantic_kernel_message():
    with pytest.raises(ConfigError, match="not a semantic_kernel message"):
        convert([{"role": "user", "content": "a"}], format="semantic_kernel")


def test_docs_example_sends_a_shorter_history_and_keeps_the_original():
    seen: list[ChatHistory] = []

    class Service:
        async def get_chat_message_contents(self, chat_history: ChatHistory, settings: Any) -> Any:
            seen.append(chat_history)
            return [message(AuthorRole.ASSISTANT, TextContent(text="ok"))]

    history = ChatHistory(messages=[m for _ in range(8) for m in conversation()[6:8]])
    history.add_user_message("new question")
    ask = example("compression_semantic_kernel")["ask_semantic_kernel"]
    (reply,) = asyncio.run(ask(Service(), history, None))
    assert reply.content == "ok" and len(history.messages) == 17
    assert isinstance(seen[0], ChatHistory) and len(seen[0].messages) < 17
    assert seen[0].messages[-1] is history.messages[-1]
