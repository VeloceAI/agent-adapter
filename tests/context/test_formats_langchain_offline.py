"""The langchain format without langchain-core: detection by class, and a clear install hint."""

import importlib.util

import pytest

from agent_adapter.context import CompressionPolicy, MyOwnStep
from agent_adapter.context.formats import compress_for_sync, convert, detect_format
from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.core.messages import Message


class BaseMessage:
    def __init__(self, content, **fields):
        self.content = content
        self.__dict__.update(fields)

    def model_copy(self, update):
        return type(self)(**{**self.__dict__, **update})


BaseMessage.__module__ = "langchain_core.messages.base"
HumanMessage = type("HumanMessage", (BaseMessage,), {})
SystemMessage = type("SystemMessage", (BaseMessage,), {})
AIMessage = type("AIMessage", (BaseMessage,), {})
ToolMessage = type("ToolMessage", (BaseMessage,), {})
ChatMessage = type("ChatMessage", (BaseMessage,), {})
RemoveMessage = type("RemoveMessage", (BaseMessage,), {})


def _history():
    return [
        SystemMessage("s"),
        ChatMessage("x" * 400, role="human"),
        AIMessage("", tool_calls=[{"name": "f", "args": {}, "id": None}]),
        ToolMessage("r", tool_call_id="langchain-2-0", name="f"),
        HumanMessage([{"type": "text", "text": "new"}, "and more"]),
    ]


def test_classes_are_recognised_by_name_and_module():
    items = _history()
    assert detect_format(items) == "langchain"
    messages = convert(items).messages
    assert [m.role for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert messages[2].tool_calls[0].id == "langchain-2-0"  # made up for a call without one
    assert messages[4].content == "new\nand more"


def test_other_langchain_messages_are_rejected():
    with pytest.raises(ConfigError, match="not a langchain message"):
        convert([RemoveMessage("", id="x")], format="langchain")
    with pytest.raises(ConfigError, match="not a langchain message"):
        convert([ChatMessage("x", role="robot")], format="langchain")


@pytest.mark.skipif(importlib.util.find_spec("langchain_core") is not None, reason="installed")
def test_new_messages_need_langchain_core():
    def add(messages):
        return [messages[0], Message(role="user", content="summary"), messages[-1]]

    policy = CompressionPolicy(start_when_tokens_over=5, shrink_to_tokens=1, steps=[MyOwnStep(add)])
    with pytest.raises(MissingDependency, match="langchain-core"):
        compress_for_sync(_history(), policy)


def test_cut_text_goes_through_model_copy():
    from agent_adapter.context import CutLongMessages

    policy = CompressionPolicy(
        start_when_tokens_over=5,
        shrink_to_tokens=1,
        steps=[CutLongMessages(max_tokens_per_message=5)],
    )
    items = _history()
    smaller, _ = compress_for_sync(items, policy)
    assert type(smaller[1]) is ChatMessage and smaller[1].role == "human"
    assert "characters cut" in smaller[1].content
    assert smaller[0] is items[0] and smaller[-1] is items[-1]
