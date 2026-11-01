"""LangChain `BaseMessage` lists (LangChain, LangGraph); `langchain_core` is imported lazily."""

from __future__ import annotations

from typing import Any

from agent_adapter.context.formats._shared import (
    Converter,
    Group,
    Present,
    arguments_of,
    not_a_message,
    text_of,
    with_text,
)
from agent_adapter.core.errors import MissingDependency
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall

NAME = "langchain"
TEXT = frozenset({"text"})
_ROLES = {
    "SystemMessage": "system",
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "ToolMessage": "tool",
    "FunctionMessage": "tool",
}
_CHAT_ROLES = {"system": "system", "user": "user", "human": "user", "assistant": "assistant"}
_CHAT_ROLES |= {"ai": "assistant", "tool": "tool"}
_SYNTHETIC = "langchain-"  # prefix of call ids made up for tool calls without one


def is_langchain_message(item: Any) -> bool:
    """True for a `langchain_core` message, found by class, without importing LangChain."""
    return any(
        cls.__name__ == "BaseMessage" and cls.__module__.startswith("langchain_core.")
        for cls in type(item).__mro__
    )


def _role(item: Any) -> str | None:
    for cls in type(item).__mro__:
        if cls.__name__ in _ROLES:
            return _ROLES[cls.__name__]
        if cls.__name__ == "ChatMessage":
            return _CHAT_ROLES.get(str(getattr(item, "role", "")).lower())
    return None


class LangChainConverter(Converter):
    """System, human, AI (with tool calls) and tool messages; others are rejected."""

    name = NAME

    def groups(self) -> list[Group]:
        groups: list[Group] = []
        for index, item in enumerate(self.items):
            role = _role(item) if is_langchain_message(item) else None
            if role is None:
                raise not_a_message(NAME, index, item)
            text = text_of(item.content, TEXT)
            if role == "assistant":
                calls = tuple(
                    ToolCall(
                        id=str(call.get("id") or f"{_SYNTHETIC}{index}-{n}"),
                        name=str(call.get("name", "")),
                        arguments=arguments_of(call.get("args")),
                    )
                    for n, call in enumerate(getattr(item, "tool_calls", None) or ())
                )
                message = Message(role="assistant", content=text, tool_calls=calls)
            elif role == "tool":
                message = Message(
                    role="tool",
                    tool_call_id=str(getattr(item, "tool_call_id", "") or ""),
                    name=str(getattr(item, "name", "") or ""),
                    content=text,
                )
            else:
                message = Message(role=role, content=text)
            groups.append(Group((index,), (message,)))
        return groups

    def rebuild(self, group: Group, present: list[Present]) -> list[Any]:
        item = self.items[group.natives[0]]
        part = present[0]
        if part.unchanged:
            return [item]
        return [
            item.model_copy(update={"content": with_text(item.content, part.message.content, TEXT)})
        ]

    def render(self, messages: list[Message]) -> list[Any]:
        classes = _classes()
        out: list[Any] = []
        for message in messages:
            if message.role == "tool":
                out.append(
                    classes["tool"](
                        content=message.content,
                        tool_call_id=message.tool_call_id,
                        name=message.name or None,
                    )
                )
            elif message.role == "assistant":
                calls = [
                    {"name": c.name, "args": dict(c.arguments), "id": c.id, "type": "tool_call"}
                    for c in message.tool_calls
                ]
                out.append(classes["assistant"](content=message.content, tool_calls=calls))
            else:
                out.append(classes[message.role](content=message.content))
        return out


def _classes() -> dict[str, Any]:
    """The langchain_core message classes by role.

    Raises:
        MissingDependency: langchain-core is not installed.
    """
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except ImportError as exc:
        raise MissingDependency(
            'the langchain format needs langchain-core: pip install "agent-adapter[langchain]"'
        ) from exc
    return {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
        "tool": ToolMessage,
    }
