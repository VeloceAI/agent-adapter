"""smolagents `ChatMessage` lists as a compression format, and a model wrapper.

Importing this module registers the "smolagents" format. smolagents writes tool rounds as
text: a "tool-call" message, then a "tool-response" one. Each pair becomes one tool round
with a made-up call id, so a step keeps or drops the pair together.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from agent_adapter.context.formats import Pinned, register_format
from agent_adapter.context.formats._shared import not_a_message, text_of, with_text
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.core.errors import MissingDependency
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations._compress_hooks import Kind, Shrink, wrap_model
from agent_adapter.integrations._compress_parts import PartsConverter, Piece

NAME = "smolagents"
TEXT = frozenset({"text"})
_PLAIN = {"system": "system", "user": "user", "assistant": "assistant"}
_CALL, _RESPONSE = "tool-call", "tool-response"
_SYNTHETIC = "smolagents-"  # prefix of the made-up call ids
_TOOL = "tools"  # name of the made-up call: smolagents does not say which tools ran
_METHODS: dict[str, Kind] = {"generate": "call", "generate_stream": "iter"}


def is_smolagents_message(item: Any) -> bool:
    """True for a `smolagents.models.ChatMessage`, found by class, without importing it."""
    return any(
        cls.__name__ == "ChatMessage" and cls.__module__.startswith("smolagents.")
        for cls in type(item).__mro__
    )


def _role(item: Any) -> str:
    role = getattr(item, "role", "")
    return str(getattr(role, "value", role))


class SmolagentsConverter(PartsConverter):
    """System, user and assistant messages; tool-call/tool-response pairs are tool rounds.

    An observation image (a user message between a call and its response) rides with the
    response. A response with no call before it is a user message, as smolagents sends it.
    """

    name = NAME

    def __init__(self, items: Any) -> None:
        super().__init__(items)
        self._open: str | None = None  # id of a tool-call still waiting for its response

    def pieces(self, index: int, item: Any) -> list[Piece]:
        role = _role(item) if is_smolagents_message(item) else ""
        if role not in (*_PLAIN, _CALL, _RESPONSE):
            raise not_a_message(NAME, index, item)
        text = text_of(item.content, TEXT)
        if role == _CALL:
            self._open = f"{_SYNTHETIC}{index}"
            call = ToolCall(id=self._open, name=_TOOL, arguments={})
            return [(Message(role="assistant", content=text, tool_calls=(call,)), None)]
        if role == _RESPONSE and self._open is not None:
            message = Message(role="tool", tool_call_id=self._open, name=_TOOL, content=text)
            self._open = None
            return [(message, None)]
        if role == "user" and self._open is not None and self._answered_after(index):
            return []  # an observation image: rides with the response after it
        self._open = None
        return [(Message(role=_PLAIN.get(role, "user"), content=text), None)]

    def _answered_after(self, index: int) -> bool:
        """True when only user messages stand between `index` and a tool-response."""
        for item in self.items[index + 1 :]:
            role = _role(item)
            if role != "user":
                return role == _RESPONSE
        return False

    def retext(self, item: Any, message: Message) -> Any:
        return dataclasses.replace(item, content=with_text(item.content, message.content, TEXT))

    def render(self, messages: list[Message]) -> list[Any]:
        chat_message, role = _classes()
        out: list[Any] = []
        for message in messages:
            kind = {"tool": role.TOOL_RESPONSE, "system": role.SYSTEM}.get(message.role)
            if message.role == "assistant":
                kind = role.TOOL_CALL if message.tool_calls else role.ASSISTANT
            content = [{"type": "text", "text": message.content}]
            out.append(chat_message(role=kind or role.USER, content=content))
        return out


def _classes() -> tuple[Any, Any]:
    """`smolagents.models.ChatMessage` and `MessageRole`.

    Raises:
        MissingDependency: smolagents is not installed.
    """
    try:
        from smolagents.models import ChatMessage, MessageRole
    except ImportError as exc:
        raise MissingDependency(
            'the smolagents format needs smolagents: pip install "agent-adapter[smolagents]"'
        ) from exc
    return ChatMessage, MessageRole


def smolagents_compressed_model(
    model: Any,
    policy: CompressionPolicy,
    *,
    pinned: Pinned | None = None,
    model_max_tokens: int | None = None,
) -> Any:
    """A copy of a smolagents model whose `generate` and `generate_stream` get fewer messages.

    Use it as `CodeAgent(model=smolagents_compressed_model(model, policy), tools=...)`. The
    agent's memory keeps every step.

    Raises:
        ConfigError: `model` has no `generate`, `policy` is not a `CompressionPolicy`, or
            `pinned` is not callable.
    """
    return wrap_model(model, Shrink(policy, NAME, pinned, model_max_tokens), _METHODS)


register_format(NAME, SmolagentsConverter, detect=is_smolagents_message)
