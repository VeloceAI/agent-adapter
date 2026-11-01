"""Strands Agents messages (Bedrock Converse dicts) as a compression format.

Importing this module registers the "strands" format. Strands messages are plain dicts,
so nothing is imported from `strands`.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from agent_adapter.context.formats import Pinned, register_format
from agent_adapter.context.formats._shared import arguments_of, not_a_message
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations._compress_hooks import Shrink
from agent_adapter.integrations._compress_parts import PartsConverter, Piece

NAME = "strands"
_ROLES = frozenset({"user", "assistant"})
_BLOCKS = frozenset(
    {
        "text",
        "toolUse",
        "toolResult",
        "reasoningContent",
        "image",
        "document",
        "video",
        "audio",
        "cachePoint",
        "guardContent",
        "citationsContent",
    }
)
_TEXTUAL = frozenset({"text", "json"})  # tool result items whose text a step may change


def is_strands_message(item: Any) -> bool:
    """True for a Strands message: blocks keyed by kind (one known), none with a "type" field."""
    if not isinstance(item, Mapping) or item.get("role") not in _ROLES:
        return False
    content = item.get("content")
    if not isinstance(content, list) or not content:
        return False
    if not all(isinstance(b, Mapping) and "type" not in b for b in content):
        return False
    return any(_BLOCKS & b.keys() for b in content)  # newer block kinds may sit beside them


def _blocks_text(blocks: Sequence[Any]) -> str:
    return "\n".join(b["text"] for b in blocks if isinstance(b.get("text"), str) and b["text"])


def _with_text(blocks: Sequence[Any], text: str) -> list[Any]:
    """`blocks` with the first text block holding `text`; later text blocks go."""
    out: list[Any] = []
    placed = False
    for block in blocks:
        if not isinstance(block.get("text"), str):
            out.append(block)
        elif not placed:
            out.append({**block, "text": text})
            placed = True
    if not placed:
        out.insert(0, {"text": text})
    return out


def _result_text(items: Sequence[Any]) -> str:
    texts = []
    for item in items:
        if isinstance(item.get("text"), str):
            texts.append(item["text"])
        elif "json" in item:
            texts.append(json.dumps(item["json"], ensure_ascii=False, sort_keys=True, default=str))
    return "\n".join(t for t in texts if t)


class StrandsConverter(PartsConverter):
    """Assistant turns with toolUse blocks; a user turn's toolResult blocks are tool messages."""

    name = NAME

    def pieces(self, index: int, item: Any) -> list[Piece]:
        if not isinstance(item, Mapping) or item.get("role") not in _ROLES:
            raise not_a_message(NAME, index, item)
        blocks = item.get("content")
        if not isinstance(blocks, list) or not all(isinstance(b, Mapping) for b in blocks):
            raise not_a_message(NAME, index, item)
        if item["role"] == "assistant":
            calls = tuple(
                ToolCall(
                    id=str(use.get("toolUseId", "")),
                    name=str(use.get("name", "")),
                    arguments=arguments_of(use.get("input")),
                )
                for use in (b["toolUse"] for b in blocks if isinstance(b.get("toolUse"), Mapping))
            )
            text = _blocks_text(blocks)
            return [(Message(role="assistant", content=text, tool_calls=calls), None)]
        pieces: list[Piece] = []
        rest: list[int] = []
        for position, block in enumerate(blocks):
            result = block.get("toolResult")
            if isinstance(result, Mapping):
                message = Message(
                    role="tool",
                    tool_call_id=str(result.get("toolUseId", "")),
                    content=_result_text(result.get("content") or ()),
                )
                pieces.append((message, (position,)))
            else:
                rest.append(position)
        if rest or not pieces:
            text = _blocks_text([blocks[p] for p in rest])
            pieces.append((Message(role="user", content=text), tuple(rest)))
        return pieces

    def retext(self, item: Any, message: Message) -> Any:
        return {**item, "content": _with_text(item["content"], message.content)}

    def parts(self, item: Any) -> Sequence[Any]:
        return list(item["content"])

    def with_parts(self, item: Any, parts: list[Any]) -> Any:
        return {**item, "content": parts}

    def retext_parts(self, item: Any, parts: list[Any], message: Message) -> list[Any]:
        if message.role != "tool":
            return _with_text(parts, message.content)
        result = parts[0]["toolResult"]
        others = [i for i in result.get("content") or () if not _TEXTUAL & i.keys()]
        content = [{"text": message.content}, *others]
        return [{**parts[0], "toolResult": {**result, "content": content}}]

    def render(self, messages: list[Message]) -> list[Any]:
        out: list[Any] = []
        for message in messages:
            if message.role == "tool":
                result = {"toolUseId": message.tool_call_id, "status": "success"}
                block = {"toolResult": {**result, "content": [{"text": message.content}]}}
                out.append({"role": "user", "content": [block]})
                continue
            blocks: list[dict[str, Any]] = [{"text": message.content}] if message.content else []
            blocks.extend(
                {"toolUse": {"toolUseId": c.id, "name": c.name, "input": dict(c.arguments)}}
                for c in message.tool_calls
            )
            role = "assistant" if message.role == "assistant" else "user"  # no system turns
            out.append({"role": role, "content": blocks or [{"text": ""}]})
        return out


def strands_compress_middleware(
    policy: CompressionPolicy,
    *,
    pinned: Pinned | None = None,
    model_max_tokens: int | None = None,
) -> Callable[[Any], Awaitable[Any]]:
    """An Input handler: `agent.add_middleware(InvokeModelStage.Input, handler)`.

    It shortens what each model call sends; `agent.messages` keeps the full history.

    Raises:
        ConfigError: `policy` is not a `CompressionPolicy` or `pinned` is not callable.
    """
    shrink = Shrink(policy, NAME, pinned, model_max_tokens)

    async def compress_model_input(context: Any) -> Any:
        return context.replace(messages=await shrink(context.messages))

    return compress_model_input


register_format(NAME, StrandsConverter, detect=is_strands_message)
