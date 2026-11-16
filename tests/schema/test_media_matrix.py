"""Every media type x every provider rewrite x with and without structured data: none lost.

Each media item must reach the provider payload either as a block or as a note naming it,
never both and never neither. The expected "sent" set per provider pins the rules.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from agent_adapter.core.media import Media
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ToolCall, ToolResult
from agent_adapter.exec._loop_steps import _tool_parts
from agent_adapter.integrations.mcp.annotations import result_blocks
from agent_adapter.models._anthropic_convert import build_anthropic_request
from agent_adapter.models._gemini_convert import build_gemini_request
from agent_adapter.models._openai_chat import build_chat_request
from agent_adapter.models._openai_responses import build_responses_request

BIG = b"\x89PNG\r\n\x1a\n" + b"\0" * (6 * 1024 * 1024)
MEDIA: dict[str, Media] = {
    "png": Media(data=b"\x89PNG\r\n\x1a\nPNGDATA", name="shot.png"),
    "jpeg": Media(data=b"\xff\xd8\xffJPEGDATA", name="photo.jpg"),
    "gif": Media(data=b"GIF89aGIFDATA", name="anim.gif"),
    "webp": Media(data=b"RIFF\0\0\0\0WEBPVP8 WEBPDATA", name="pic.webp"),
    "svg": Media(data=b"<svg xmlns='http://www.w3.org/2000/svg'/>", name="logo.svg"),
    "pdf": Media(data=b"%PDF-1.7 PDFDATA", name="report.pdf"),
    "text": Media(data=b"plain notes", mime_type="text/plain", name="notes.txt"),
    "wav": Media(data=b"RIFF\0\0\0\0WAVEfmt WAVDATA", name="clip.wav"),
    "mp3": Media(data=b"ID3MP3DATA", name="song.mp3"),
    "mp4": Media(data=b"\0\0\0\x18ftypisomMP4DATA", name="movie.mp4"),
    "zip": Media(data=b"PK\x03\x04ZIPDATA", name="bundle.zip"),
    "image_url": Media(url="https://example.com/cat.png"),
    "pdf_url": Media(url="https://example.com/paper.pdf"),
    "file_uri": Media(url="file:///tmp/local.png"),
    "big_png": Media(data=BIG, name="huge.png"),
}
IMAGES = {"png", "jpeg", "gif", "webp"}
SENT: dict[str, set[str]] = {
    "openai_responses": IMAGES | {"pdf", "image_url", "pdf_url", "big_png"},
    "openai_chat": set(),
    "anthropic": IMAGES | {"pdf", "text", "image_url", "pdf_url"},
    "gemini_3": {"png", "jpeg", "webp", "pdf", "big_png"},
    "gemini_2": set(),
    "mcp": set(MEDIA),
}
DATA = {"temp": 21}


def _message(value: Any) -> Message:
    text, media = _tool_parts(ToolResult(call_id="c1", name="look", status="ok", value=value))
    return Message(role="tool", tool_call_id="c1", name="look", content=text, media=media)


def _request(value: Any) -> LLMRequest:
    call = ToolCall(name="look", arguments={}, id="c1")
    history = (
        Message(role="user", content="look"),
        Message(role="assistant", tool_calls=(call,)),
        _message(value),
    )
    return LLMRequest(messages=history)


def _mcp(value: Any) -> Any:
    registry = Registry()

    @registry.tool
    def look() -> dict[str, int]:
        """Look."""
        return DATA

    return result_blocks(registry.get("look"), value)


RENDER: dict[str, Callable[[Any], Any]] = {
    "openai_responses": lambda v: build_responses_request(_request(v), model="gpt-5")["input"],
    "openai_chat": lambda v: build_chat_request(_request(v), model="gpt-5")["messages"],
    "anthropic": lambda v: build_anthropic_request(_request(v), model="c", max_tokens=9),
    "gemini_3": lambda v: build_gemini_request(_request(v), model="gemini-3-pro-preview"),
    "gemini_2": lambda v: build_gemini_request(_request(v), model="gemini-2.5-flash"),
    "mcp": _mcp,
}


def _leaves(node: Any) -> Iterator[Any]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)
    elif isinstance(node, list | tuple):
        for value in node:
            yield from _leaves(value)
    else:
        yield node


def _sent(payload: Any, media: Media) -> bool:
    leaves = list(_leaves(payload))
    if media.url:
        return media.url in leaves
    strings = [leaf for leaf in leaves if isinstance(leaf, str)]
    encoded = media.base64()
    return (
        media.data in leaves
        or any(encoded in leaf for leaf in strings)
        or media.data.decode("latin-1") in strings
    )


def _noted(payload: Any, media: Media) -> bool:
    marker = f"{media.label!r}"
    return any(
        isinstance(leaf, str) and "not shown" in leaf and marker in leaf
        for leaf in _leaves(payload)
    )


@pytest.mark.parametrize("structured", [False, True], ids=["media-only", "with-structured"])
@pytest.mark.parametrize("provider", sorted(RENDER))
@pytest.mark.parametrize("kind", sorted(MEDIA))
def test_media_is_sent_or_noted_never_dropped(kind: str, provider: str, structured: bool) -> None:
    media = MEDIA[kind]
    value = [DATA, media] if structured else media
    payload = RENDER[provider](value)
    sent, noted = _sent(payload, media), _noted(payload, media)
    assert sent != noted, f"{kind} on {provider}: sent={sent} noted={noted}"
    assert sent == (kind in SENT[provider])
    assert ("temp" in json.dumps(payload, default=repr)) == structured


@pytest.mark.parametrize("provider", sorted(RENDER))
def test_mixed_list_keeps_every_item_in_order(provider: str) -> None:
    items = [MEDIA["png"], "caption", MEDIA["zip"], MEDIA["pdf"]]
    payload = RENDER[provider](items)
    for media in (MEDIA["png"], MEDIA["zip"], MEDIA["pdf"]):
        assert _sent(payload, media) != _noted(payload, media)
    assert "caption" in json.dumps(payload, default=repr)


def test_mcp_keeps_media_next_to_structured_content() -> None:
    blocks, structured = _mcp([DATA, MEDIA["png"], MEDIA["wav"], MEDIA["pdf"], MEDIA["image_url"]])
    assert structured == DATA
    assert [block["type"] for block in blocks] == [
        "text",
        "image",
        "audio",
        "resource",
        "resource_link",
    ]
    assert json.loads(blocks[0]["text"]) == DATA
    assert base64.b64decode(blocks[3]["resource"]["blob"]) == MEDIA["pdf"].data


def test_mcp_media_only_has_no_structured_content() -> None:
    blocks, structured = _mcp(MEDIA["text"])
    assert structured is None
    assert blocks == [
        {
            "type": "resource",
            "resource": {
                "uri": "media:///notes.txt",
                "mimeType": "text/plain",
                "text": "plain notes",
            },
        }
    ]
