"""Media in tool results, rewritten per provider; what an API cannot take becomes a text note.

Every function returns each `Media` either as a provider block or as a note that names it
and says why it was left out, so nothing is dropped silently.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_adapter.core.media import Media, media_note

logger = logging.getLogger("agent_adapter.media")
_warned: set[tuple[str, str]] = set()

_IMAGES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_MB = 1024 * 1024


@dataclass(frozen=True)
class MediaRules:
    """What one API takes back in a tool result: MIME types, a size limit, URL support."""

    api: str
    mime_types: frozenset[str]
    max_bytes: int
    url_schemes: tuple[str, ...] = ("https://", "http://")

    def refusal(self, media: Media) -> str:
        """Why `media` cannot be sent, or "" when it can."""
        if not self.mime_types:
            return f"{self.api} cannot take media in tool results"
        if media.mime_type not in self.mime_types:
            return f"{self.api} does not take {media.mime_type} in tool results"
        if media.url and not media.url.startswith(self.url_schemes):
            return f"{self.api} cannot fetch this URL in a tool result"
        if len(media.data) > self.max_bytes:
            return f"over the {self.max_bytes // _MB} MB limit of {self.api}"
        return ""


# Sources: OpenAI Responses `function_call_output` (input_image, input_file); Anthropic
# `tool_result` (image, document; 5 MB per image); Gemini 3 multimodal function responses;
# MCP CallToolResult content (image, audio, resource_link, resource).
OPENAI_RESPONSES = MediaRules("OpenAI Responses", _IMAGES | {"application/pdf"}, 20 * _MB)
OPENAI_CHAT = MediaRules("OpenAI Chat Completions", frozenset(), 0)
ANTHROPIC = MediaRules("Anthropic", _IMAGES | {"application/pdf", "text/plain"}, 5 * _MB)
GEMINI = MediaRules(
    "Gemini",
    frozenset({"image/png", "image/jpeg", "image/webp", "application/pdf"}),
    20 * _MB,
    url_schemes=("gs://", "https://generativelanguage.googleapis.com/"),  # Files API URIs
)
GEMINI_TEXT_ONLY = MediaRules("this Gemini model", frozenset(), 0)


def note(media: Media, reason: str) -> str:
    """The note for `media` left out; logged once per reason and MIME type."""
    key = (reason, media.mime_type)
    if key not in _warned:
        _warned.add(key)
        logger.warning("tool result media not sent: %s", media_note(media, reason))
    return media_note(media, reason)


def _with_notes(text: str, notes: list[str]) -> str:
    return "\n".join([text, *notes]) if text else "\n".join(notes)


def openai_chat_content(text: str, media: Sequence[Media]) -> str:
    """A Chat Completions tool message body: tool messages take text only."""
    if not media:
        return text
    reason = OPENAI_CHAT.refusal(media[0]) + "; use the Responses API (api='responses')"
    return _with_notes(text, [note(item, reason) for item in media])


def openai_output(text: str, media: Sequence[Media]) -> str | list[dict[str, Any]]:
    """A Responses `function_call_output.output`: the text, or input_text/image/file items."""
    if not media:
        return text
    items: list[dict[str, Any]] = []
    notes: list[str] = []
    for item in media:
        if reason := OPENAI_RESPONSES.refusal(item):
            notes.append(note(item, reason))
        elif item.kind == "image":
            items.append({"type": "input_image", "image_url": item.data_url(), "detail": "auto"})
        elif item.url:
            items.append({"type": "input_file", "file_url": item.url})
        else:
            items.append(
                {"type": "input_file", "filename": item.label, "file_data": item.data_url()}
            )
    body = _with_notes(text, notes)
    return [{"type": "input_text", "text": body}, *items] if body else items


def anthropic_content(text: str, media: Sequence[Media]) -> str | list[dict[str, Any]]:
    """A `tool_result.content`: the text, or text, image and document blocks."""
    if not media:
        return text
    blocks: list[dict[str, Any]] = []
    notes: list[str] = []
    for item in media:
        reason = ANTHROPIC.refusal(item)
        if not reason and item.url and item.mime_type == "text/plain":
            reason = "Anthropic takes URL documents only as PDF"
        if reason:
            notes.append(note(item, reason))
        elif item.mime_type == "text/plain":
            source = {"type": "text", "media_type": "text/plain", "data": _text(item)}
            blocks.append({"type": "document", "source": source, "title": item.label})
        else:
            block_type = "image" if item.kind == "image" else "document"
            blocks.append({"type": block_type, "source": _anthropic_source(item)})
    body = _with_notes(text, notes)
    return [{"type": "text", "text": body}, *blocks] if body else blocks


def _anthropic_source(item: Media) -> dict[str, Any]:
    if item.url:
        return {"type": "url", "url": item.url}
    return {"type": "base64", "media_type": item.mime_type, "data": item.base64()}


def _text(item: Media) -> str:
    return item.data.decode("utf-8", errors="replace")


def gemini_takes_media(model: str) -> bool:
    """Whether a Gemini model takes media inside function responses (Gemini 3 and later)."""
    name = model.rsplit("/", 1)[-1]
    if not name.startswith("gemini-"):
        return False
    major = name.removeprefix("gemini-").split("-", 1)[0].split(".", 1)[0]
    return major.isdigit() and int(major) >= 3


def gemini_parts(
    media: Sequence[Media], *, model: str
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    """`functionResponse.parts`, the `{"$ref": name}` entries that point at them, and notes."""
    rules = GEMINI if gemini_takes_media(model) else GEMINI_TEXT_ONLY
    parts: list[dict[str, Any]] = []
    refs: list[dict[str, str]] = []
    notes: list[str] = []
    for item in media:
        if reason := rules.refusal(item):
            notes.append(note(item, reason))
            continue
        name = _unique(item.label, refs)
        body: dict[str, Any]
        if item.url:
            body = {"mime_type": item.mime_type, "file_uri": item.url, "display_name": name}
            parts.append({"file_data": body})
        else:
            body = {"mime_type": item.mime_type, "data": item.data, "display_name": name}
            parts.append({"inline_data": body})
        refs.append({"$ref": name})
    return parts, refs, notes


def _unique(name: str, refs: list[dict[str, str]]) -> str:
    taken = {ref["$ref"] for ref in refs}
    candidate, n = name, 2
    while candidate in taken:
        candidate, n = f"{n}-{name}", n + 1
    return candidate


def mcp_content(text: str, media: Sequence[Media]) -> list[dict[str, Any]]:
    """MCP `CallToolResult.content` blocks: text, image, audio, resource_link or resource."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text or not media else []
    for item in media:
        if item.url:
            blocks.append(
                {"type": "resource_link", "uri": item.url, "name": item.label}
                | {"mimeType": item.mime_type}
            )
        elif item.kind in ("image", "audio"):
            blocks.append({"type": item.kind, "data": item.base64(), "mimeType": item.mime_type})
        else:
            blocks.append({"type": "resource", "resource": _resource(item)})
    return blocks


def _resource(item: Media) -> dict[str, Any]:
    resource: dict[str, Any] = {"uri": f"media:///{item.label}", "mimeType": item.mime_type}
    if item.mime_type.startswith("text/"):
        resource["text"] = _text(item)
    else:
        resource["blob"] = item.base64()
    return resource


__all__ = [
    "ANTHROPIC",
    "GEMINI",
    "OPENAI_CHAT",
    "OPENAI_RESPONSES",
    "MediaRules",
    "anthropic_content",
    "gemini_parts",
    "gemini_takes_media",
    "mcp_content",
    "openai_chat_content",
    "openai_output",
]
