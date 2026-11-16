"""MCP content blocks from a remote tool as `Media`: image, audio, resource_link, resource."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent_adapter.core.media import Media


def block_media(block: Any) -> Media | None:
    """The block as Media, or None for text and unknown blocks."""
    kind = getattr(block, "type", "")
    mime = getattr(block, "mime_type", None) or ""
    if kind in ("image", "audio"):
        return Media(data=block.data, mime_type=mime)
    if kind == "resource_link":
        return Media(url=str(block.uri), mime_type=mime, name=block.name)
    if kind != "resource":
        return None
    resource = block.resource
    mime = getattr(resource, "mime_type", None) or ""
    name = str(resource.uri).rstrip("/").rsplit("/", 1)[-1]
    blob = getattr(resource, "blob", None)
    if blob is not None:
        return Media(data=blob, mime_type=mime, name=name)
    return Media(data=resource.text.encode("utf-8"), mime_type=mime or "text/plain", name=name)


def remote_media(blocks: Iterable[Any]) -> list[Media]:
    """Every media block of a remote tool result, in order."""
    return [media for block in blocks if (media := block_media(block)) is not None]
