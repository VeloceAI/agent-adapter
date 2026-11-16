"""The Media result type: sniffing, limits, JSON round trips and splitting tool values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import agent_adapter as aa
from agent_adapter.core import media as media_module
from agent_adapter.core.media import (
    Media,
    as_media,
    media_kind,
    media_note,
    mentions_media,
    sniff_mime,
    split_media,
)
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolResult, to_jsonable
from agent_adapter.exec._loop_steps import _tool_content

PNG = b"\x89PNG\r\n\x1a\nrest"


@pytest.mark.parametrize(
    ("data", "mime"),
    [
        (PNG, "image/png"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"GIF87a..", "image/gif"),
        (b"RIFF\0\0\0\0WEBPVP8 ", "image/webp"),
        (b"RIFF\0\0\0\0WAVEfmt ", "audio/wav"),
        (b"RIFF\0\0\0\0AVI LIST", ""),
        (b"%PDF-1.4", "application/pdf"),
        (b"\xff\xfb\x90", "audio/mpeg"),
        (b"OggS\0", "audio/ogg"),
        (b"fLaC\0", "audio/flac"),
        (b"\0\0\0\x20ftypM4A ", "audio/mp4"),
        (b"\0\0\0\x20ftypisom", "video/mp4"),
        (b"\x1a\x45\xdf\xa3", "video/webm"),
        (b'<?xml version="1.0"?><svg/>', "image/svg+xml"),
        (b"hello", ""),
    ],
)
def test_sniff_mime(data: bytes, mime: str) -> None:
    assert sniff_mime(data) == mime


def test_mime_from_bytes_then_name_then_url() -> None:
    assert Media(data=PNG, name="x.jpg").mime_type == "image/png"
    assert Media(data=b"hello", name="notes.txt").mime_type == "text/plain"
    assert Media(url="https://e.com/a.pdf?x=1").mime_type == "application/pdf"
    assert Media(data=b"\0\1").mime_type == "application/octet-stream"
    assert Media(data=PNG, mime_type="image/x-mine").mime_type == "image/x-mine"


@pytest.mark.parametrize(
    ("mime", "kind"),
    [
        ("image/png", "image"),
        ("audio/wav", "audio"),
        ("video/mp4", "video"),
        ("application/pdf", "document"),
        ("text/csv", "document"),
        ("application/zip", "file"),
    ],
)
def test_media_kind(mime: str, kind: str) -> None:
    assert media_kind(mime) == kind


def test_needs_exactly_one_of_data_or_url() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        Media()
    with pytest.raises(ValidationError, match="exactly one"):
        Media(data=PNG, url="https://e.com/a.png")


def test_base64_text_and_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Media(data="aGk=", mime_type="text/plain").data == b"hi"
    with pytest.raises(ValidationError, match="base64"):
        Media(data="not base64!")
    monkeypatch.setattr(media_module, "MAX_MEDIA_BYTES", 4)
    with pytest.raises(ValidationError, match="limit is 4"):
        Media(data=PNG)


def test_from_file(tmp_path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(PNG)
    media = Media.from_file(path)
    assert (media.name, media.mime_type, media.data) == ("shot.png", "image/png", PNG)
    with pytest.raises(ValueError, match="limit is 3"):
        Media.from_file(path, max_bytes=3)


def test_labels_and_descriptions() -> None:
    media = Media(data=PNG, name="shot.png")
    assert media.kind == "image"
    assert media.data_url().startswith("data:image/png;base64,iVBORw0KGg")
    assert media.describe() == "image/png 'shot.png' (1 KB)"
    link = Media(url="https://e.com/img/cat.png?size=2")
    assert (link.label, link.data_url(), link.base64()) == ("cat.png", link.url, "")
    assert link.describe() == f"image/png 'cat.png' ({link.url})"
    assert Media(url="https://e.com/", mime_type="image/png").label == "e.com"
    note = media_note(media, "too big")
    assert note == "[image not shown: image/png 'shot.png' (1 KB); too big]"


def test_json_round_trip_and_revive() -> None:
    media = Media(data=b"\0\xff binary", mime_type="application/zip", name="b.zip")
    assert Media.model_validate_json(media.model_dump_json()) == media
    dumped = to_jsonable(media)
    assert dumped["type"] == "media" and isinstance(dumped["data"], str)
    assert as_media(dumped) == media
    assert as_media({"type": "media", "data": "bad!"}) is None
    assert as_media({"type": "other"}) is None
    odd = Media(data=bytes([0xFB, 0xFF, 0xFE]) * 4, mime_type="application/zip")
    assert to_jsonable(odd)["data"] == "+//+" * 4  # standard base64, as providers need
    assert as_media({**to_jsonable(odd), "data": "-__-" * 4}) == odd  # URL-safe is read too
    assert Media(data=b"a,b", name="t.csv").mime_type == "text/csv"  # not the registry's
    assert as_media({"type": "other"}) is None


def test_split_media() -> None:
    png = Media(data=PNG)
    assert split_media(png) == (None, (png,))
    assert split_media("text") == ("text", ())
    assert split_media(["a", "b"]) == (["a", "b"], ())
    assert split_media([png, "caption"]) == ("caption", (png,))
    assert split_media(("a", png, {"x": 1})) == (["a", {"x": 1}], (png,))
    assert split_media([png, png.model_dump(mode="json")]) == (None, (png, png))


def test_mentions_media() -> None:
    assert mentions_media(Media)
    assert mentions_media(list[str | Media])
    assert not mentions_media(dict[str, int])


def test_tool_returning_media_has_no_result_schema() -> None:
    registry = aa.Registry()

    @registry.tool
    def shot() -> Media:
        """Screenshot."""
        return Media(data=PNG)

    assert registry.get("shot").result_schema() is None


def test_message_dump_leaves_out_empty_media() -> None:
    assert "media" not in Message(role="tool", content="x").model_dump()
    message = Message(role="tool", content="", media=(Media(data=PNG),))
    assert Message.model_validate_json(message.model_dump_json()) == message


def test_text_only_channels_get_a_note() -> None:
    result = ToolResult(call_id="c", name="t", status="ok", value=["seen", Media(data=PNG)])
    assert _tool_content(result) == (
        "seen\n[image not shown: image/png 'image' (1 KB); this channel takes text only]"
    )
    plain = ToolResult(call_id="c", name="t", status="ok", value={"a": 1})
    assert _tool_content(plain) == '{"a": 1}'


async def test_tool_loop_sends_media_to_the_model_and_survives_the_cache() -> None:
    registry = aa.Registry()
    calls = 0

    @registry.tool(cache_ttl=60, cache_scope="global")
    def shot(page: str) -> list[str | Media]:
        """Screenshot a page."""
        nonlocal calls
        calls += 1
        return [f"page {page}", Media(data=PNG, name=f"{page}.png")]

    seen: list[Message] = []

    def model(request: aa.LLMRequest) -> dict:
        last = request.messages[-1]
        if last.role != "tool":
            return {"tool_calls": [{"name": "shot", "arguments": {"page": "home"}}]}
        seen.append(last)
        return {"text": "a home page"}

    executor = aa.Executor(registry, cache=aa.MemoryCache())
    llm = aa.CallableModel(model)
    for _ in range(2):
        ask = [aa.Message(role="user", content="look")]
        result = await aa.run_tool_loop(llm, registry, ask, executor=executor)
        assert result.final.text == "a home page"
    assert calls == 1  # the second run is a cache hit, and its media revive
    for message in seen:
        assert message.content == "page home"
        assert message.media == (Media(data=PNG, name="home.png"),)
