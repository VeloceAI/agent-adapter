"""Provider media rules: which Gemini models take media, notes are logged once."""

from __future__ import annotations

import logging

import pytest

from agent_adapter.core.media import Media
from agent_adapter.schema import media as rules


@pytest.mark.parametrize(
    ("model", "takes"),
    [
        ("gemini-3-pro-preview", True),
        ("models/gemini-3.1-flash", True),
        ("gemini-4", True),
        ("gemini-2.5-flash", False),
        ("gemini-pro", False),
        ("learnlm-2", False),
    ],
)
def test_gemini_takes_media(model: str, takes: bool) -> None:
    assert rules.gemini_takes_media(model) is takes


def test_note_is_logged_once_per_reason_and_type(caplog: pytest.LogCaptureFixture) -> None:
    rules._warned.clear()
    shot = Media(data=b"\x89PNG\r\n\x1a\n", name="a.png")
    with caplog.at_level(logging.WARNING, logger="agent_adapter.media"):
        first = rules.openai_chat_content("", [shot])
        rules.openai_chat_content("", [shot, shot])
    assert first.startswith("[image not shown: image/png 'a.png'")
    assert len(caplog.records) == 1


def test_gemini_names_are_unique_and_urls_use_file_data() -> None:
    png = Media(data=b"\x89PNG\r\n\x1a\n", name="a.png")
    link = Media(url="gs://bucket/b.pdf")
    parts, refs, notes = rules.gemini_parts([png, png, link], model="gemini-3-pro")
    assert refs == [{"$ref": "a.png"}, {"$ref": "2-a.png"}, {"$ref": "b.pdf"}]
    assert parts[2] == {
        "file_data": {"mime_type": "application/pdf", "file_uri": link.url, "display_name": "b.pdf"}
    }
    assert notes == []


def test_text_media_with_no_text_part() -> None:
    doc = Media(data=b"hello", mime_type="text/plain", name="n.txt")
    assert rules.anthropic_content("", [doc]) == [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": "hello"},
            "title": "n.txt",
        }
    ]
    link = Media(url="https://e.com/n.txt")
    assert "only as PDF" in rules.anthropic_content("", [link])[0]["text"]
    assert rules.mcp_content("", []) == [{"type": "text", "text": ""}]
