"""The docs' chat demo page: the chat scripts plus the demo's stand-in API, offline and safe."""

from __future__ import annotations

import json
import re

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.viewer.assets import (
    CHAT_SCRIPTS,
    DEMO_SCRIPTS,
    SCRIPTS,
    _read,
    export_chat_demo_html,
    export_html,
    static_file,
)
from tests.viewer.test_export import EXTERNAL, FETCHED

ORDER = ("chatapi.js", "chatdemo.js", "markdown.js", "chat.js")
CHAT = {"info": {"enabled": True, "mode": "demo"}, "live": "python app.py", "chats": []}


def test_the_demo_needs_recorded_chats() -> None:
    with pytest.raises(ConfigError, match="recorded chats"):
        export_chat_demo_html({"runs": [], "details": {}})


def test_the_demo_keeps_the_chat_page_and_adds_its_stand_in_api_after_the_real_one() -> None:
    html = export_chat_demo_html({"runs": [], "details": {}, "chat": CHAT})
    assert 'src="static/' not in html and 'href="static/' not in html
    assert ".chat-shell" in html and "AA.chatApi = {" in html and "AA.chatDemo" in html
    order = [html.index(_read(name).splitlines()[0]) for name in ORDER]
    assert order == sorted(order), "the stand-in replaces the real API before any view uses it"
    assert 'name="aa-chat" content="demo"' in html
    assert html.count("</script>") == len((*SCRIPTS, *CHAT_SCRIPTS, *DEMO_SCRIPTS)) + 1


def test_the_demo_is_offline_and_its_data_cannot_close_the_script() -> None:
    tricky = dict(CHAT, live="</script><script>alert(1)</script>")
    html = export_chat_demo_html({"runs": [], "details": {}, "chat": tricky})
    assert "default-src 'none'" in html and "Content-Security-Policy" in html
    assert not EXTERNAL.search(html)
    assert set(FETCHED.findall(html.replace("http://www.w3.org/2000/svg", ""))) == set()
    assert "<script>alert(1)" not in html
    found = re.search(r'id="aa-data">(.*?)</script>', html, re.S)
    assert found is not None and json.loads(found.group(1))["chat"]["live"] == tricky["live"]


def test_the_demo_script_is_never_served_live_nor_in_a_plain_export() -> None:
    assert static_file("chatdemo.js") is None
    html = export_html({"runs": [], "details": {}, "chat": CHAT})
    assert "AA.chatDemo" not in html and "AA.chatApi" not in html
