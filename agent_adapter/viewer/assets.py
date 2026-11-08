"""The page's static files: served one by one, or inlined into one export file."""

from __future__ import annotations

import base64
import functools
import json
import re
from pathlib import Path
from typing import Any

from agent_adapter.core.errors import ConfigError

STATIC = Path(__file__).resolve().parent / "static"
SCRIPTS = (
    "data.js",
    "graphlive.js",
    "graphnav.js",
    "graph.js",
    "graphbar.js",
    "timeline.js",
    "events.js",
    "statepanel.js",
    "compare.js",
    "panels.js",
    "views.js",
    "runsearch.js",
    "app.js",
)
STYLES = ("viewer.css", "graph.css", "tabs.css")
# The chat page: served live, left out of an export (chat never runs offline).
CHAT_SCRIPTS = (
    "chatapi.js",
    "markdown.js",
    "chatcards.js",
    "chatmemory.js",
    "inspector.js",
    "chatextras.js",
    "chatsessions.js",
    "chatcomposer.js",
    "chat.js",
)
CHAT_STYLES = ("chat.css",)
# The Evals view and saving runs as cases: served live, left out of every export (server-backed).
EVAL_SCRIPTS = ("evals.js", "evalscards.js", "evalsview.js", "runcases.js")
EVAL_STYLES = ("evals.css",)
# The docs' recorded chat demo: stands in for the chat API; only in `export_chat_demo_html`.
DEMO_SCRIPTS = ("chatdemo.js",)
# The browser-tab icon: a copy of docs/assets/favicon.svg (a test keeps the two identical).
ICONS = ("favicon.svg",)
_LIVE = (*SCRIPTS, *STYLES, *CHAT_SCRIPTS, *CHAT_STYLES, *EVAL_SCRIPTS, *EVAL_STYLES, *ICONS)


def _tags(scripts: tuple[str, ...], styles: tuple[str, ...]) -> re.Pattern[str]:
    """The page's tags for these files with their line end (LF or CRLF), to drop from an export."""
    return re.compile(
        r'(?:<script src="static/(?:{})"></script>|<link rel="stylesheet" href="static/(?:{})">)'
        r"[ \t]*(?:\r?\n)?".format(
            "|".join(map(re.escape, scripts)), "|".join(map(re.escape, styles))
        )
    )


_CHAT_TAGS = _tags(CHAT_SCRIPTS, CHAT_STYLES)
_EVAL_TAGS = _tags(EVAL_SCRIPTS, EVAL_STYLES)
_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
}
_CSRF = "__AA_CSRF__"
_APPROVALS = "__AA_APPROVALS__"
_CHAT = "__AA_CHAT__"
_CHAT_MODES = frozenset({"", "builtin", "agent"})
_DATA = "<!-- AA_DATA -->"
_CHAT_META = '<meta name="aa-chat" content="{}">'
# Export CSP: nothing may load from anywhere; only the inlined code and styles run.
EXPORT_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src data:; base-uri 'none'; form-action 'none'"
)


@functools.cache
def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def static_file(name: str) -> tuple[bytes, str] | None:
    """A static file's bytes and content type; None for anything not shipped with the page."""
    if name not in _LIVE:
        return None
    return _read(name).encode("utf-8"), _TYPES[Path(name).suffix]


def page(csrf: str, *, approvals: bool, chat: str = "") -> str:
    """The served index page, with this start's CSRF token; `chat` is "builtin", "agent" or ""."""
    if chat not in _CHAT_MODES:
        raise ConfigError(f"chat must be one of {sorted(_CHAT_MODES)}, got {chat!r}")
    html = _read("index.html").replace(_CSRF, csrf).replace(_CHAT, chat)
    return html.replace(_APPROVALS, "on" if approvals else "")


def export_html(data: dict[str, Any]) -> str:
    """One self-contained HTML file: styles, scripts and run data inline, no external URL."""
    html = _inline(page("", approvals=False), data, SCRIPTS, STYLES)
    return _EVAL_TAGS.sub("", _CHAT_TAGS.sub("", html))


def export_chat_demo_html(data: dict[str, Any]) -> str:
    """The chat page as one offline file replaying recorded chats (`data["chat"]`); internal.

    Raises:
        ConfigError: `data` holds no recorded chats.
    """
    if not isinstance(data.get("chat"), dict):
        raise ConfigError("the chat demo needs the recorded chats under data['chat']")
    html = page("", approvals=False).replace(_CHAT_META.format(""), _CHAT_META.format("demo"))
    api = '<script src="static/chatapi.js"></script>'
    demo = "".join(f'\n<script src="static/{name}"></script>' for name in DEMO_SCRIPTS)
    html = html.replace(api, api + demo)
    html = _inline(html, data, (*SCRIPTS, *CHAT_SCRIPTS, *DEMO_SCRIPTS), (*STYLES, *CHAT_STYLES))
    return _EVAL_TAGS.sub("", html)


def _inline(
    html: str, data: dict[str, Any], scripts: tuple[str, ...], styles: tuple[str, ...]
) -> str:
    """`html` with the export CSP, then `styles`, the icon, `data` and `scripts` inlined."""
    csp = f'<meta http-equiv="Content-Security-Policy" content="{EXPORT_CSP}">'
    html = html.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n' + csp)
    for name in styles:
        html = html.replace(
            f'<link rel="stylesheet" href="static/{name}">', f"<style>\n{_read(name)}</style>"
        )
    for name in ICONS:
        html = html.replace(f'href="static/{name}"', f'href="{_data_uri(name)}"')
    payload = f'<script type="application/json" id="aa-data">{embed_json(data)}</script>'
    html = html.replace(_DATA, payload)
    for name in scripts:
        html = html.replace(
            f'<script src="static/{name}"></script>', f"<script>\n{_read(name)}</script>"
        )
    return html


def _data_uri(name: str) -> str:
    """A static file as a base64 `data:` URI, which the export CSP's `img-src data:` allows."""
    encoded = base64.b64encode(_read(name).encode("utf-8")).decode("ascii")
    return f"data:{_TYPES[Path(name).suffix]};base64,{encoded}"


def embed_json(data: Any) -> str:
    """JSON safe inside a <script> element: no `<`, `>`, `&` or line separators unescaped."""
    text = json.dumps(data, ensure_ascii=False, default=str)
    for char, escape in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        text = text.replace(char, escape)
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
