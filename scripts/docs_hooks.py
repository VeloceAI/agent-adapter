"""MkDocs hooks: a description per page for search and link previews; versioned own assets."""

from __future__ import annotations

import hashlib
import html
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_CHARS = 160  # what search results and link previews show before cutting
MIN_CHARS = 40  # shorter paragraphs (a status line, "Options:") say too little
_PARAGRAPH = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_ICON = re.compile(r"<svg\b|class=\"twemoji", re.I)


def description_from_html(content: str) -> str:
    """The page's first real paragraph as plain text, cut at a word to MAX_CHARS; "" if none."""
    for match in _PARAGRAPH.finditer(content):
        inner = match.group(1)
        if _ICON.search(inner):
            continue  # a status marker line, not prose
        text = " ".join(html.unescape(_TAG.sub(" ", inner)).split())
        text = re.sub(r" ([,.;:!?)])", r"\1", text).replace("( ", "(")
        if len(text) < MIN_CHARS:
            continue
        if len(text) <= MAX_CHARS:
            return text
        cut = text[: MAX_CHARS - 1].rsplit(" ", 1)[0].rstrip(",;:")
        return cut + "…"
    return ""


def on_page_content(content: str, page: Any, **_: Any) -> str:
    """Set `description` from the first paragraph unless the page's front matter has one.

    The theme writes `description` into HTML attributes unescaped, so it is stored escaped.
    """
    if not page.meta.get("description"):
        found = description_from_html(content)
        if found:
            page.meta["description"] = html.escape(found, quote=True)
        else:
            logger.debug("docs_hooks: %s has no paragraph to describe it", page.file.src_uri)
    return content


_versions: dict[str, str] = {}


def asset_version(path: Path) -> str:
    """A short hash of the file's bytes: it changes exactly when the file does."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def versioned(output: str, versions: dict[str, str]) -> str:
    """Add `?v=<hash>` to each link to one of our assets: browsers never keep a stale copy."""
    for path, version in versions.items():
        pattern = re.compile(r'((?:src|href)="(?:\.\./|\./)*' + re.escape(path) + r')"')
        output = pattern.sub(r"\1?v=" + version + '"', output)
    return output


def on_config(config: Any, **_: Any) -> Any:
    """Hash the site's own stylesheets and scripts (extra_css, extra_javascript)."""
    _versions.clear()
    docs = Path(config["docs_dir"])
    for entry in [*config["extra_css"], *config["extra_javascript"]]:
        path = str(getattr(entry, "path", entry))
        if (docs / path).is_file():
            _versions[path] = asset_version(docs / path)
    return config


def on_post_page(output: str, **_: Any) -> str:
    """Version every link to an asset hashed in on_config."""
    return versioned(output, _versions)
