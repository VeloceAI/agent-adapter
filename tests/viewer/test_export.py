"""`--export` and the static page: one offline file, no external URL, safe embedding, packaging."""

from __future__ import annotations

import base64
import fnmatch
import json
import re
import tomllib
from pathlib import Path

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.viewer import Recorder, assets, export
from agent_adapter.viewer.assets import (
    CHAT_SCRIPTS,
    CHAT_STYLES,
    DEMO_SCRIPTS,
    EVAL_SCRIPTS,
    EVAL_STYLES,
    ICONS,
    SCRIPTS,
    STATIC,
    STYLES,
    embed_json,
    static_file,
)
from tests.viewer.helpers import event

ROOT = Path(__file__).resolve().parents[2]
assets_read = assets._read
EXTERNAL = re.compile(
    r"""(?:src|href)\s*=\s*["']?(?:https?:)?//|@import|url\(\s*["']?(?:https?:)?//""", re.I
)
FETCHED = re.compile(r"""https?://""", re.I)


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    folder = tmp_path / "runs"
    recorder = Recorder(folder)
    recorder.emit(event("run.started", 1.0))
    recorder.emit(event("custom.note", 1.1, text="</script><script>alert(1)</script>"))
    recorder.emit(event("run.finished", 1.2, ok=True))
    return folder


def test_export_embeds_every_run_and_inlines_the_page(runs: Path, tmp_path: Path) -> None:
    out = export(runs, tmp_path / "out" / "report.html")
    html = out.read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="aa-data">(.*?)</script>', html, re.S)
    assert match is not None
    data = json.loads(match.group(1))
    (run,) = data["runs"]
    assert run["status"] == "ok"
    assert data["details"][run["key"]]["events"][1]["data"]["text"].startswith("</script>")
    assert 'src="static/' not in html and 'href="static/' not in html
    assert "<style>" in html and "AA.renderTimeline" in html
    # Chat never runs offline: its scripts and styles stay out of an export.
    assert "AA.chatApi" not in html and "static/chat" not in html and ".chat-shell" not in html


def test_export_loads_no_external_url_and_forbids_it(runs: Path, tmp_path: Path) -> None:
    html = export(runs, tmp_path / "report.html").read_text(encoding="utf-8")
    assert not EXTERNAL.search(html)
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    # The only absolute URL is the SVG namespace, which is a name, never fetched.
    assert set(FETCHED.findall(html.replace("http://www.w3.org/2000/svg", ""))) == set()


def test_embedded_json_cannot_close_its_script_element(runs: Path, tmp_path: Path) -> None:
    html = export(runs, tmp_path / "report.html").read_text(encoding="utf-8")
    assert html.count("</script>") == len(SCRIPTS) + 1
    assert "<script>alert(1)" not in html
    tricky = "<" + chr(0x2028) + "&>" + chr(0x2029)
    text = embed_json({"x": tricky})
    assert not {"<", ">", "&", chr(0x2028), chr(0x2029)} & set(text)
    assert json.loads(text) == {"x": tricky}


def test_export_needs_an_html_path_and_handles_an_empty_folder(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"\.html"):
        export(tmp_path, tmp_path / "report.txt")
    html = export(tmp_path / "none", tmp_path / "empty.html").read_text(encoding="utf-8")
    assert '"runs": []' in html


@pytest.mark.parametrize("name", sorted(p.name for p in STATIC.iterdir()))
def test_static_files_stay_small_and_never_close_a_script_element(name: str) -> None:
    text = (STATIC / name).read_text(encoding="utf-8")
    assert len(text.splitlines()) < 500
    if name.endswith((".js", ".css")):
        assert "</script" not in text.lower()
        assert not EXTERNAL.search(text)


def test_static_file_lookup_serves_only_shipped_names() -> None:
    assert static_file("app.js") is not None
    for name in ("index.html", "../server.py", "missing.js", ""):
        assert static_file(name) is None
    shipped = {
        *SCRIPTS,
        *STYLES,
        *CHAT_SCRIPTS,
        *CHAT_STYLES,
        *EVAL_SCRIPTS,
        *EVAL_STYLES,
        *DEMO_SCRIPTS,
        *ICONS,
        "index.html",
    }
    assert {p.name for p in STATIC.iterdir()} == shipped
    assert all(static_file(name) is not None for name in (*CHAT_SCRIPTS, *CHAT_STYLES))


def test_the_page_icon_is_the_docs_favicon_and_an_export_inlines_it(
    runs: Path, tmp_path: Path
) -> None:
    docs_icon = (ROOT / "docs" / "assets" / "favicon.svg").read_bytes()
    assert (STATIC / "favicon.svg").read_bytes() == docs_icon, "copy docs/assets/favicon.svg"
    assert static_file("favicon.svg") == (docs_icon, "image/svg+xml")
    assert '<link rel="icon" type="image/svg+xml" href="static/favicon.svg">' in assets_read(
        "index.html"
    )
    html = export(runs, tmp_path / "report.html").read_text(encoding="utf-8")
    encoded = base64.b64encode(docs_icon).decode("ascii")
    assert f'href="data:image/svg+xml;base64,{encoded}"' in html
    assert "img-src data:" in html, "the export CSP lets the data: icon load"


def test_every_static_file_ships_in_the_wheel_and_the_sdist() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    targets = config["tool"]["hatch"]["build"]["targets"]
    assert "agent_adapter" in targets["wheel"]["packages"]
    assert "agent_adapter" in targets["sdist"]["only-include"]
    # Hatchling drops files the root .gitignore matches; none may match the page's files.
    patterns = [
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for path in STATIC.iterdir():
        rel = path.relative_to(ROOT).as_posix()
        assert not any(
            fnmatch.fnmatch(path.name, p) or fnmatch.fnmatch(rel, p.rstrip("/") + "*")
            for p in patterns
        ), rel


def test_chat_tags_leave_an_export_whatever_the_line_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    lf = assets._read("index.html")
    crlf = lf.replace("\n", "\r\n")
    monkeypatch.setattr(
        assets, "_read", lambda name: crlf if name == "index.html" else assets_read(name)
    )
    html = assets.export_html({"runs": [], "details": {}})
    assert "static/chat" not in html and "static/evals" not in html
    assert "AA.chatApi" not in html and ".chat-shell" not in html
    assert "AA.renderTimeline" in html, "the Runs view is still inlined"
