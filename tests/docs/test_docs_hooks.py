"""scripts/docs_hooks.py: each page's description comes from its first real paragraph."""

import importlib.util
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("docs_hooks", ROOT / "scripts" / "docs_hooks.py")
assert _SPEC is not None and _SPEC.loader is not None
hooks = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hooks)

LONG = "Chat with any agent in the local page, and watch it work: streamed text, tool calls, "


def _page(meta: dict[str, str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(meta=meta, file=types.SimpleNamespace(src_uri="x.md"))


def test_the_first_real_paragraph_becomes_plain_text() -> None:
    html = (
        '<p><span class="twemoji"><svg></svg></span> Recipe · checked</p>'
        "<p>Options:</p>"
        "<p>Write a tool <em>once</em>, as a plain <code>Python</code> function"
        " &amp; use it anywhere .</p>"
    )
    assert hooks.description_from_html(html) == (
        "Write a tool once, as a plain Python function & use it anywhere."
    )


def test_a_long_paragraph_is_cut_at_a_word() -> None:
    found = hooks.description_from_html(f"<p>{LONG * 4}</p>")
    assert len(found) <= hooks.MAX_CHARS and found.endswith("…")
    assert not found[:-1].endswith(" ") and LONG.startswith(found[: len(LONG) - 20])


def test_no_paragraph_gives_no_description() -> None:
    assert hooks.description_from_html("<h1>Title</h1><pre>code</pre><p>short</p>") == ""


def test_the_hook_keeps_front_matter_and_escapes_what_it_sets() -> None:
    kept = _page({"description": "Set by hand."})
    hooks.on_page_content(f"<p>{LONG}</p>", page=kept)
    assert kept.meta == {"description": "Set by hand."}
    found = _page({})
    out = hooks.on_page_content(f'<p>{LONG} "quoted" &lt;tag&gt;</p>', page=found)
    assert out.startswith("<p>")
    assert '"' not in found.meta["description"] and "<" not in found.meta["description"]
    assert "&quot;quoted&quot; &lt;tag&gt;" in found.meta["description"]
    empty = _page({})
    hooks.on_page_content("<h1>Only a title</h1>", page=empty)
    assert empty.meta == {}


def test_own_assets_get_a_content_version_and_others_stay_as_they_are() -> None:
    page = (
        '<link rel="stylesheet" href="../../stylesheets/extra.css">'
        '<script src="assets/javascripts/diagrams.js" defer></script>'
        '<script src="https://example.com/assets/javascripts/diagrams.js.map"></script>'
    )
    out = hooks.versioned(
        page, {"stylesheets/extra.css": "aaa", "assets/javascripts/diagrams.js": "bbb"}
    )
    assert 'href="../../stylesheets/extra.css?v=aaa"' in out
    assert 'src="assets/javascripts/diagrams.js?v=bbb"' in out
    assert 'diagrams.js.map"' in out


def test_the_version_changes_with_the_file(tmp_path: Path) -> None:
    file = tmp_path / "a.js"
    file.write_text("one", encoding="utf-8")
    first = hooks.asset_version(file)
    file.write_text("two", encoding="utf-8")
    assert hooks.asset_version(file) != first and len(first) == 10
