"""The run viewer demo in the docs matches its generator, stays offline, and the page finds it.

Regenerate it with `python scripts/gen_docs_demo.py`.
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path

import pytest

from scripts import gen_docs_demo
from tests.viewer.test_export import EXTERNAL, FETCHED

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PAGE = DOCS / "how-to" / "navigate-the-graph.md"
DEMO = DOCS / "assets" / "demo" / "sample-run.html"
IFRAME = re.compile(r'<iframe class="aa-demo" src="([^"]+)" title="([^"]+)"[^>]*loading="lazy"')


@pytest.fixture(scope="module")
def built() -> str:
    return gen_docs_demo.build()


def test_the_committed_demo_matches_its_generator(built: str) -> None:
    assert DEMO.read_text(encoding="utf-8") == built, "run python scripts/gen_docs_demo.py"


def test_the_generator_is_byte_stable(built: str) -> None:
    assert gen_docs_demo.build() == built


def test_check_mode_reports_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stale = tmp_path / "sample-run.html"
    stale.write_text("old", encoding="utf-8")
    monkeypatch.setattr(gen_docs_demo, "OUT", stale)
    monkeypatch.setattr(gen_docs_demo, "ROOT", tmp_path)
    assert gen_docs_demo.main(["--check"]) == 1
    assert gen_docs_demo.main([]) == 0
    assert gen_docs_demo.main(["--check"]) == 0


def test_the_demo_is_offline_and_holds_the_sample_runs() -> None:
    html = DEMO.read_text(encoding="utf-8")
    assert "default-src 'none'" in html
    assert not EXTERNAL.search(html)
    assert set(FETCHED.findall(html.replace("http://www.w3.org/2000/svg", ""))) == set()
    assert "AA.chatApi" not in html, "chat is left out of an export"
    data = json.loads(re.search(r'id="aa-data">(.*?)</script>', html, re.S).group(1))  # type: ignore[union-attr]
    assert sorted(run["status"] for run in data["runs"]) == ["ok", "ok", "paused"]
    flow_runs = [run for run in data["runs"] if run["flows"] == ["help_search"]]
    assert [run["key"] for run in flow_runs] == ["run_" + "0" * 31 + "1"]  # the page links it
    events = data["details"][flow_runs[0]["key"]]["events"]
    assert any(e["type"] == "node.finished" and "updates" in e["data"] for e in events), "traced"
    local_path = re.compile(r"[A-Za-z]:\\\\|/home/|/Users/|\.agent_adapter[/\\]")
    assert not local_path.search(html), "no local paths"
    assert len(html.encode("utf-8")) < 400_000


def test_the_graph_page_embeds_the_demo_where_the_built_site_serves_it() -> None:
    text = PAGE.read_text(encoding="utf-8")
    match = IFRAME.search(text)
    assert match is not None, "an iframe with a title and lazy loading"
    assert "Agent Adapter" in match.group(2)
    directory_urls = "use_directory_urls: false" not in (ROOT / "mkdocs.yml").read_text("utf-8")
    base = "how-to/navigate-the-graph/" if directory_urls else "how-to/"  # the page's URL folder
    src, _, fragment = match.group(1).partition("#")
    served = posixpath.normpath(posixpath.join(base, src))
    assert (DOCS / served).resolve() == DEMO.resolve(), served
    flow_run = "run=run_" + "0" * 31 + "1"  # the flow run opens first, not the newest run
    assert fragment == flow_run
    assert f"](../assets/demo/sample-run.html#{flow_run})" in text, (
        "a full-screen link checked by mkdocs --strict"
    )


def test_the_demo_frame_is_sandboxed_without_the_docs_origin() -> None:
    tag = re.search(r"<iframe class=\"aa-demo\"[^>]*>", PAGE.read_text(encoding="utf-8"))
    assert tag is not None
    sandbox = re.search(r'sandbox="([^"]*)"', tag.group(0))
    assert sandbox is not None, "the demo runs sandboxed"
    assert set(sandbox.group(1).split()) == {"allow-scripts", "allow-downloads"}
    assert 'allow="clipboard-write"' in tag.group(0), "the Mermaid button can copy"


def test_the_demo_frame_never_fills_a_phone_screen() -> None:
    css = (DOCS / "stylesheets" / "extra.css").read_text(encoding="utf-8")
    heights = re.findall(r"iframe\.aa-demo\s*\{[^}]*height:\s*([^;]+);", css)
    assert any("vh" in h for h in heights), "capped by the viewport height"
    small = re.search(
        r"@media \(max-width: (\d+)px\)\s*\{[^@]*iframe\.aa-demo\s*\{[^}]*height:\s*(\d+)vh", css
    )
    assert small is not None and int(small.group(2)) <= 70, "room above or below it to scroll"
