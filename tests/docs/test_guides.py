"""Run the guide and troubleshooting examples offline; check them against their pages."""

import importlib
import importlib.util
import json
import re
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
DOCS = ROOT / "docs"
SNIPPET = re.compile(r'--8<-- "(examples/[\w/]+\.py)')


def _lines(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def _page_output(page: str) -> list[str]:
    """Every line of the page's ```text blocks, in order."""
    blocks = re.findall(r"```text\n(.*?)```", (DOCS / page).read_text(encoding="utf-8"), re.S)
    return [line for block in blocks for line in block.splitlines()]


@pytest.mark.parametrize(
    ("example", "page"),
    [
        ("guide_troubleshooting.py", "troubleshooting.md"),
        ("guide_security.py", "guides/security.md"),
        ("guide_versioning.py", "guides/versioning.md"),
    ],
)
def test_guide_prints_what_its_page_shows(
    example: str, page: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _lines(example, capsys) == _page_output(page)


def test_troubleshooting_lists_every_printed_message(capsys: pytest.CaptureFixture[str]) -> None:
    page = (DOCS / "troubleshooting.md").read_text(encoding="utf-8")
    for line in _lines("guide_troubleshooting.py", capsys):
        message = line.split(" | ", 1)[-1]
        if "known:" not in message and "|" in line:
            assert message.split(": ")[-1].split(" (")[0] in page, message


def test_performance_import_is_cheap_and_caps_hold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["guide_performance.py"])
    lines = _lines("guide_performance.py", capsys)
    assert lines == _page_output("guides/performance.md")


async def test_performance_overhead_timer_measures_something() -> None:
    ns = runpy.run_path(str(EXAMPLES / "guide_performance.py"))
    assert await ns["overhead_us"](n=20) > 0


def test_observability_streams_share_the_run_id(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _lines("guide_observability.py", capsys)
    page = _page_output("guides/observability.md")
    assert lines[:-1] == page[:-1]
    otel = importlib.util.find_spec("opentelemetry") is not None
    assert lines[-1] == f"otel spans: {otel}"
    assert json.loads(lines[0].replace("'", '"'))[0] == "run.started"


def test_performance_page_describes_loop_runner() -> None:
    page = (DOCS / "guides" / "performance.md").read_text(encoding="utf-8")
    assert hasattr(importlib.import_module("agent_adapter.runtime"), "LoopRunner")
    assert "to fill in" not in page and "runner.run(work, timeout=)" in page


def test_security_page_links_the_http_security_model() -> None:
    page = (DOCS / "guides" / "security.md").read_text(encoding="utf-8")
    target = (DOCS / "cross-language" / "http-server.md").read_text(encoding="utf-8")
    assert "../cross-language/http-server.md#security-model" in page
    assert "\n## Security model\n" in target


def test_guide_pages_include_only_existing_guide_examples() -> None:
    pages = [DOCS / "troubleshooting.md", *sorted((DOCS / "guides").glob("*.md"))]
    for page in pages:
        for example in SNIPPET.findall(page.read_text(encoding="utf-8")):
            assert example.startswith("examples/guide_"), (page.name, example)
            assert (ROOT / example).is_file(), (page.name, example)


def test_choose_page_links_existing_pages() -> None:
    page = DOCS / "choose.md"
    links = re.findall(r"\]\(([\w/.-]+\.md)(?:#[\w-]+)?\)", page.read_text(encoding="utf-8"))
    assert links and all((DOCS / link).is_file() for link in links)
