"""The "Connect anywhere" pages: our side runs offline as shown, every page keeps its shape."""

import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "connect_server.py"
CONNECT = ROOT / "docs" / "connect"
PAGES = sorted(page for page in CONNECT.glob("*.md") if page.name != "index.md")
SNIPPET = re.compile(r'--8<-- "(examples/[\w/]+\.py):(\w+)"')
SECTION = re.compile(r"^## (.+)$", re.M)
SOURCE = re.compile(r"<https://[^>\s]+>")
FENCE = re.compile(r"^```(\w+)\n(.*?)^```", re.M | re.S)


def _sections(page: Path) -> dict[str, str]:
    text = page.read_text(encoding="utf-8")
    parts = SECTION.split(text)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def test_connect_example_replays_a_remote_host(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("mcp")
    pytest.importorskip("starlette")
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "['shipping_rate', 'send_invoice']",
        "{'result': 18.0}",
        "True True",  # parked: isError, `approval required: <id>`
        "202 approved",  # decided over REST, not run yet
        "True",  # a re-call without resume keys parks a new approval
        "invoiced cus_1024 for 49.50 USD",  # the resume keys run the approved call once
        "202 pending_approval",  # REST: invoke parks
        "invoiced cus_1024 for 49.50 USD",  # REST: decide runs it
        "421",  # a foreign Host header
        "401",  # no token
        "['shipping_rate']",  # the hosted registry has no confirm tools
    ]


def test_local_app_serves_mcp_only() -> None:
    pytest.importorskip("mcp")
    app = runpy.run_path(str(EXAMPLE))["local_app"]()
    assert [route.path for route in app.routes] == ["/mcp"]


def test_serve_never_falls_back_to_a_default_token(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AA_PUBLIC_HOST", "AA_AGENT_TOKEN", "AA_REVIEWER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(KeyError):
        runpy.run_path(str(EXAMPLE))["serve"]()


def test_connect_example_type_checks_strictly() -> None:
    pytest.importorskip("mcp")
    mypy_api = pytest.importorskip("mypy.api")
    out, err, status = mypy_api.run(["--strict", "--no-error-summary", str(EXAMPLE)])
    assert status == 0, out + err


def test_index_links_every_recipe_page() -> None:
    index = (CONNECT / "index.md").read_text(encoding="utf-8")
    assert len(PAGES) == 12
    assert [page.name for page in PAGES if f"]({page.name})" not in index] == []


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.stem)
def test_recipe_page_has_both_sides_a_checklist_and_sources(page: Path) -> None:
    sections = _sections(page)
    assert "Our side" in sections
    assert any(title.startswith("Their side") for title in sections)
    assert "public-host-checklist-every-page" in sections["Checklist"]
    assert SOURCE.search(sections["Sources"]), "cite the stack's own docs"
    assert len(page.read_text(encoding="utf-8").splitlines()) <= 150


@pytest.mark.parametrize("page", [CONNECT / "index.md", *PAGES], ids=lambda page: page.stem)
def test_our_side_is_only_tested_snippets(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    example = EXAMPLE.read_text(encoding="utf-8")
    for path, name in SNIPPET.findall(text):
        assert path == "examples/connect_server.py"
        assert f"# --8<-- [start:{name}]" in example and f"# --8<-- [end:{name}]" in example
    our_side = _sections(page).get("Our side", "")
    for language, code in FENCE.findall(our_side):
        assert language == "python"
        lines = [line for line in code.splitlines() if line.strip()]
        assert lines and all(SNIPPET.fullmatch(line) for line in lines), code
