"""Run the migration examples offline and check them against their pages."""

import importlib.util
import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
MIGRATION = ROOT / "docs" / "migration"
SNIPPET = re.compile(r'--8<-- "(examples/[\w/]+\.py)')


def _lines(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def _page_output(page: str) -> list[str]:
    text = (MIGRATION / page).read_text(encoding="utf-8")
    blocks = re.findall(r"```text\n(.*?)```", text, re.S)
    return [line for block in blocks for line in block.splitlines()]


@pytest.mark.parametrize(
    ("example", "page"),
    [
        ("guide_migrate_raw.py", "raw-tool-calling.md"),
        ("guide_migrate_approval.py", "hand-rolled-approval.md"),
    ],
)
def test_migration_prints_what_its_page_shows(
    example: str, page: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _lines(example, capsys) == _page_output(page)


def test_langchain_migration_runs_without_langchain(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _lines("guide_migrate_langchain.py", capsys)
    if importlib.util.find_spec("langchain_core") is None:
        assert lines == ["3 ok", "langchain-core not installed: import and export steps skipped"]
    else:
        assert lines == _page_output("langchain-tools.md")


async def test_langchain_import_and_export_steps() -> None:
    pytest.importorskip("langchain_core")
    ns = runpy.run_path(str(EXAMPLES / "guide_migrate_langchain.py"))
    assert await ns["import_existing"]() == "FAQ"
    assert (await ns["export_back"]()).content == "2"


def test_old_approval_flow_pays_twice_and_the_new_one_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ns = runpy.run_path(str(EXAMPLES / "guide_migrate_approval.py"), run_name="__main__")
    capsys.readouterr()
    assert ns["refunds"] == [500.0]


def test_index_links_every_migration_page() -> None:
    index = (MIGRATION / "index.md").read_text(encoding="utf-8")
    pages = {p.name for p in MIGRATION.glob("*.md")} - {"index.md"}
    assert pages and all(f"]({page})" in index for page in pages)


def test_migration_pages_include_only_existing_migration_examples() -> None:
    for page in MIGRATION.glob("*.md"):
        for example in SNIPPET.findall(page.read_text(encoding="utf-8")):
            assert example.startswith("examples/guide_migrate_"), (page.name, example)
            assert (ROOT / example).is_file(), (page.name, example)
