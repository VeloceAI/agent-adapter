"""The Examples gallery lists every example and matches its generator.

Regenerate it with `python scripts/gen_examples_gallery.py`.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from scripts.examples_gallery import areas, install, render

ROOT = Path(__file__).resolve().parents[2]
EXTRAS_IN_LINE = re.compile(r'"agent-adapter\[([^\]]+)\]"')


@pytest.fixture(scope="module")
def examples() -> list[render.Example]:
    return render.collect()


def _gallery_text() -> str:
    return "\n".join(
        (ROOT / "docs" / page).read_text(encoding="utf-8") for page in render.gallery_pages()
    )


def test_gallery_files_are_up_to_date() -> None:
    stale = [path.relative_to(ROOT).as_posix() for path in render.stale()]
    assert not stale, f"run `python scripts/gen_examples_gallery.py`: {stale}"


def test_every_example_is_listed_once(examples: list[render.Example]) -> None:
    text = _gallery_text()
    files = [p.relative_to(ROOT).as_posix() for p in render.example_files()]
    assert files and len(examples) == len(files)
    for file in files:
        assert text.count(f"| `{file}` |") == 1, file


def test_every_gallery_page_is_in_the_nav() -> None:
    nav = render.nav_pages()
    assert all(page in nav for page in render.gallery_pages())


def test_every_area_has_examples(examples: list[render.Example]) -> None:
    assert {e.area for e in examples} == {a.slug for a in areas.AREAS}


def test_every_docs_link_resolves(examples: list[render.Example]) -> None:
    for example in examples:
        assert example.docs is not None, f"examples/{example.rel}.py is on no docs page"
        assert (ROOT / "docs" / example.docs[0]).is_file()


def test_install_lines_name_only_real_extras(examples: list[render.Example]) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    real = set(pyproject["project"]["optional-dependencies"])
    for example in examples:
        lines = example.install + " " + example.optional
        named = {e for m in EXTRAS_IN_LINE.findall(lines) for e in m.split(",")}
        assert named <= real, example.rel


def test_install_line_follows_imports_helpers_and_implied_extras() -> None:
    assert install.install_line("hello") == ("pip install agent-adapter", "")
    assert install.install_line("crewai_tools") == ('pip install "agent-adapter[crewai]"', "")
    fastapi = install.install_line("backend_fastapi")[0]  # through backend_common
    assert fastapi == 'pip install "agent-adapter[server]" fastapi'
    langgraph = install.install_line("langgraph_flow")[0]  # langgraph brings langchain-core
    assert langgraph == 'pip install "agent-adapter[langgraph]"'


def test_optional_paths_get_their_own_line_with_what_they_really_import() -> None:
    otel = 'pip install "agent-adapter[otel]" opentelemetry-sdk'  # the extra is the API only
    assert install.install_line("tracing") == ("pip install agent-adapter", otel)
    assert install.install_line("compression_backend") == (
        "pip install agent-adapter",
        "pip install fastapi",
    )
    hooks = install.install_line("compression_framework_hooks")  # main() needs no framework
    assert hooks[0] == "pip install agent-adapter" and "strands" in hooks[1]
    troubleshooting = install.install_line("guide_troubleshooting")  # shows crewai missing
    assert troubleshooting == ("pip install agent-adapter", "")


def test_run_cells_say_import_only_and_starts_a_server(examples: list[render.Example]) -> None:
    by_rel = {e.rel: e for e in examples}
    assert by_rel["hello"].run == "python examples/hello.py"  # a top-level print
    assert by_rel["harness_frameworks"].run == ""  # only a top-level registry.wrap(...)
    assert by_rel["backend_fastapi"].serves == "on :8000"
    assert by_rel["mcp_http"].serves == ""  # serves only behind --serve
    assert by_rel["quickstart"].serves == ""


def test_template_rows_do_not_repeat_the_scaffolded_run_line(
    examples: list[render.Example],
) -> None:
    for example in examples:
        if example.rel.startswith("template_"):
            assert " Run: " not in example.summary and " Demo: " not in example.summary
            assert example.summary.endswith("Scaffolded by `agent-adapter init`.")


def test_an_example_without_an_area_fails_loudly() -> None:
    with pytest.raises(areas.GalleryError, match="no gallery area"):
        areas.area_of("zz_unplaced")


def test_check_mode_reports_stale_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "stale", lambda: [ROOT / "docs" / "examples.md"])
    assert render.main(["--check"]) == 1
    assert render.main(["--bogus"]) == 2
