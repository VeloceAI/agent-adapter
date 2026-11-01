"""Every name in a public `__all__` renders once in the API reference, at its import path."""

import re
from collections import Counter
from pathlib import Path

import pytest

from tests.docs.reference_layout import (
    ERRORS_PAGE,
    FOLDER,
    PAGES,
    ROOT,
    canonical_home,
    page_of,
    public_modules,
    render,
    render_errors_api,
)

INDEX = ROOT / "docs" / "reference.md"
REFERENCE_PAGES = [INDEX, *sorted(FOLDER.glob("*.md"))]
DIRECTIVE = re.compile(r"^::: (\S+)\n((?:[ ]{4}.*\n)*)", re.M)
MEMBERS = re.compile(r"^\s+members: \[(.*)\]\s*$", re.M)
REGENERATE = "run `python -m tests.docs.reference_layout`"


def directives() -> list[tuple[Path, str, list[str]]]:
    """(page, module, members) for every `:::` block on the reference pages."""
    found = []
    for page in REFERENCE_PAGES:
        for match in DIRECTIVE.finditer(page.read_text(encoding="utf-8")):
            members = MEMBERS.search(match.group(2))
            assert members, f"{page.name}: `::: {match.group(1)}` needs a `members: [...]` list"
            names = [name.strip() for name in members.group(1).split(",") if name.strip()]
            found.append((page, match.group(1), names))
    return found


def test_every_directive_names_a_public_module() -> None:
    for page, module, names in directives():
        assert module in public_modules(), f"{page.name}: {module} is not a public module"
        unknown = set(names) - set(public_modules()[module])
        assert not unknown, f"{page.name}: {module} has no {sorted(unknown)} in __all__"


def test_every_public_name_renders_exactly_once_at_its_import_path() -> None:
    rendered = Counter((module, name) for _, module, names in directives() for name in names)
    twice = sorted(key for key, count in rendered.items() if count > 1)
    assert twice == []
    expected = {
        (canonical_home(module, name), name)
        for module, names in public_modules().items()
        for name in names
    }
    assert sorted(expected - set(rendered)) == [], "public names missing from the reference"
    assert sorted(set(rendered) - expected) == [], "names rendered away from their import path"


@pytest.mark.parametrize("name", ["Decision", "default_registry", "tool", "wrap", "__version__"])
def test_names_without_docstrings_are_listed_explicitly(name: str) -> None:
    # An explicit `members` list renders a name even when it has no docstring.
    top = [names for _, module, names in directives() if module == "agent_adapter"]
    assert len(top) == 1
    assert name in top[0]


def test_every_public_module_has_a_page_and_every_page_module_exists() -> None:
    assert sorted(set(public_modules()) - set(page_of())) == [], REGENERATE
    assert sorted(set(page_of()) - set(public_modules())) == []


@pytest.mark.parametrize("page", sorted(PAGES))
def test_generated_page_is_current(page: str) -> None:
    text = (FOLDER / f"{page}.md").read_text(encoding="utf-8")
    assert text == render(page), REGENERATE


def test_errors_api_section_is_current() -> None:
    text = (FOLDER / f"{ERRORS_PAGE}.md").read_text(encoding="utf-8")
    assert text == render_errors_api(text), REGENERATE


def test_every_reexported_name_links_to_where_it_is_documented() -> None:
    pages = page_of()
    for module, names in public_modules().items():
        text = (FOLDER / f"{pages[module]}.md").read_text(encoding="utf-8")
        for name in names:
            home = canonical_home(module, name)
            if home != module:
                assert f"#{home}.{name})" in text, (module, name)


def test_the_index_links_every_reference_page() -> None:
    index = INDEX.read_text(encoding="utf-8")
    for page in REFERENCE_PAGES[1:]:
        assert f"(reference/{page.name})" in index, page.name
