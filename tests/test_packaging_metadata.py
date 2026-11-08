"""pyproject guards: the sdist is allow-listed, the licence is Apache-2.0, pins are capped."""

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    return tomllib.loads(ROOT.joinpath("pyproject.toml").read_text(encoding="utf-8"))


def test_sdist_ships_only_the_package_readme_and_licence(pyproject: dict[str, Any]) -> None:
    sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert sdist["only-include"] == [
        "agent_adapter",
        "README.md",
        "PYPI.md",
        "CHANGELOG.md",
        "LICENSE",
        "NOTICE",
    ]
    assert pyproject["project"]["readme"] in sdist["only-include"]


def test_licence_is_apache_2_with_no_upload_block(pyproject: dict[str, Any]) -> None:
    project = pyproject["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert "Version 2.0, January 2004" in ROOT.joinpath("LICENSE").read_text(encoding="utf-8")
    # PyPI refuses a Private :: classifier, and a License :: classifier beside an expression.
    assert not [c for c in project["classifiers"] if c.startswith(("Private ::", "License ::"))]


def test_every_extra_pin_has_an_upper_bound(pyproject: dict[str, Any]) -> None:
    extras = pyproject["project"]["optional-dependencies"]
    tools = {"dev", "docs"}  # dev tooling floats on purpose
    uncapped = [
        dep
        for name, deps in extras.items()
        if name not in tools
        for dep in deps
        if not dep.startswith("agent-adapter[") and "<" not in dep
    ]
    assert uncapped == []


def test_the_pypi_page_shows_nothing_pypi_cannot_render() -> None:
    page = ROOT.joinpath("PYPI.md").read_text(encoding="utf-8")
    assert "```mermaid" not in page and "<img" not in page and "<picture" not in page
    relative = re.findall(r"\]\((?!https://)[^)]+\)", page)
    assert relative == [], "PyPI cannot follow links into the repository"


def test_the_pypi_page_examples_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    page = ROOT.joinpath("PYPI.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", page, re.S)
    assert len(blocks) == 2
    monkeypatch.chdir(tmp_path)  # the approval example writes approvals.db here
    namespace: dict[str, Any] = {}
    for block in blocks:
        exec(compile(block, "PYPI.md", "exec"), namespace)
