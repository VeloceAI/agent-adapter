"""Spec drift: the public API, the generated LLM guide and the docs name the same things."""

# test_agent_guide.py and test_reference_pages.py check each generator against its output;
# these cover the gaps: the two walking different modules, a removed name left in the
# committed guide, and prose or guide code naming an import path that is gone.

from __future__ import annotations

import importlib
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_adapter.integrations import dbos, temporal
from agent_adapter.integrations.ag_ui._events import TRACE_EVENT
from agent_adapter.integrations.ag_ui._resume import RESUME_IGNORED
from agent_adapter.tracing import semconv
from scripts.agent_guide import api_index
from tests.docs.reference_layout import public_modules
from tests.docs.test_canonical_imports import FENCE, _parse, _problems

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "agent_adapter"
GUIDE = (ROOT / "AGENTS.md", ROOT / "docs" / "llms-full.txt")
UNPUBLISHED = ROOT / "docs" / "superpowers"
DOTTED = re.compile(r"(?<![\w./-])(agent_adapter(?:\.[A-Za-z_]\w*)+)")
MODULE_HEADING = re.compile(r"^### `(agent_adapter[\w.]*)`$", re.M)
ENTRY = re.compile(r"^- `(?:async )?([A-Za-z_]\w*)", re.M)
# Wire names that look like import paths: span attributes (`agent_adapter.approval_id`),
# the durable engines' workflow, activity and topic names, and
# AG-UI custom event names. Anything else must import.
WIRE_NAMES = frozenset(
    value for name, value in vars(semconv).items() if name.startswith("AA_") and name != "AA_PREFIX"
) | {TRACE_EVENT, RESUME_IGNORED}
WIRE_NAMES |= {
    value
    for module in (temporal, dbos)
    for value in vars(module).values()
    if isinstance(value, str) and value.startswith("agent_adapter.")
}


def _published() -> list[Path]:
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md")), *GUIDE]
    return [page for page in pages if UNPUBLISHED not in page.parents]


def _resolves(path: str) -> bool:
    """True when `path` imports as a module, or names an attribute of the deepest module."""
    parts = path.split(".")
    for depth in range(len(parts), 0, -1):
        try:
            found: object = importlib.import_module(".".join(parts[:depth]))
        except ImportError:
            continue
        for attribute in parts[depth:]:
            if not hasattr(found, attribute):
                return False
            found = getattr(found, attribute)
        return True
    return False


def dangling_references(text: str) -> list[str]:
    """`agent_adapter....` paths in `text` that do not resolve, wire names aside."""
    found = {match.group(1) for match in DOTTED.finditer(text)}
    return sorted(path for path in found - WIRE_NAMES if not _resolves(path))


def _guide_sections(text: str) -> Iterator[tuple[str, list[str]]]:
    """(module, names) for each module section of the guide's API index."""
    index = text[text.index("## API index") :]
    headings = list(MODULE_HEADING.finditer(index))
    for here, after in zip(headings, [*headings[1:], None], strict=True):
        body = index[here.end() : after.start() if after else len(index)]
        yield here.group(1), ENTRY.findall(body)


def test_the_guide_index_and_the_reference_walk_the_same_packages() -> None:
    guide = set(api_index.public_modules())
    reference = public_modules()
    assert sorted(guide - set(reference)) == [], "guide modules without a reference page"
    expected = {
        module
        for module in reference
        if (len(module.split(".")) == 2 and (PACKAGE / module.split(".")[1]).is_dir())
        or (module.startswith("agent_adapter.integrations.") and module.count(".") == 2)
    } | {"agent_adapter", "agent_adapter.errors"}
    assert sorted(expected - guide) == [], "public packages missing from the guide's API index"
    for module in guide:
        names = sorted(name for name in reference[module] if not name.startswith("__"))
        assert api_index.exported(module) == names, module


def test_every_name_in_the_committed_guide_index_is_still_public() -> None:
    path = GUIDE[1]  # the full tier holds the API index
    sections = dict(_guide_sections(path.read_text(encoding="utf-8")))
    reference = public_modules()
    assert sorted(set(sections) - set(reference)) == [], "documented modules that are gone"
    for module, names in sections.items():
        gone = sorted(set(names) - set(reference[module]))
        assert gone == [], f"{path.name}: {module} documents {gone}; run gen_agent_guide.py"


@pytest.mark.parametrize("path", GUIDE, ids=lambda path: path.name)
def test_guide_code_imports_only_public_names(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    problems = [problem for block in FENCE.findall(text) for problem in _problems(_parse(block))]
    assert problems == []


def test_import_paths_named_in_docs_and_the_guide_exist() -> None:
    dangling = {
        page.relative_to(ROOT).as_posix(): refs
        for page in _published()
        if (refs := dangling_references(page.read_text(encoding="utf-8")))
    }
    assert dangling == {}


def test_the_reference_check_flags_a_removed_name_and_skips_attribute_keys() -> None:
    text = (
        "Use `agent_adapter.cost.Prices` and agent_adapter.models.normalise_usage; "
        "`agent_adapter.cost.NoSuchTable` is gone, as is agent_adapter.nothing_here.x. "
        "Spans carry `agent_adapter.approval_id`. See agent_adapter/cost/prices.py."
    )
    assert dangling_references(text) == [
        "agent_adapter.cost.NoSuchTable",
        "agent_adapter.nothing_here.x",
    ]


def test_the_guide_sections_parser_reads_module_entries() -> None:
    text = "## API index\n\n### `agent_adapter.cost`\n\nIntro.\n\n- `Prices(x) -> None`: A.\n"
    text += "- `async acall()`: B.\n\n### `agent_adapter.errors`\n\n- `AdapterError`: C.\n"
    assert list(_guide_sections(text)) == [
        ("agent_adapter.cost", ["Prices", "acall"]),
        ("agent_adapter.errors", ["AdapterError"]),
    ]
