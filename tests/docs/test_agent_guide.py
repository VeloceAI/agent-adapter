"""The AI-agent guide is generated in two tiers, complete, safe-first and runnable.

Concise (`AGENTS.md`, packaged `AGENTS.md`) and full (`docs/llms-full.txt`, packaged
`llms-full.txt`). Regenerate all four with `python scripts/gen_agent_guide.py`.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

import agent_adapter
from scripts.agent_guide import api_index, recipes, render
from scripts.agent_guide.install import CHECKOUT_ONLY, pyproject
from tests.docs.test_no_internal_refs import _hits

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_DOMAIN = re.compile(r"\b(tax|taxes|gst|pan|80c|taxmann|income-tax)\b", re.I)
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```$", re.M | re.S)
CONCISE_MAX_BYTES = 32_000  # about 7.5k tokens: cheap to load into every agent session
ORDER = (
    "## What it is",
    "## Install",
    "## Import rules",
    "## Safety rules",
    "## I want to",
    "## Common errors",
    "## CLI commands",
)


@pytest.fixture(scope="module")
def concise() -> str:
    return render.render_concise()


@pytest.fixture(scope="module")
def full() -> str:
    return render.render_full()


def test_agent_guide_files_match_the_generator(concise: str, full: str) -> None:
    for path in render.CONCISE:
        assert path.read_text(encoding="utf-8") == concise, f"{path}: run gen_agent_guide.py"
    for path in render.FULL:
        assert path.read_text(encoding="utf-8") == full, f"{path}: run gen_agent_guide.py"


def test_agent_guide_check_mode_reports_fresh_files() -> None:
    assert render.stale() == []
    assert render.main(["--check"]) == 0
    assert render.main(["--bogus"]) == 2


@pytest.mark.parametrize("tier", ["concise", "full"])
def test_agent_guide_puts_safety_first_and_the_api_index_last(tier: str, request) -> None:
    text = request.getfixturevalue(tier)
    positions = [text.index(f"\n{heading}") for heading in ORDER]
    assert positions == sorted(positions)
    last = "## API index" if tier == "full" else "## Full API index"
    assert text.rindex("\n## ") == text.index(f"\n{last}")


def test_agent_guide_concise_tier_is_small_and_points_to_the_full_one(concise: str) -> None:
    assert len(concise.encode("utf-8")) <= CONCISE_MAX_BYTES
    assert "## API index" not in concise
    assert "agent-adapter guide --full" in concise and "llms-full.txt>" in concise


def _module_section(full: str, module: str) -> str:
    start = full.index(f"\n### `{module}`\n")
    end = full.find("\n### `", start + 1)
    return full[start:] if end == -1 else full[start:end]


@pytest.mark.parametrize("module", api_index.public_modules())
def test_agent_guide_lists_every_public_name(full: str, module: str) -> None:
    section = _module_section(full, module)
    missing = [
        name
        for name in api_index.exported(module)
        if not re.search(rf"^- `(async )?{re.escape(name)}\b", section, re.M)
    ]
    assert missing == []


def test_agent_guide_api_index_covers_top_level_and_integrations(full: str) -> None:
    modules = api_index.public_modules()
    assert modules[:2] == ["agent_adapter", "agent_adapter.errors"]
    assert "agent_adapter.integrations.langgraph" in modules
    assert "agent_adapter.core" not in modules  # never an import path for users
    assert all(f"`{name}" in full for name in agent_adapter.__all__ if name != "__version__")


def test_agent_guide_api_index_hides_internals(full: str) -> None:
    index = full[full.index("## API index") :]
    assert " at 0x" not in index
    assert "agent_adapter.core." not in index
    assert not re.search(r"[(, ]_[a-z]\w*[:=]", index), "private parameters leak"
    assert "- `Decision`: alias of `ApprovalDecision`" in index


def test_agent_guide_docstring_summaries_end_on_a_sentence(full: str) -> None:
    index = full[full.index("## API index") :]
    cut = [line for line in index.splitlines() if line.startswith("- ") and line.endswith(",")]
    assert cut == []


@pytest.mark.parametrize("tier", ["concise", "full"])
def test_agent_guide_has_no_forbidden_domain_words_or_internal_refs(tier: str, request) -> None:
    text = request.getfixturevalue(tier)
    assert sorted({m.group(0) for m in FORBIDDEN_DOMAIN.finditer(text)}) == []
    assert _hits(text) == []


def test_llms_txt_links_the_full_guide() -> None:
    llms = (ROOT / "docs" / "llms.txt").read_text(encoding="utf-8")
    assert "/llms-full.txt>" in llms.split("\n## ", 1)[0]  # in the header, before any section


@pytest.mark.parametrize("tier", ["concise", "full"])
def test_agent_guide_install_table_lists_every_extra(tier: str, request) -> None:
    text = request.getfixturevalue(tier)
    install = text[text.index("## Install") : text.index("## Import rules")]
    for extra in pyproject()["project"]["optional-dependencies"]:
        shown = f'"agent-adapter[{extra}]"' in install or f'".[{extra}]"' in install
        assert shown is (tier == "full" or extra not in CHECKOUT_ONLY), extra
    assert "on PyPI" in install and "internal-index" not in install
    assert "install '" not in install  # single quotes fail in cmd.exe


def test_agent_guide_recipe_count_stays_readable() -> None:
    assert 30 <= len(recipes.recipes()) <= 70  # full_only ones cost AGENTS.md one line each


def test_indexed_recipes_are_only_named_in_the_concise_tier(concise: str, full: str) -> None:
    indexed = [recipe for recipe in recipes.recipes() if recipe.index]
    assert indexed, "later features are named in the one index line"
    line = next(row for row in concise.splitlines() if row.startswith("More features ("))
    for recipe in indexed:
        assert recipe.index in line and recipe.code not in concise and recipe.use not in concise
        assert recipe.code in full
    assert "More features (" not in full


def test_full_only_recipes_are_one_line_in_the_concise_tier(concise: str, full: str) -> None:
    only = [recipe for recipe in recipes.recipes() if recipe.full_only and not recipe.index]
    assert only, "the concise tier lists at least one recipe by name"
    for recipe in only:
        title = recipe.want[0].upper() + recipe.want[1:]
        assert f"\n- {title}\n" in concise and recipe.use not in concise
        assert recipe.code not in concise
        assert recipe.code in full and f"\n- {title}\n" not in full


def test_agent_guide_viewer_row_follows_the_package(concise: str) -> None:
    assert ("agent-adapter ui" in concise.split("## Common errors")[0]) is recipes.has_viewer()


def test_runtime_pip_hints_use_double_quotes() -> None:
    """Messages the code prints are copied into shells; single quotes break in cmd.exe."""
    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in sorted((ROOT / "agent_adapter").rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "install 'agent-adapter" in line
    ]
    assert offenders == []


def _ruff_format(code: str) -> str:
    """`code` as `ruff format` writes it under this repository's configuration."""
    done = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", "guide_block.py", "-"],
        input=code,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        check=True,
    )
    return done.stdout


@pytest.mark.skipif(importlib.util.find_spec("ruff") is None, reason="ruff is not installed")
def test_agent_guide_code_blocks_are_ruff_format_clean(concise: str, full: str) -> None:
    """`ruff format .` formats Python blocks in Markdown: the generated files must not change."""
    blocks = sorted({block for text in (concise, full) for block in PYTHON_BLOCK.findall(text)})
    assert len(blocks) > 30
    unformatted = [block.splitlines()[0] for block in blocks if _ruff_format(block) != block]
    assert unformatted == []
