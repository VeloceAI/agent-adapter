"""Apply a reviewed harness: load `harness.after.json` and build the prompt layout it describes."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from agent_adapter.cache.layout import PromptLayout
from agent_adapter.core.errors import ConfigError, OptimiserError
from agent_adapter.optimiser.model import HarnessTree
from agent_adapter.optimiser.surface import parse_path

logger = logging.getLogger(__name__)

AFTER_FILE = "harness.after.json"
HINTS_SECTION = "aa.tool_hints"  # not a valid tree section name, so it never collides


def load_harness(out_dir: str | os.PathLike[str]) -> HarnessTree:
    """The tree an optimiser run chose (or its base), read from `<out_dir>/harness.after.json`.

    Raises:
        ConfigError: the file is missing, not JSON, of another version, or holds a bad path.
    """
    path = Path(out_dir) / AFTER_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: not found; run the optimiser first") from exc
    except (OSError, ValueError) as exc:
        logger.error("cannot read %s: %s", path, type(exc).__name__)
        raise ConfigError(f"{path}: cannot read it ({exc})") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ConfigError(f"{path}: not a version 1 harness file")
    files = data.get("files")
    if not isinstance(files, dict) or not all(isinstance(v, str) for v in files.values()):
        raise ConfigError(f"{path}: 'files' must map paths to text")
    for name in files:
        try:
            parse_path(name)
        except OptimiserError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    return HarnessTree(files=files)


def layout_from_tree(base: PromptLayout, tree: HarnessTree) -> PromptLayout:
    """The tree's system prompt, sections and tool hints; only `base.dynamic` is kept."""
    hints = [
        (parse_path(path).name, text)
        for path, text in tree.files.items()
        if path.startswith("tool/") and path.endswith("/prompt_hint") and text
    ]
    return layout_with_hints(base, tree, hints)


def layout_with_hints(
    base: PromptLayout, tree: HarnessTree, hints: Sequence[tuple[str, str]]
) -> PromptLayout:
    """The layout of `tree` with `(tool, hint)` pairs as a hints section; keeps `base.dynamic`."""
    sections = {
        path.removeprefix("prompt/layout/"): text
        for path, text in tree.files.items()
        if path.startswith("prompt/layout/")
    }
    if hints:
        lines = "\n".join(f"- {name}: {hint}" for name, hint in hints)
        sections[HINTS_SECTION] = "When to use each tool:\n" + lines
    return PromptLayout(
        static=tree.files.get("prompt/system", ""), dynamic=base.dynamic, sections=sections
    )
