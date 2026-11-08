"""The "I want to... -> use..." section; `tests/docs/test_agent_guide.py` runs or checks each."""

from __future__ import annotations

import importlib
import importlib.util

from scripts.agent_guide.recipe_model import MAX_LINES, PRELUDE, Recipe
from scripts.agent_guide.recipes_core import TOOLS
from scripts.agent_guide.recipes_data import DATA
from scripts.agent_guide.recipes_hosts import CHAT, HOSTS, VIEWER
from scripts.agent_guide.recipes_ops import OPS
from scripts.agent_guide.recipes_trust import TRUST

__all__ = ["MAX_LINES", "PRELUDE", "Recipe", "has_viewer", "recipes", "render"]


def has_viewer(name: str = "record") -> bool:
    """True when `agent_adapter.viewer` exists and exports `name`."""
    if importlib.util.find_spec("agent_adapter.viewer") is None:
        return False
    viewer = importlib.import_module("agent_adapter.viewer")
    return name in getattr(viewer, "__all__", ())


def recipes() -> tuple[Recipe, ...]:
    """Every recipe; the viewer and chat rows only when `agent_adapter.viewer` has them."""
    rows = (*TOOLS, *DATA, *HOSTS, *OPS, *TRUST)
    rows = (*rows, VIEWER) if has_viewer() else rows
    return (*rows, *CHAT) if has_viewer("serve") else rows


def render(*, full: bool = True) -> str:
    """The set-up, then a snippet per recipe; the concise tier lists the others by name."""
    out = [
        "## I want to… → use…",
        "",
        "Every snippet runs after this set-up (a test runs each one, or compiles it and checks",
        "it against the API when its framework is not installed). Lines with `await` go in an",
        "`async def`; in sync code, wrap them with `aa.run_sync()`.",
        "",
        "```python",
        PRELUDE.rstrip(),
        "```",
        "",
    ]
    shown = [recipe for recipe in recipes() if full or recipe.concise]
    for i, recipe in enumerate(shown, start=1):
        out += [
            f"### {i}. {_title(recipe)} → {recipe.use}",
            "",
            "```python",
            recipe.code,
            "```",
            "",
        ]
    listed = [recipe for recipe in recipes() if not full and recipe.full_only and not recipe.index]
    if listed:
        out += ["### More recipes: code in the full guide (`agent-adapter guide --full`)", ""]
        out += [f"- {_title(recipe)}" for recipe in listed] + [""]  # names only: the cap
    names = [recipe.index for recipe in recipes() if not full and recipe.index]
    if names:
        out += [f"More features (full recipes in llms-full.txt): {', '.join(names)}.", ""]
    return "\n".join(out)


def _title(recipe: Recipe) -> str:
    return recipe.want[0].upper() + recipe.want[1:]
