"""The "I want to... -> use..." section; `tests/docs/test_agent_guide.py` runs or checks each."""

from __future__ import annotations

import importlib
import importlib.util

from scripts.agent_guide.recipe_model import MAX_LINES, PRELUDE, Recipe
from scripts.agent_guide.recipes_core import TOOLS
from scripts.agent_guide.recipes_hosts import CHAT, HOSTS, VIEWER

__all__ = ["MAX_LINES", "PRELUDE", "Recipe", "has_viewer", "recipes", "render"]


def has_viewer(name: str = "record") -> bool:
    """True when `agent_adapter.viewer` exists and exports `name`."""
    if importlib.util.find_spec("agent_adapter.viewer") is None:
        return False
    viewer = importlib.import_module("agent_adapter.viewer")
    return name in getattr(viewer, "__all__", ())


def recipes() -> tuple[Recipe, ...]:
    """Every recipe; the viewer and chat rows only when `agent_adapter.viewer` has them."""
    rows = (*TOOLS, *HOSTS)
    rows = (*rows, VIEWER) if has_viewer() else rows
    return (*rows, *CHAT) if has_viewer("serve") else rows


def render() -> str:
    """The set-up, then one heading and snippet per recipe."""
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
    for i, recipe in enumerate(recipes(), start=1):
        title = recipe.want[0].upper() + recipe.want[1:]
        out += [f"### {i}. {title} → {recipe.use}", "", "```python", recipe.code, "```", ""]
    return "\n".join(out)
