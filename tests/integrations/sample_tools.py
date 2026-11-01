"""Small tools several integration tests share; their names and docstrings are asserted on."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.errors import ToolError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import build_spec


async def gst_rate(item: str, region: str = "IN") -> float:
    """GST rate for an item.

    Args:
        item: What is sold.
        region: Tax region.
    """
    if item == "kite":
        raise ToolError("no rate for 'kite'")
    return 18.0


def lookup(section: str) -> str:
    """Look up a section."""
    return section


def targeted_registry() -> Registry:
    """`lookup` for every target, plus one copy limited to each built-in provider."""
    reg = Registry()
    reg.add(build_spec(lookup))
    reg.add(build_spec(lookup, name="only_openai", targets="openai"))
    reg.add(build_spec(lookup, name="only_gemini", targets="gemini"))
    reg.add(build_spec(lookup, name="only_anthropic", targets="anthropic"))
    return reg


def add_wipe_registry() -> Registry:
    """A fresh registry: `add` (plain) and `wipe` (needs approval)."""
    reg = Registry()

    @reg.tool
    async def add(a: int, b: int = 2) -> int:
        """Add two integers."""
        return a + b

    @reg.tool(confirm=True)
    async def wipe(path: str) -> str:
        """Delete a path."""
        return f"wiped {path}"

    return reg


class RecordingTools:
    """`pay` (approval over 100) and `wipe` (always approval); records what ran and was asked."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict[str, Any]]] = []
        self.asked: list[int] = []
        registry = Registry()

        def big(call: Any, ctx: Any) -> bool:
            self.asked.append(call.arguments["amount"])
            return bool(call.arguments["amount"] > 100)

        @registry.tool(confirm=big)
        async def pay(amount: int) -> str:
            """Pay an amount (large payments need approval)."""
            self.ran.append(("pay", {"amount": amount}))
            return f"paid {amount}"

        @registry.tool(confirm=True)
        async def wipe(path: str) -> str:
            """Delete a path."""
            self.ran.append(("wipe", {"path": path}))
            return f"wiped {path}"

        self.registry = registry
