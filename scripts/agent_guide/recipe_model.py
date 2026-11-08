"""The recipe record and the set-up every recipe snippet runs after."""

from __future__ import annotations

from dataclasses import dataclass

# Shown once above the recipes; every snippet runs after it in a fresh namespace.
PRELUDE = '''import agent_adapter as aa

registry = aa.Registry()


@registry.tool
def get_weather(city: str) -> str:
    """Today's weather for a city."""
    return f"{city}: sunny, 21 C"


@registry.tool(confirm=True)  # a side effect: runs only after an approval
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    return f"cancelled {order_id}"


def my_llm(request: aa.LLMRequest) -> str:  # your model: any SDK, HTTP API or local model
    return "It is sunny in Oslo."


model = aa.CallableModel(my_llm, provider="local", model="my-llm")
executor = aa.Executor(registry)
'''

# Test-only set-up for recipes that take a compression policy as input.
POLICY = (
    "from agent_adapter.context import CompressionPolicy, RemoveOldMessages\n"
    "policy = CompressionPolicy(start_when_tokens_over=100, shrink_to_tokens=50,\n"
    "                           steps=[RemoveOldMessages(keep_last=2)])\n"
)
# Non-blank lines per snippet: most are 1-4; a few need more to stay `ruff format` clean.
MAX_LINES = 10


@dataclass(frozen=True)
class Recipe:
    """One row: what you want, what to use, and a short snippet."""

    want: str
    use: str
    lines: tuple[str, ...]
    requires: tuple[str, ...] = ()  # modules the snippet imports; it runs only when installed
    setup: str = ""  # test-only lines run before the snippet (fakes for its inputs)
    env: tuple[tuple[str, str], ...] = ()  # test-only environment variables
    full_only: bool = False  # AGENTS.md lists it in one line; llms-full.txt has the snippet
    index: str = ""  # a short name: AGENTS.md names it in one shared line, nothing more

    @property
    def concise(self) -> bool:
        """True when AGENTS.md shows the whole snippet."""
        return not (self.full_only or self.index)

    @property
    def code(self) -> str:
        """The snippet as shown in the guide."""
        return "\n".join(self.lines)
