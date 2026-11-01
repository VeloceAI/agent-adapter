"""Typed errors: catch them narrowly; send argument errors back to the model."""

import asyncio
import json

from agent_adapter import AdapterError, Executor, Registry
from agent_adapter.errors import (
    SchemaNotPortable,
    ToolArgsInvalid,
    ToolDefinitionError,
    ToolNotFound,
)

registry = Registry()


@registry.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def tag_counts(counts: dict[str, int]) -> int:
    """Total of all tag counts."""
    return sum(counts.values())


def no_docstring(x: int) -> int:
    return x


async def main() -> None:
    # --8<-- [start:args]
    try:
        await Executor(registry).run("add", {"a": 1, "b": "two", "c": 3})
    except ToolArgsInvalid as exc:
        # JSON-safe and free of input values: safe to log or return to the model.
        reply = {"tool": exc.tool, "errors": exc.errors}
        print(json.dumps(reply))
    # --8<-- [end:args]

    # --8<-- [start:startup]
    try:
        registry.wrap(tag_counts)  # dict[str, int] is an open map
    except SchemaNotPortable as exc:
        print(exc.tool, exc.problems)

    try:
        registry.wrap(no_docstring)  # no description anywhere
    except ToolDefinitionError as exc:
        print(exc)
    # --8<-- [end:startup]

    # --8<-- [start:base]
    try:
        registry.get("subtract")
    except ToolNotFound as exc:
        print(exc)
    except AdapterError:  # every adapter error shares this base
        raise
    # --8<-- [end:base]


if __name__ == "__main__":
    asyncio.run(main())
