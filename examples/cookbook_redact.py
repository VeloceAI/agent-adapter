"""Cookbook: keep secrets and personal data out of logs, and out of what the model sees."""

import asyncio
import dataclasses
import logging
import re
from typing import Any

from agent_adapter import Executor, Hook, LoggingHook, Registry, ToolResult
from agent_adapter.hooks import DEFAULT_REDACT, ToolInvocation, redact

registry = Registry()
CARD = re.compile(r"\b(?:\d{4}-){3}(\d{4})\b")  # a card number: 4242-4242-4242-4242


# --8<-- [start:tools]
@registry.tool
def lookup_customer(email: str, api_key: str) -> dict[str, str]:
    """Look up a customer's billing profile."""
    card = "4242-4242-4242-4242"
    return {"email": email, "card": card, "note": f"card {card} on file"}


# --8<-- [end:tools]


# --8<-- [start:hook]
class MaskCard(Hook):
    """Masks card numbers in tool results before the model or the caller sees them."""

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> ToolResult | None:
        if not result.ok:
            return None  # keep failures as they are
        return dataclasses.replace(result, value=_mask(result.value))


def _mask(value: Any) -> Any:
    if isinstance(value, str):
        return CARD.sub(r"****-****-****-\1", value)
    if isinstance(value, dict):
        return {key: _mask(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask(item) for item in value]
    return value


# --8<-- [end:hook]


class Lines(logging.Handler):
    """Collects formatted log lines, so the example can print them."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


async def main() -> None:
    # --8<-- [start:executor]
    log = logging.getLogger("app.tools")
    patterns = (*DEFAULT_REDACT, "email", "card")  # key names to mask, case-insensitive
    executor = Executor(registry, hooks=[LoggingHook(logger=log, redact=patterns), MaskCard()])
    # --8<-- [end:executor]
    lines = Lines()
    log.addHandler(lines)
    log.setLevel(logging.INFO)
    log.propagate = False
    # --8<-- [start:run]
    value = await executor.run("lookup_customer", {"email": "a@example.com", "api_key": "sk-123"})
    print(value)  # the model sees masked card numbers
    print(lines.lines[0])  # tool.start: argument values masked by key name
    print(redact({"Authorization": "Bearer x", "query": "returns"}))  # for your own log lines
    # --8<-- [end:run]


if __name__ == "__main__":
    asyncio.run(main())
