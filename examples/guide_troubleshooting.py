"""Troubleshooting: reproduce common failures and print the error each one gives."""

import asyncio
import logging

from agent_adapter import (
    BudgetHook,
    DeferredApproval,
    Executor,
    MemoryApprovalStore,
    MissingDependency,
    Registry,
    ToolCall,
    run_sync,
)
from agent_adapter.errors import (
    DuplicateTool,
    SchemaNotPortable,
    ToolDefinitionError,
    ToolUsageError,
)
from agent_adapter.integrations.crewai import crewai_tool

registry = Registry()


@registry.tool(confirm=True)
def close_account(account_id: str) -> str:
    """Close an account."""
    return f"closed {account_id}"


@registry.tool(cache_ttl=60)  # cache_scope="run" by default
def rate(currency: str) -> float:
    """Today's EUR rate."""
    return 0.92


@registry.tool
def parse_invoice(text: str) -> dict[str, str]:
    """Parse an invoice."""
    raise KeyError("total")  # a bug in the tool


def show(error: BaseException) -> None:
    """One line: the error's kind (or class) and its message."""
    print(getattr(error, "kind", type(error).__name__), "|", error)


def definitions() -> None:
    # --8<-- [start:define]
    try:
        exec("from agent_adapter import ToolNotFound")  # a name that moved in 1.0
    except ImportError as exc:
        show(exc)
    try:
        registry.wrap(rate)  # the same name twice: a re-run notebook cell, a re-import
    except DuplicateTool as exc:
        show(exc)

    def double(x: int) -> int:
        return 2 * x  # no docstring: the model would get no description

    try:
        registry.wrap(double)
    except ToolDefinitionError as exc:
        show(exc)

    def tally(counts: dict[str, int]) -> int:
        """Sum the counts."""
        return sum(counts.values())

    try:
        registry.wrap(tally)  # an open map: some providers reject it
    except SchemaNotPortable as exc:
        show(exc)
    # --8<-- [end:define]


async def calls() -> None:
    # --8<-- [start:calls]
    executor = Executor(registry)  # no approval port
    for call in (
        ToolCall(name="clos_account", arguments={"account_id": "a1"}),  # misspelt
        ToolCall(name="close_account", arguments={"account_id": 7}),  # wrong type
        ToolCall(name="close_account", arguments={"account_id": "a1"}),  # confirm, no port
        ToolCall(name="parse_invoice", arguments={"text": "..."}),  # the tool crashed
    ):
        result = await executor.run_call(call)
        if result.error is not None:
            print(result.error.kind, "|", result.error.message)
    first = await executor.run_call(ToolCall(name="rate", arguments={"currency": "USD"}))
    again = await executor.run_call(ToolCall(name="rate", arguments={"currency": "USD"}))
    print("cached:", first.cached, again.cached)  # no ctx: each call is a new run
    # --8<-- [end:calls]


async def approvals() -> None:
    # --8<-- [start:approvals]
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    ctx = executor.context(run_id="run_1")
    call = ToolCall(name="close_account", arguments={"account_id": "a1"}, id="c1")
    await executor.run_call(call, ctx)  # pending
    other = executor.context(run_id="run_2")  # resumed under a new run id
    again = await executor.run_call(call, other)
    print(again.status, len(await store.list(status="pending")))  # a second review
    changed = ToolCall(name="close_account", arguments={"account_id": "a2"}, id="c1")
    try:
        await executor.run(changed.name, changed.arguments, ctx, call_id="c1")
    except ToolUsageError as exc:
        show(exc)
    # --8<-- [end:approvals]


async def setup() -> None:
    # --8<-- [start:setup]
    try:
        Executor(registry, hooks=[print])  # type: ignore[list-item]
    except ToolUsageError as exc:
        print(exc.kind, "|", str(exc).split(",")[0])
    no_ledger = Executor(registry, hooks=[BudgetHook(1.0)])  # no ledger to read
    await no_ledger.run("rate", {"currency": "USD"})  # runs; logs a WARNING once
    try:
        crewai_tool(registry.get("rate"))  # the framework is not installed
    except MissingDependency as exc:
        print(exc.kind, "| ImportError:", isinstance(exc, ImportError))
    print(run_sync(no_ledger.run("rate", {"currency": "EUR"}), timeout=5))  # inside a loop
    # --8<-- [end:setup]


async def main() -> None:
    definitions()
    await calls()
    await approvals()
    await setup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
