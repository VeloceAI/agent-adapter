"""Scripts and batch jobs: argparse, run_sync, one batch through run_many, exit codes."""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

from backend_common import PRICES, registry

from agent_adapter import (
    ApprovalRequest,
    AutoDeny,
    CallbackApproval,
    Executor,
    Ledger,
    LoggingHook,
    ToolCall,
    run_sync,
)
from agent_adapter.approval import ApprovalPort


# --8<-- [start:approval]
async def ask_on_terminal(request: ApprovalRequest) -> bool:
    """Ask the person at the terminal; `input` runs in a thread so the loop keeps going."""
    answer = await asyncio.to_thread(input, f"run {request.tool} {request.arguments}? [y/N] ")
    return answer.strip().lower() == "y"


def approval_port(interactive: bool) -> ApprovalPort:
    """Unattended runs deny every confirm tool; `--interactive` asks on the terminal."""
    if interactive:
        return CallbackApproval(ask_on_terminal)
    return AutoDeny(reason="run with --interactive to approve")


# --8<-- [end:approval]


# --8<-- [start:batch]
async def run_batch(executor: Executor, ledger: Ledger, calls: list[ToolCall]) -> int:
    """Run the batch as one run; print one line per call and the cost; 1 if any failed."""
    async with executor.run_scope() as ctx:
        results = await executor.run_many(calls, ctx, concurrency=4)  # failures are collected
    for result in results:
        outcome = result.value if result.ok else (result.error.kind if result.error else "")
        print(f"{result.call_id} {result.status} {outcome}")
    summary = ledger.summary(ctx.run_id)
    print(f"calls={sum(s.calls for s in summary.by_kind.values())} cost=${summary.total_usd}")
    return 0 if all(result.ok for result in results) else 1


# --8<-- [end:batch]


# --8<-- [start:main]
def main(argv: Sequence[str] | None = None) -> int:
    """`python backend_script.py eu us --invoice cus_1024 49.5 [--interactive]`."""
    parser = argparse.ArgumentParser(
        description="Look up shipping rates; optionally send an invoice."
    )
    parser.add_argument("regions", nargs="*", default=["eu"])
    parser.add_argument("--invoice", nargs=2, metavar=("CUSTOMER", "AMOUNT"))
    parser.add_argument("--interactive", action="store_true", help="approve on the terminal")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)  # logs never mix with stdout

    regions = dict.fromkeys(args.regions)  # call ids must be unique within a run
    calls = [ToolCall(name="shipping_rate", arguments={"region": r}, id=r) for r in regions]
    if args.invoice:
        customer_id, amount = args.invoice
        arguments = {"customer_id": customer_id, "amount": amount}
        calls.append(ToolCall(name="send_invoice", arguments=arguments, id="invoice"))
    ledger = Ledger(prices=PRICES)
    executor = Executor(
        registry, approval=approval_port(args.interactive), ledger=ledger, hooks=[LoggingHook()]
    )
    return run_sync(run_batch(executor, ledger, calls))  # also works from a notebook cell


# --8<-- [end:main]


if __name__ == "__main__":
    sys.exit(main())
