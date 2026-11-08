"""Approval memory: past decisions as advice, a proposed rule, and a person who accepts it.

A shop's refunds need approval. After two reviewers approve five small refunds for damaged
goods, the next one still asks; its approval shows what was decided before. `propose_rules`
then suggests a rule bounded to what the reviewers saw (10 <= amount <= 40, reason
"damaged"). It changes nothing until a person calls `accept` and commits the code.
"""

import asyncio

from agent_adapter import ApprovalPending, DeferredApproval, Executor, MemoryApprovalStore, Registry
from agent_adapter.approval import accept, approval_context, propose_rules

registry = Registry()


def refund(amount: int, reason: str) -> str:
    """Refund an order (needs approval)."""
    return f"refunded {amount} ({reason})"


registry.tool(confirm=True)(refund)


async def ask(executor: Executor, amount: int, reason: str = "damaged") -> str:
    """Run a refund; return its approval id when it waits for a person."""
    try:
        await executor.run("refund", {"amount": amount, "reason": reason})
    except ApprovalPending as pending:
        return pending.approval_id
    return ""


async def main() -> None:
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    for n, amount in enumerate((10, 20, 25, 30, 40)):  # two reviewers approve five refunds
        approval_id = await ask(executor, amount)
        await executor.resume(approval_id, True, decided_by="ann" if n % 2 else "bob")

    # --8<-- [start:context]
    waiting = await ask(executor, 30)  # still asks: memory never approves
    record = await store.get(waiting)
    context = approval_context(record.request, await store.list())
    print(record.status, "-", context.summary())  # pending - 5 similar: 5 approved, ...
    # --8<-- [end:context]

    # --8<-- [start:rules]
    (rule,) = propose_rules(await store.list(), min_count=5, min_deciders=2)
    print("proposed:", rule.describe())  # data only: nothing changed yet
    accepted = accept(rule, accepted_by="ann")  # a person's explicit choice
    print(accepted.code)  # commit this; it builds the same confirm= rule
    # --8<-- [end:rules]

    # --8<-- [start:apply]
    reviewed = Registry()
    reviewed.tool(confirm=accepted.confirm)(refund)  # the rule a person accepted
    checked = Executor(reviewed, approval=DeferredApproval(store))
    print(await checked.run("refund", {"amount": 15, "reason": "damaged"}))  # inside the rule
    for amount, reason in ((90, "damaged"), (-5, "damaged"), (15, "changed my mind")):
        print("asks again:", bool(await ask(checked, amount, reason)))  # outside: a person
    # --8<-- [end:apply]


if __name__ == "__main__":
    asyncio.run(main())
