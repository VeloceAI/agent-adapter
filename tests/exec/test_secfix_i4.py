"""I4: edited arguments pass the same before_tool checks (guardrails, budget) as the originals."""

from typing import Any

import pytest

from agent_adapter import ApprovalDecision, Hook, Registry, ToolCall, ToolContext
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import BudgetExceeded
from agent_adapter.core.types import LedgerEntry
from agent_adapter.errors import GuardrailBlocked
from agent_adapter.exec import Executor
from agent_adapter.hooks import BudgetHook, GuardrailHook, ShortCircuit
from agent_adapter.hooks.base import ToolInvocation
from tests.fakes.approval import ScriptedApproval

registry = Registry()
sent: list[int] = []


@registry.tool(confirm=True)
async def wire(amount: int) -> str:
    """Wire money."""
    sent.append(amount)
    return f"sent {amount}"


def _at_most_100(stage: str, payload: Any) -> bool | str:
    return payload["amount"] <= 100 or "amount over 100"


GUARD = GuardrailHook(_at_most_100)


class Seen(Hook):
    """Records the arguments every before_tool call sees."""

    def __init__(self, short: dict[str, Any] | None = None) -> None:
        self.args: list[dict[str, Any]] = []
        self.after: list[dict[str, Any]] = []
        self.short = short

    async def before_tool(self, inv: ToolInvocation) -> ShortCircuit | None:
        self.args.append(dict(inv.call.arguments))
        if self.short is not None and inv.call.arguments == self.short:
            return ShortCircuit("stubbed")
        return None

    async def after_tool(self, inv: ToolInvocation, result: Any) -> None:
        self.after.append(dict(inv.call.arguments))


def _edit(amount: int) -> ApprovalDecision:
    return ApprovalDecision(approved=True, edited_args={"amount": amount})


@pytest.fixture(autouse=True)
def _reset() -> None:
    sent.clear()


async def test_secfix_i4_guardrail_blocks_the_original_arguments() -> None:
    ex = Executor(registry, hooks=[GUARD], approval=ScriptedApproval([]))
    with pytest.raises(GuardrailBlocked):
        await ex.run("wire", {"amount": 10**6})
    assert sent == []


async def test_secfix_i4_port_edit_is_checked_by_the_guardrail() -> None:
    ex = Executor(registry, hooks=[GUARD], approval=ScriptedApproval([_edit(10**6)]))
    with pytest.raises(GuardrailBlocked, match="amount over 100"):
        await ex.run("wire", {"amount": 10})
    assert sent == []


async def test_secfix_i4_pre_approval_edit_is_checked_by_the_guardrail() -> None:
    ex = Executor(registry, hooks=[GUARD])
    ctx = ToolContext(run_id="r", call_id="c1", approval=_edit(10**6))
    result = await ex.run_call(ToolCall(name="wire", arguments={"amount": 10}, id="c1"), ctx)
    assert result.error is not None and result.error.kind == "guardrail-blocked"
    assert sent == []


async def test_secfix_i4_stored_decision_edit_is_checked_on_resume() -> None:
    store = MemoryApprovalStore()
    ex = Executor(registry, hooks=[GUARD], approval=DeferredApproval(store))
    ctx = ex.context(run_id="r")
    call = ToolCall(name="wire", arguments={"amount": 10}, id="c1")
    first = await ex.run_call(call, ctx)
    assert first.status == "pending" and first.error is not None
    await store.decide(first.error.details["approval_id"], _edit(10**6))
    resumed = await ex.run_call(call, ctx)
    assert resumed.error is not None and resumed.error.kind == "guardrail-blocked"
    assert sent == []


async def test_secfix_i4_budget_is_checked_again_for_an_edit() -> None:
    class SpentLedger:
        def __init__(self) -> None:
            self.spent = 0.0

        def record(self, entry: LedgerEntry) -> None:
            return None

        def total_usd(self, run_id: str | None = None, *, thread_id: str | None = None) -> float:
            return self.spent

        def entries(
            self, run_id: str | None = None, *, thread_id: str | None = None
        ) -> list[LedgerEntry]:
            return []

    ledger = SpentLedger()

    class SpendWhileAsking:
        async def request(self, request: Any) -> ApprovalDecision:
            ledger.spent = 5.0  # the budget runs out while the human decides
            return _edit(2)

    ex = Executor(registry, hooks=[BudgetHook(1.0)], approval=SpendWhileAsking(), ledger=ledger)
    with pytest.raises(BudgetExceeded):
        await ex.run("wire", {"amount": 1})
    assert sent == []


async def test_secfix_i4_hooks_see_the_edit_then_after_tool_sees_what_ran() -> None:
    seen = Seen()
    ex = Executor(registry, hooks=[seen, GUARD], approval=ScriptedApproval([_edit(50)]))
    assert await ex.run("wire", {"amount": 10}) == "sent 50"
    assert seen.args == [{"amount": 10}, {"amount": 50}]
    assert seen.after == [{"amount": 50}] and sent == [50]


async def test_secfix_i4_an_unchanged_edit_runs_the_hooks_once() -> None:
    seen = Seen()
    ex = Executor(registry, hooks=[seen], approval=ScriptedApproval([_edit(10)]))
    assert await ex.run("wire", {"amount": 10}) == "sent 10"
    assert seen.args == [{"amount": 10}]


async def test_secfix_i4_a_short_circuit_on_the_edit_skips_the_tool() -> None:
    seen = Seen(short={"amount": 7})
    ex = Executor(registry, hooks=[seen], approval=ScriptedApproval([_edit(7)]))
    result = await ex.run_call(ToolCall(name="wire", arguments={"amount": 10}))
    assert result.ok and result.value == "stubbed" and result.attempts == 0
    assert sent == []
