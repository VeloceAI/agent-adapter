"""Review fixes: resume needs an explicit, signed decision; resume races; default reach."""

import logging
from typing import Any

import pytest

from agent_adapter import (
    ApprovalDecision,
    ApprovalPending,
    DeferredApproval,
    Executor,
    MemoryApprovalStore,
    Registry,
    tool,
)
from agent_adapter.core._default import default_registry, reset_default_registry
from agent_adapter.errors import ApprovalAlreadyDecided, ToolNotFound, ToolUsageError
from agent_adapter.exec import default_executor

ran: list[str] = []
registry = Registry()


@registry.tool(confirm=True)
def remove(target: str) -> str:
    """Remove a target (needs approval)."""
    ran.append(target)
    return target


@pytest.fixture(autouse=True)
def _clear() -> None:
    ran.clear()


async def parked() -> tuple[Executor, MemoryApprovalStore, str]:
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    with pytest.raises(ApprovalPending) as caught:
        await executor.run("remove", {"target": "a"})
    return executor, store, caught.value.approval_id


async def test_resume_of_a_pending_record_without_a_decision_never_approves() -> None:
    executor, store, approval_id = await parked()
    with pytest.raises(ToolUsageError, match="is still pending: pass the decision"):
        await executor.resume(approval_id)
    assert ran == [] and (await store.get(approval_id)).status == "pending"


@pytest.mark.parametrize("decision", [True, False, ApprovalDecision(approved=True)])
async def test_resume_that_decides_needs_decided_by(decision: Any) -> None:
    executor, store, approval_id = await parked()
    with pytest.raises(ToolUsageError, match="pass decided_by="):
        await executor.resume(approval_id, decision)
    assert ran == [] and (await store.get(approval_id)).status == "pending"


async def test_resume_records_decided_by_and_when() -> None:
    executor, store, approval_id = await parked()
    await executor.resume(approval_id, ApprovalDecision(approved=True), decided_by="ana")
    decision = (await store.get(approval_id)).decision
    assert decision is not None and decision.decided_by == "ana" and decision.decided_at > 0


async def test_resume_rejects_a_decision_of_the_wrong_type() -> None:
    executor, _, approval_id = await parked()
    with pytest.raises(ToolUsageError, match="decision must be True, False"):
        await executor.resume(approval_id, "yes", decided_by="ana")  # type: ignore[arg-type]


async def test_resume_losing_an_agreeing_decide_race_still_runs() -> None:
    executor, store, approval_id = await parked()
    real = store.decide

    async def racer(aid: str, decision: ApprovalDecision) -> Any:
        await real(aid, ApprovalDecision(approved=True, decided_by="bo"))
        return await real(aid, decision)  # the loser: ApprovalAlreadyDecided

    store.decide = racer  # type: ignore[method-assign]
    result = await executor.resume(approval_id, True, decided_by="ana")
    assert result.ok and ran == ["a"]


async def test_resume_losing_a_contradicting_decide_race_is_refused() -> None:
    executor, store, approval_id = await parked()
    real = store.decide

    async def racer(aid: str, decision: ApprovalDecision) -> Any:
        await real(aid, ApprovalDecision(approved=False, decided_by="bo"))
        return await real(aid, decision)

    store.decide = racer  # type: ignore[method-assign]
    with pytest.raises(ApprovalAlreadyDecided):
        await executor.resume(approval_id, True, decided_by="ana")
    assert ran == []


async def test_resume_losing_a_race_to_a_consumer_says_used_once() -> None:
    executor, store, approval_id = await parked()
    real = store.decide

    async def racer(aid: str, decision: ApprovalDecision) -> Any:
        await real(aid, ApprovalDecision(approved=True, decided_by="bo"))
        await store.consume(aid)
        return await real(aid, decision)

    store.decide = racer  # type: ignore[method-assign]
    with pytest.raises(ApprovalAlreadyDecided, match="consumed"):
        await executor.resume(approval_id, True, decided_by="ana")
    assert ran == []


async def test_resume_ends_the_run_it_opened() -> None:
    executor, _, approval_id = await parked()
    ended: list[str] = []
    real = executor.end_run
    executor.end_run = lambda run_id: (ended.append(run_id), real(run_id))[1]  # type: ignore[method-assign]
    await executor.resume(approval_id, True, decided_by="ana")
    assert len(ended) == 1


async def test_run_spec_with_non_mapping_args_is_a_usage_error() -> None:
    spec = registry.get("remove")
    with pytest.raises(ToolUsageError, match="args must be a mapping"):
        await Executor(registry).run_spec(spec, ["a"])  # type: ignore[arg-type]


def test_reset_default_registry_empties_it_and_restores_it() -> None:
    @tool
    def fx_probe() -> str:
        """A probe tool."""
        return "x"

    try:
        restore = reset_default_registry()
        assert "fx_probe" not in default_registry
        with pytest.raises(ToolNotFound):
            Executor().run_sync("fx_probe")
        restore()
        assert "fx_probe" in default_registry
    finally:
        default_registry.remove("fx_probe")


def test_isolated_defaults_fixture_hides_every_default_tool(isolated_defaults: None) -> None:
    assert len(default_registry) == 0
    assert len(default_executor().registry) == 0


def test_executor_on_the_default_registry_says_so_at_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="agent_adapter.exec.executor")
    Executor()
    assert any("process-wide default registry" in r.getMessage() for r in caplog.records)


def test_reset_default_executor_makes_a_fresh_one() -> None:
    from agent_adapter.exec.defaults import reset_default_executor, set_default_executor

    mine = Executor(registry)
    set_default_executor(mine)
    reset_default_executor()
    try:
        assert default_executor() is not mine
    finally:
        reset_default_executor()
