"""DX sugar for human approval: executor.resume, DeferredApproval.sqlite, fix-it errors."""

import os
from pathlib import Path

import pytest

from agent_adapter import (
    ApprovalDecision,
    ApprovalPending,
    AutoApprove,
    DeferredApproval,
    Executor,
    MemoryApprovalStore,
    Registry,
    SQLiteApprovalStore,
    ToolCall,
)
from agent_adapter.errors import (
    ApprovalAlreadyDecided,
    ApprovalDenied,
    ApprovalNotFound,
    ToolFailed,
    ToolUsageError,
)

registry = Registry()
paid: list[float] = []


@registry.tool(confirm=True)
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    paid.append(amount)
    return f"refunded {amount} on {order_id}"


@registry.tool
def boom() -> str:
    """Always fails."""
    raise ValueError("secret detail")


@pytest.fixture(autouse=True)
def _no_payments() -> None:
    paid.clear()


def deferred() -> tuple[Executor, MemoryApprovalStore]:
    store = MemoryApprovalStore()
    return Executor(registry, approval=DeferredApproval(store)), store


async def pending_id(executor: Executor, amount: float = 500) -> str:
    with pytest.raises(ApprovalPending) as caught:
        await executor.run("refund", {"order_id": "o-7", "amount": amount})
    return caught.value.approval_id


async def test_resume_approves_and_runs_the_parked_call_once() -> None:
    executor, store = deferred()
    approval_id = await pending_id(executor)
    result = await executor.resume(approval_id, True, decided_by="ana")
    assert (result.ok, result.value) == (True, "refunded 500.0 on o-7")
    record = await store.get(approval_id)
    assert record.consumed and record.decision is not None
    assert record.decision.decided_by == "ana"
    assert result.call_id == record.request.call_id
    assert paid == [500.0]


async def test_resume_twice_is_refused_and_runs_nothing_more() -> None:
    executor, _ = deferred()
    approval_id = await pending_id(executor)
    await executor.resume(approval_id, True, decided_by="ana")
    with pytest.raises(ApprovalAlreadyDecided) as caught:
        await executor.resume(approval_id)
    assert str(caught.value) == f"approval {approval_id!r} is already consumed"
    assert caught.value.__notes__ == [
        "a decision runs its call once; to run the tool again, make a new call "
        "(it asks for a new approval)"
    ]
    assert paid == [500.0]


async def test_resume_with_a_denial_never_runs_the_tool() -> None:
    executor, store = deferred()
    approval_id = await pending_id(executor)
    result = await executor.resume(approval_id, False, decided_by="ana", reason="over the limit")
    assert result.status == "error" and result.error is not None
    assert result.error.kind == "approval-denied"
    assert isinstance(result.exception, ApprovalDenied)
    assert result.exception.reason == "over the limit"
    assert (await store.get(approval_id)).consumed
    assert paid == []


async def test_resume_with_edited_arguments_runs_the_edit() -> None:
    executor, _ = deferred()
    approval_id = await pending_id(executor)
    edit = ApprovalDecision(approved=True, edited_args={"order_id": "o-7", "amount": 450})
    result = await executor.resume(approval_id, edit, decided_by="ana")
    assert result.value == "refunded 450.0 on o-7"


async def test_resume_uses_a_decision_already_in_the_store() -> None:
    executor, store = deferred()
    approval_id = await pending_id(executor)
    await store.decide(approval_id, ApprovalDecision(approved=False, reason="no"))
    result = await executor.resume(approval_id)  # no decision given: the stored one applies
    assert result.error is not None and result.error.kind == "approval-denied"
    assert paid == []


async def test_resume_refuses_a_decision_that_contradicts_the_store() -> None:
    executor, store = deferred()
    approval_id = await pending_id(executor)
    await store.decide(approval_id, ApprovalDecision(approved=False))
    with pytest.raises(ApprovalAlreadyDecided, match="is already denied"):
        await executor.resume(approval_id, True)
    same = await executor.resume(approval_id, False)  # agreeing with the store is fine
    assert same.error is not None and same.error.kind == "approval-denied"


async def test_resume_of_an_unknown_id_is_not_found() -> None:
    executor, _ = deferred()
    with pytest.raises(ApprovalNotFound):
        await executor.resume("apr_nope")


async def test_resume_without_a_store_says_how_to_configure_one() -> None:
    executor = Executor(registry, approval=AutoApprove())
    with pytest.raises(ToolUsageError) as caught:
        await executor.resume("apr_x")
    assert str(caught.value) == (
        "resume needs an approval port with a store, got AutoApprove: "
        "Executor(approval=DeferredApproval(store))"
    )


async def test_resume_with_a_context_for_another_run_is_refused() -> None:
    executor, store = deferred()
    approval_id = await pending_id(executor)
    with pytest.raises(ToolUsageError, match="belongs to run 'run_"):
        await executor.resume(
            approval_id, True, executor.context(run_id="run_other"), decided_by="ana"
        )
    assert paid == []
    assert (await store.get(approval_id)).status == "pending"  # refused before deciding


async def test_resume_keeps_a_given_context_of_the_same_run() -> None:
    executor, store = deferred()
    ctx = executor.context(run_id="run_keep", state={"user": "u-1"})
    call = ToolCall(name="refund", arguments={"order_id": "o-1", "amount": 10}, id="call_k")
    pending = await executor.run_call(call, ctx)
    approval_id = pending.error.details["approval_id"] if pending.error else ""
    result = await executor.resume(approval_id, True, ctx, decided_by="ana")
    assert (result.ok, result.call_id) == (True, "call_k")
    assert (await store.get(approval_id)).request.run_id == "run_keep"


async def test_resume_of_a_flow_node_points_at_the_flow_runner() -> None:
    executor, store = deferred()
    from agent_adapter.core.types import ApprovalRequest

    node = ApprovalRequest(
        approval_id="", run_id="r", call_id="c", tool="node:f.n", arguments={}, kind="node"
    )
    record = await store.create(node)
    with pytest.raises(ToolUsageError) as caught:
        await executor.resume(record.approval_id)
    assert str(caught.value) == (
        f"approval {record.approval_id!r} pauses flow node 'node:f.n', not a tool call: "
        "resume it with FlowRunner.resume"
    )


def test_resume_sync_is_the_sync_twin() -> None:
    executor, _ = deferred()
    with pytest.raises(ApprovalPending) as caught:
        executor.run_sync("refund", {"order_id": "o-2", "amount": 9})
    result = executor.resume_sync(caught.value.approval_id, True, decided_by="bo", timeout=5)
    assert result.value == "refunded 9.0 on o-2"


async def test_a_resume_across_processes_needs_only_the_approval_id(tmp_path: Path) -> None:
    path = tmp_path / "approvals.db"
    worker = Executor(registry, approval=DeferredApproval.sqlite(path))
    approval_id = await pending_id(worker, amount=70)
    reviewer = Executor(registry, approval=DeferredApproval.sqlite(path))  # a new process
    result = await reviewer.resume(approval_id, True, decided_by="ana")
    assert result.value == "refunded 70.0 on o-7"


def test_deferred_approval_sqlite_builds_a_sqlite_store(tmp_path: Path) -> None:
    port = DeferredApproval.sqlite(tmp_path / "a.db", max_records=5, timeout=2)
    assert isinstance(port, DeferredApproval)
    assert isinstance(port.store, SQLiteApprovalStore)
    assert (port.store.path, port.store.max_records, port.store.timeout) == (
        os.fspath(tmp_path / "a.db"),
        5,
        2.0,
    )


@pytest.mark.parametrize("decision", [True, False, "yes", None])
async def test_store_decide_with_a_bool_says_how(decision: object) -> None:
    executor, store = deferred()
    approval_id = await pending_id(executor)
    with pytest.raises(ToolUsageError) as caught:
        await store.decide(approval_id, decision)  # type: ignore[arg-type]
    assert str(caught.value) == (
        f"decide needs an ApprovalDecision, got {type(decision).__name__}: "
        f"store.decide(id, ApprovalDecision(approved={bool(decision)!r})), "
        "or executor.resume(id, True, decided_by='you') to decide and run in one call"
    )
    assert (await store.get(approval_id)).status == "pending"


async def test_store_decide_with_a_bool_says_how_in_sqlite(tmp_path: Path) -> None:
    store = SQLiteApprovalStore(tmp_path / "s.db")
    executor = Executor(registry, approval=DeferredApproval(store))
    approval_id = await pending_id(executor)
    with pytest.raises(ToolUsageError, match="decide needs an ApprovalDecision, got bool"):
        await store.decide(approval_id, True)  # type: ignore[arg-type]


async def test_a_pending_error_says_how_to_resume() -> None:
    executor, _ = deferred()
    with pytest.raises(ApprovalPending) as caught:
        await executor.run("refund", {"order_id": "o-7", "amount": 1})
    assert caught.value.__notes__ == [
        "decide and run it with executor.resume(approval_id, True, decided_by='you'), or "
        "decide it with "
        "store.decide(approval_id, ApprovalDecision(...)) and then call executor.resume("
        "approval_id); the approval id is on this error's .approval_id"
    ]
    assert "resume" not in str(caught.value)  # the model-visible message is unchanged


async def test_no_approval_port_says_how_to_add_one() -> None:
    with pytest.raises(ApprovalDenied) as caught:
        await Executor(registry).run("refund", {"order_id": "o", "amount": 1})
    assert str(caught.value) == "approval denied for tool 'refund': no approval port configured"
    assert caught.value.__notes__ == [
        "pass approval= to Executor (or agent_adapter.configure): AutoApprove() for tests, "
        "DeferredApproval(store) for reviews, CallbackApproval(fn) to ask a function"
    ]


async def test_a_failed_tool_says_where_its_error_went() -> None:
    with pytest.raises(ToolFailed) as caught:
        await Executor(registry).run("boom")
    assert str(caught.value) == "tool 'boom' failed after 1 attempt"
    assert caught.value.__notes__ == [
        "cause: ValueError (its text is hidden from the model and logged at DEBUG by "
        "agent_adapter.exec.pipeline); raise ToolError('...') in the tool to show a "
        "message to the model"
    ]
    assert "secret detail" not in str(caught.value)


def test_a_duplicate_tool_says_how_to_avoid_it() -> None:
    mine = Registry()
    mine.wrap(lambda: 1, name="one", description="One.")
    from agent_adapter.errors import DuplicateTool

    with pytest.raises(DuplicateTool) as caught:
        mine.wrap(lambda: 2, name="one", description="Two.")
    assert caught.value.__notes__ == [
        "pass name= to register it under another name, use a separate Registry(), or "
        "registry.remove(name) first (e.g. a notebook cell that runs twice)"
    ]


def test_an_untyped_parameter_says_how_to_type_it() -> None:
    from typing import Any

    from agent_adapter.errors import SchemaNotPortable

    def loose(payload: Any) -> None:
        """Take anything."""

    with pytest.raises(SchemaNotPortable) as caught:
        Registry().wrap(loose)
    assert "untyped value (Any)" in str(caught.value)
    assert caught.value.__notes__ == [
        "give each flagged field a concrete type (str, int, list[str], a TypedDict or a "
        "pydantic model), or pass portable=False when every target provider accepts it"
    ]
