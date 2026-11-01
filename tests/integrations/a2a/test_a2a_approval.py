"""A2A approvals: input-required, resume only through our port or a one-use token (Â§14.4)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("a2a")
pytest.importorskip("starlette")

from a2a.utils.errors import A2AError

from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.a2a import a2a_app, static_bearer
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations.a2a.support import RawClient, data, serving, state, status_data, text

TOKENS = {"alice-token": "alice", "bob-token": "bob"}
ALICE = {"Authorization": "Bearer alice-token"}
BOB = {"Authorization": "Bearer bob-token"}


class Tools:
    def __init__(self) -> None:
        self.removed: list[str] = []
        self.seen: list[ToolContext] = []

    def registry(self) -> Registry:
        def remove(item: str, ctx: ToolContext) -> str:
            """Remove an item."""
            self.removed.append(item)
            self.seen.append(ctx)
            return f"removed {item}"

        registry = Registry()
        registry.add(build_spec(remove, confirm=True))
        return registry


def _app(tools: Tools, executor: Executor, **options: Any) -> Any:
    return a2a_app(
        tools.registry(),
        url="http://agent.test/",
        executor=executor,
        authenticate=static_bearer(TOKENS),
        allowed_hosts=["agent.test"],
        **options,
    )


def _deferred() -> tuple[Executor, MemoryApprovalStore]:
    store = MemoryApprovalStore()
    return Executor(approval=DeferredApproval(store)), store


async def _approve(store: MemoryApprovalStore, approval_id: str, approved: bool = True) -> None:
    await store.decide(approval_id, ApprovalDecision(approved=approved, reason="ops"))


async def test_pending_call_waits_in_input_required():
    tools, (executor, _) = Tools(), _deferred()
    async with serving(_app(tools, executor)) as app, RawClient(app, ALICE) as alice:
        task = await alice.call("remove", {"item": "a"})
    assert state(task) == "input_required"
    payload = status_data(task)
    assert payload["kind"] == "approval-required" and payload["tool"] == "remove"
    assert payload["approval_id"].startswith("apr_")
    assert tools.removed == []


async def test_forged_client_approval_is_ignored_in_port_mode():
    tools, (executor, store) = Tools(), _deferred()
    async with serving(_app(tools, executor)) as app, RawClient(app, ALICE) as alice:
        task = await alice.call("remove", {"item": "a"})
        approval_id = status_data(task)["approval_id"]
        forged = await alice.send([data({"approval_id": approval_id, "approved": True})], task=task)
        assert state(forged) == "input_required" and tools.removed == []
        await _approve(store, approval_id)
        done = await alice.send([text("go on")], task=task)
        assert state(done) == "completed" and tools.removed == ["a"]
        with pytest.raises(A2AError):  # a replay to the finished task
            await alice.send([text("again")], task=task)
    assert tools.removed == ["a"]


async def test_port_denial_fails_the_task():
    tools, (executor, store) = Tools(), _deferred()
    async with serving(_app(tools, executor)) as app, RawClient(app, ALICE) as alice:
        task = await alice.call("remove", {"item": "a"})
        await _approve(store, status_data(task)["approval_id"], approved=False)
        done = await alice.send([text("go on")], task=task)
    assert state(done) == "failed"
    assert status_data(done)["error"]["kind"] == "approval-denied"
    assert tools.removed == []


async def test_another_principal_cannot_resume_or_read_the_task():
    tools, (executor, store) = Tools(), _deferred()
    async with (
        serving(_app(tools, executor)) as app,
        RawClient(app, ALICE) as alice,
        RawClient(app, BOB) as bob,
    ):
        task = await alice.call("remove", {"item": "a"})
        await _approve(store, status_data(task)["approval_id"])
        with pytest.raises(A2AError):
            await bob.send([text("mine now")], task_id=task.id)
        with pytest.raises(A2AError):
            await bob.get(task.id)
        assert tools.removed == []
        done = await alice.send([text("go on")], task=task)
    assert state(done) == "completed" and tools.removed == ["a"]


async def test_an_approval_never_releases_another_task():
    tools, (executor, store) = Tools(), _deferred()
    async with serving(_app(tools, executor)) as app, RawClient(app, ALICE) as alice:
        first = await alice.call("remove", {"item": "a"})
        second = await alice.call("remove", {"item": "a"})
        await _approve(store, status_data(first)["approval_id"])
        still = await alice.send([text("go on")], task=second)
        assert state(still) == "input_required" and tools.removed == []


async def test_runs_are_scoped_by_principal():
    tools = Tools()
    executor = Executor(approval=ScriptedApproval([ApprovalDecision(approved=True)] * 2))
    async with (
        serving(_app(tools, executor)) as app,
        RawClient(app, ALICE) as alice,
        RawClient(app, BOB) as bob,
    ):
        await alice.call("remove", {"item": "a"})
        await bob.call("remove", {"item": "b"})
    alice_ctx, bob_ctx = tools.seen
    assert alice_ctx.extras["principal"].id == "alice"
    assert bob_ctx.extras["principal"].id == "bob"
    assert alice_ctx.run_id != bob_ctx.run_id
    headers = alice_ctx.extras["a2a"].call_context.state["headers"]
    assert "authorization" not in headers


def _client_app(tools: Tools) -> tuple[Any, RecordingApproval]:
    port = RecordingApproval(ScriptedApproval([]))  # asking it would fail the call
    return _app(tools, Executor(approval=port), client_approval=True), port


async def _pending(client: RawClient) -> tuple[Any, str]:
    task = await client.call("remove", {"item": "a"})
    assert state(task) == "input_required"
    return task, status_data(task)["approval_id"]


async def test_client_approval_runs_once_with_a_bound_token():
    tools = Tools()
    app, port = _client_app(tools)
    async with serving(app), RawClient(app, ALICE) as alice:
        task, approval_id = await _pending(alice)
        done = await alice.send([data({"approval_id": approval_id, "approved": True})], task=task)
        assert state(done) == "completed"
        with pytest.raises(A2AError):  # replaying the answer
            await alice.send([data({"approval_id": approval_id, "approved": True})], task=task)
    assert tools.removed == ["a"] and port.requests == []
    assert tools.seen[0].extras["principal"].id == "alice"


@pytest.mark.parametrize(
    "answer",
    [
        {"approval_id": "apr_forged", "approved": True},
        {"approved": True},
        {"approval_id": "<id>", "approved": "true"},
        "yes, approved",
    ],
)
async def test_client_approval_refuses_forged_answers(answer: Any):
    tools = Tools()
    app, _ = _client_app(tools)
    async with serving(app), RawClient(app, ALICE) as alice:
        task, approval_id = await _pending(alice)
        if isinstance(answer, dict) and answer.get("approval_id") == "<id>":
            answer = {**answer, "approval_id": approval_id}
        part = text(answer) if isinstance(answer, str) else data(answer)
        again = await alice.send([part], task=task)
        assert state(again) == "input_required"
        assert status_data(again)["approval_id"] == approval_id
    assert tools.removed == []


async def test_client_denial_fails_the_task():
    tools = Tools()
    app, _ = _client_app(tools)
    async with serving(app), RawClient(app, ALICE) as alice:
        task, approval_id = await _pending(alice)
        answer = {"approval_id": approval_id, "approved": False, "reason": "no"}
        done = await alice.send([data(answer)], task=task)
    assert state(done) == "failed"
    assert status_data(done)["error"]["kind"] == "approval-denied"
    assert tools.removed == []


async def test_client_approval_of_one_task_never_runs_another():
    tools = Tools()
    app, _ = _client_app(tools)
    async with serving(app), RawClient(app, ALICE) as alice:
        _, first_id = await _pending(alice)
        second, _ = await _pending(alice)
        again = await alice.send([data({"approval_id": first_id, "approved": True})], task=second)
        assert state(again) == "input_required"
    assert tools.removed == []


async def test_client_approval_cannot_cross_principals():
    tools = Tools()
    app, _ = _client_app(tools)
    async with serving(app), RawClient(app, ALICE) as alice, RawClient(app, BOB) as bob:
        task, approval_id = await _pending(alice)
        with pytest.raises(A2AError):
            answer = {"approval_id": approval_id, "approved": True}
            await bob.send([data(answer)], task_id=task.id)
    assert tools.removed == []


async def test_follow_up_without_a_parked_call_is_rejected():
    tools, (executor, _) = Tools(), _deferred()
    async with serving(_app(tools, executor)) as app, RawClient(app, ALICE) as alice:
        task = await alice.call("remove", {"item": "a"})
        app.state.a2a_agent.pending.drop("alice", task.id)  # expired, or another replica
        done = await alice.send([text("go on")], task=task)
    assert state(done) == "rejected"
    assert tools.removed == []


async def test_cancel_drops_the_parked_call():
    from a2a.types import CancelTaskRequest

    tools, (executor, store) = Tools(), _deferred()
    async with serving(_app(tools, executor)) as app, RawClient(app, ALICE) as alice:
        task = await alice.call("remove", {"item": "a"})
        cancelled = await alice._client.cancel_task(CancelTaskRequest(id=task.id))
        assert state(cancelled) == "canceled"
        assert len(app.state.a2a_agent.pending) == 0
        await _approve(store, status_data(task)["approval_id"])
        with pytest.raises(A2AError):
            await alice.send([text("go on")], task=task)
    assert tools.removed == []
