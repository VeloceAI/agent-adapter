"""`run_function_calls` across a pending approval: stable ids, siblings run once.

Collected through `test_gemini_contract.py` (the `contract-gemini` session runs that file).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalPending
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations import gemini
from agent_adapter.integrations.gemini import run_function_calls
from agent_adapter.runtime.current import use_context
from tests.fakes.approval import ScriptedApproval
from tests.integrations.kit import FakeRunner


def counting_registry(ran: Counter[str]) -> Registry:
    async def echo(text: str, ctx: ToolContext) -> dict[str, Any]:
        """Echo text with the call id."""
        ran["echo"] += 1
        return {"text": text, "call_id": ctx.call_id, "run_id": ctx.run_id}

    async def remove(item: str) -> str:
        """Remove an item (needs approval)."""
        ran["remove"] += 1
        return f"removed {item}"

    registry = Registry()
    registry.add_many([build_spec(echo), build_spec(remove, confirm=True)])
    return registry


CALLS = [{"name": "echo", "args": {"text": "a"}}, {"name": "remove", "args": {"item": "x"}}]


def results(parts: list[Any]) -> list[Any]:
    return [part.function_response.response for part in parts]


async def test_calls_without_ids_get_stable_distinct_ids():
    ran: Counter[str] = Counter()
    executor = Executor(counting_registry(ran))
    calls = [{"name": "echo", "args": {"text": "a"}}] * 2
    first = [
        r["result"]["call_id"] for r in results(await run_function_calls(calls, executor=executor))
    ]
    again = [
        r["result"]["call_id"] for r in results(await run_function_calls(calls, executor=executor))
    ]
    assert first == again and len(set(first)) == 2
    assert [i.rsplit("_", 1)[1] for i in first] == ["0", "1"]  # occurrences of one call


async def test_a_deferred_approval_round_trip_runs_each_call_once():
    ran: Counter[str] = Counter()
    store = MemoryApprovalStore()
    executor = Executor(counting_registry(ran), approval=DeferredApproval(store))
    ctx = executor.context(run_id="run_gemini_turn")
    with pytest.raises(ApprovalPending) as pending:
        await run_function_calls(CALLS, executor=executor, ctx=ctx)
    assert ran == {"echo": 1}
    await store.decide(pending.value.approval_id, ApprovalDecision(approved=True))
    parts = await run_function_calls(CALLS, executor=executor, ctx=ctx)
    assert ran == {"echo": 1, "remove": 1}
    assert results(parts)[0]["result"]["text"] == "a"
    assert results(parts)[1] == {"result": "removed x"}


async def test_a_plain_runner_keeps_siblings_across_a_pending_approval():
    ran: Counter[str] = Counter()
    approval = ScriptedApproval(
        [ApprovalPending("apr_1", tool="remove"), ApprovalDecision(approved=True)]
    )
    runner, registry = FakeRunner(approval=approval), counting_registry(ran)
    calls = [CALLS[1], CALLS[0]]  # the pending call first: its sibling still runs
    ctx = ToolContext(run_id="run_plain")
    with pytest.raises(ApprovalPending):
        await run_function_calls(calls, executor=runner, registry=registry, ctx=ctx)
    assert ran == {"echo": 1}
    parts = await run_function_calls(calls, executor=runner, registry=registry, ctx=ctx)
    assert ran == {"echo": 1, "remove": 1} and results(parts)[0] == {"result": "removed x"}


async def test_without_a_caller_run_id_nothing_is_kept():
    ran: Counter[str] = Counter()
    executor = Executor(counting_registry(ran), approval=DeferredApproval(MemoryApprovalStore()))
    for _ in range(2):
        with pytest.raises(ApprovalPending):
            await run_function_calls(CALLS, executor=executor)
    assert ran == {"echo": 2}  # each call is a new run: the sibling runs again


async def test_a_kept_response_answers_only_the_same_call():
    ran: Counter[str] = Counter()
    approval = ScriptedApproval([ApprovalPending("apr_1", tool="remove")] * 2)
    runner, registry = FakeRunner(approval=approval), counting_registry(ran)
    ctx = ToolContext(run_id="run_change")
    calls = [{"id": "c1", "name": "echo", "args": {"text": "a"}}, {"id": "c2", **CALLS[1]}]
    with pytest.raises(ApprovalPending):
        await run_function_calls(calls, executor=runner, registry=registry, ctx=ctx)
    changed = [{"id": "c1", "name": "echo", "args": {"text": "b"}}, calls[1]]
    with pytest.raises(ApprovalPending):
        await run_function_calls(changed, executor=runner, registry=registry, ctx=ctx)
    assert ran == {"echo": 2}  # same id, other arguments: run again


def test_the_kept_responses_are_bounded():
    store = gemini._CompletedCalls(limit=2)
    calls = [gemini._function_call({"name": "echo", "args": {"n": n}}, n) for n in range(3)]
    for fc in calls:
        store.keep("run", fc, {"result": fc.args["n"]})
    assert store.take("run", calls[0]) is None
    assert store.take("run", calls[2]) == {"result": 2}
    assert store.take("run", calls[2]) is None  # answered once


async def test_a_reordered_retry_still_matches_the_kept_calls():
    ran: Counter[str] = Counter()
    store = MemoryApprovalStore()
    executor = Executor(counting_registry(ran), approval=DeferredApproval(store))
    ctx = executor.context(run_id="run_reordered")
    with pytest.raises(ApprovalPending) as pending:
        await run_function_calls(CALLS, executor=executor, ctx=ctx)
    await store.decide(pending.value.approval_id, ApprovalDecision(approved=True))
    parts = await run_function_calls(list(reversed(CALLS)), executor=executor, ctx=ctx)
    assert ran == {"echo": 1, "remove": 1}
    assert results(parts)[0] == {"result": "removed x"}


async def test_ids_depend_on_the_call_and_its_occurrence_not_the_position():
    executor = Executor(counting_registry(Counter()))
    a, b = {"name": "echo", "args": {"text": "a"}}, {"name": "echo", "args": {"text": "b"}}

    async def ids(calls: list[dict[str, Any]]) -> list[str]:
        parts = await run_function_calls(calls, executor=executor)
        return [r["result"]["call_id"] for r in results(parts)]

    first, moved = await ids([a, b, a]), await ids([b, a, a])
    assert first[0] != first[2]  # the same call twice: two ids
    assert (first[0], first[1], first[2]) == (moved[1], moved[0], moved[2])


async def test_the_current_context_is_the_base_when_no_ctx_is_given():
    executor = Executor(counting_registry(Counter()))
    with use_context(ToolContext(run_id="run_outer", call_id="call_outer")):
        parts = await run_function_calls(CALLS[:1], executor=executor)
        afc = await gemini.afc_callable(executor.registry.get("echo"), executor=executor)(text="z")
    assert results(parts)[0]["result"]["run_id"] == "run_outer"
    assert afc["run_id"] == "run_outer" and afc["call_id"] not in ("", "call_outer")
