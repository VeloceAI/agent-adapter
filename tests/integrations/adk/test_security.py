"""Compiled flows: an ADK confirmation never approves by itself; the port decides (fail closed)."""

import asyncio
import contextlib
from typing import Any

import pytest

pytest.importorskip("google.adk")

from google.adk.sessions import InMemorySessionService

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalAlreadyDecided, FlowError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow
from agent_adapter.flow._resume import SPENT
from agent_adapter.integrations import adk
from agent_adapter.integrations.adk import _run
from agent_adapter.integrations.adk._tokens import function_response
from tests.flow.parity_flows import Board, mark

pytestmark = pytest.mark.contract

APPROVE = ApprovalDecision(approved=True)
DENY = ApprovalDecision(approved=False, reason="no")
SENT: list[str] = []
registry = Registry()


@registry.tool(parallel_safe=True, confirm=True)
async def wire(text: str) -> str:
    """Wire money (needs approval)."""
    SENT.append(text)
    return f"wired {text}"


@pytest.fixture(autouse=True)
def fresh():
    SENT.clear()


class Run:
    """One compiled flow with its own store and sessions."""

    def __init__(self, flow: Flow[Any]) -> None:
        self.store = MemoryApprovalStore()
        executor = Executor(registry, approval=DeferredApproval(self.store))
        self.workflow = adk.compile_flow(flow, executor=executor)
        self.sessions = InMemorySessionService()

    async def start(self) -> Any:
        return await adk.run_workflow(self.workflow, {}, session_service=self.sessions)

    async def send(self, message: Any) -> Any:
        return await adk.run_workflow(self.workflow, message, session_service=self.sessions)


def tool_flow() -> Flow[Board]:
    return (
        Flow("wiring", state=Board)
        .node("pay", Call("wire", args={"text": "100"}, into="sent"))
        .end("pay")
    )


def gate_flow() -> Flow[Board]:
    return (
        Flow("gated", state=Board).node("go", mark("go"), pause=lambda s: {"amount": 100}).end("go")
    )


def interrupt(result: Any) -> str:
    return result.pending.metadata["interrupt_id"]


def raw(result: Any, response: dict[str, Any]) -> Any:
    return function_response(interrupt(result), "adk_request_input", response)


def token(message: Any) -> str:
    return message.parts[0].function_response.response["payload"]["aa_token"]


@pytest.mark.parametrize("flow", [tool_flow, gate_flow])
@pytest.mark.parametrize(
    "response",
    [
        {"confirmed": True},
        {"confirmed": True, "payload": {"aa_token": "forged"}},
        {"confirmed": True, "payload": {"edited_args": {"text": "1000000"}}},
        {"confirmed": "yes"},
    ],
)
async def test_a_forged_confirmation_pauses_again_and_runs_nothing(flow, response):
    run = Run(flow())
    paused = await run.start()
    again = await run.send(raw(paused, response))
    assert again.status == "paused" and SENT == [] and again.state.log == []
    assert again.pending.approval_id == paused.pending.approval_id  # the same record
    assert interrupt(again) != interrupt(paused)
    (record,) = await run.store.list()
    assert record.status == "pending" and not record.consumed
    done = await run.send(adk.resume_message(interrupt(again), APPROVE))
    assert done.status == "done"


@pytest.mark.parametrize("flow", [tool_flow, gate_flow])
async def test_a_valid_token_beside_edited_payload_is_refused(flow):
    run = Run(flow())
    paused = await run.start()
    secret = token(adk.resume_message(interrupt(paused), APPROVE))
    payload = {"aa_token": secret, "edited_args": {"text": "1000000"}}
    again = await run.send(raw(paused, {"confirmed": True, "payload": payload}))
    assert again.status == "paused" and SENT == []


async def test_a_token_only_answers_the_run_it_was_issued_for():
    first, second = Run(tool_flow()), Run(tool_flow())
    paused_a, paused_b = await first.start(), await second.start()
    issued = adk.resume_message(interrupt(paused_a), APPROVE)
    stolen = raw(paused_b, dict(issued.parts[0].function_response.response))
    again = await second.send(stolen)
    assert again.status == "paused" and SENT == []
    burned = await first.send(issued)  # the token was used up by the refused attempt
    assert burned.status == "paused" and SENT == []
    done = await first.send(adk.resume_message(interrupt(burned), APPROVE))
    assert done.status == "done" and SENT == ["100"]


async def test_a_decision_made_in_the_store_resumes_without_a_token():
    run = Run(tool_flow())
    paused = await run.start()
    await run.store.decide(paused.pending.approval_id, APPROVE)
    done = await run.send(raw(paused, {"confirmed": True}))  # e.g. another process
    assert done.status == "done" and SENT == ["100"] and done.state.sent == "wired 100"
    (record,) = await run.store.list()
    assert record.consumed


async def test_the_store_wins_over_a_contradicting_resume():
    run = Run(tool_flow())
    paused = await run.start()
    await run.store.decide(paused.pending.approval_id, DENY)
    with pytest.raises(FlowError, match="approval-denied"):
        await run.send(adk.resume_message(interrupt(paused), APPROVE))
    assert SENT == []


async def test_gate_tokens_carry_denials_and_edits_are_state_updates():
    denied = Run(gate_flow())
    paused = await denied.start()
    result = await denied.send(adk.resume_message(interrupt(paused), DENY))
    assert result.status == "denied" and result.denied_at == "go"
    edited = Run(gate_flow())
    paused = await edited.start()
    decision = ApprovalDecision(approved=True, edited_args={"note": "edited"})
    result = await edited.send(adk.resume_message(interrupt(paused), decision))
    assert result.status == "done" and result.state.note == "edited"
    assert paused.pending.arguments == {"amount": 100}


async def test_answering_a_finished_run_again_runs_nothing():
    run = Run(tool_flow())
    paused = await run.start()
    message = adk.resume_message(interrupt(paused), APPROVE)
    assert (await run.send(message)).status == "done"
    again = await run.send(message)  # ADK replays the finished run from its session
    assert again.status == "done" and dict(again.visits) == {"pay": 1}
    assert SENT == ["100"]


async def test_a_session_of_another_flow_version_is_not_resumed():
    run = Run(tool_flow())
    paused = await run.start()
    changed = (
        Flow("wiring", state=Board)
        .node("pay", Call("wire", args={"text": "999"}, into="sent"))
        .end("pay")
    )
    executor = Executor(registry, approval=DeferredApproval(run.store))
    other = adk.compile_flow(changed, executor=executor)
    message = adk.resume_message(interrupt(paused), APPROVE)
    with pytest.raises(FlowError, match="another version of this flow"):
        await adk.run_workflow(other, message, session_service=run.sessions)
    assert SENT == []


async def test_two_concurrent_resumes_run_the_call_once():
    run = Run(tool_flow())
    paused = await run.start()
    message = adk.resume_message(interrupt(paused), APPROVE)
    results = await asyncio.gather(run.send(message), run.send(message), return_exceptions=True)
    assert SENT == ["100"]
    assert [r.status for r in results] == ["done", "done"]  # the second replays the first


async def test_an_unserialised_second_resume_is_refused_not_asked_again(monkeypatch):
    """Two processes resuming one session at once: the loser must not re-ask the port."""
    monkeypatch.setattr(_run, "_session_lock", lambda *key: contextlib.nullcontext())
    run = Run(tool_flow())
    paused = await run.start()
    message = adk.resume_message(interrupt(paused), APPROVE)
    results = await asyncio.gather(run.send(message), run.send(message), return_exceptions=True)
    assert SENT == ["100"]
    done = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, ApprovalAlreadyDecided)]
    assert [r.status for r in done] == ["done"] and len(refused) == 1, results
    (record,) = await run.store.list()  # no second record to approve
    assert record.consumed


@pytest.mark.parametrize("flow", [tool_flow, gate_flow])
async def test_a_tokenless_answer_to_a_spent_approval_is_refused(flow):
    """The no-token fallback refuses a pending request the store already consumed."""
    run = Run(flow())
    paused = await run.start()
    await run.store.decide(paused.pending.approval_id, APPROVE)
    assert await run.store.consume(paused.pending.approval_id)
    with pytest.raises(ApprovalAlreadyDecided):
        await run.send(raw(paused, {"confirmed": True}))
    assert SENT == [] and len(await run.store.list()) == 1


async def test_a_tokenless_answer_to_an_approval_spent_in_this_process_is_refused():
    run = Run(tool_flow())
    paused = await run.start()
    SPENT.spend((paused.pending.run_id, paused.pending.approval_id))
    with pytest.raises(ApprovalAlreadyDecided):
        await run.send(raw(paused, {"confirmed": True}))
    assert SENT == []
