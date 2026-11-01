"""Two processes resume one paused session without a token: no orphan record, one run."""

import copy
from typing import Any

import pytest

pytest.importorskip("google.adk")

from agent_adapter.core.errors import ApprovalAlreadyDecided
from agent_adapter.flow.visit import SPENT
from agent_adapter.integrations import adk
from agent_adapter.integrations.adk import _steps
from tests.integrations.adk.test_security import APPROVE, SENT, Run, gate_flow, raw, tool_flow

pytestmark = pytest.mark.contract


async def _stale_check(request: Any, port: Any) -> None:
    """Process B checked the record before process A consumed it (the race window)."""


@pytest.fixture(autouse=True)
def fresh():
    SENT.clear()
    SPENT.clear()
    yield
    SPENT.clear()


@pytest.mark.parametrize("flow", [tool_flow, gate_flow])
async def test_the_loser_of_an_unserialised_resume_leaves_no_orphan_record(flow, monkeypatch):
    run = Run(flow())
    paused = await run.start()
    process_b = copy.deepcopy(run.sessions)  # B's view of the session, taken before A resumed
    await run.store.decide(paused.pending.approval_id, APPROVE)  # decided out of band
    answer = raw(paused, {"confirmed": True})

    done = await run.send(answer)  # process A
    assert done.status == "done"

    SPENT.clear()  # process B has its own in-process tables
    monkeypatch.setattr(_steps, "_refuse_spent", _stale_check)
    with pytest.raises(ApprovalAlreadyDecided):
        await adk.run_workflow(run.workflow, answer, session_service=process_b)

    (record,) = await run.store.list()  # no second, orphan pending record
    assert record.consumed
    sent = ["100"] if flow is tool_flow else []
    assert sent == SENT
