"""The parity cases of `tests/flow/parity.py` on ADK: same pauses, states and visits."""

import pytest

pytest.importorskip("google.adk")

from google.adk.sessions import InMemorySessionService

from agent_adapter.integrations import adk
from tests.flow.parity import CASES, assert_final_matches, assert_pauses_match
from tests.flow.parity_flows import executor

pytestmark = pytest.mark.contract


async def run_adk(parity):
    """Run a case on ADK: `(paused results in order, final result)`."""
    workflow = adk.compile_flow(parity.flow, executor=executor, max_steps=parity.max_steps)
    sessions = InMemorySessionService()
    result = await adk.run_workflow(workflow, parity.initial, session_service=sessions)
    pauses = []
    for resume in (parity.resume, *parity.then) if parity.resume else ():
        assert result.status == "paused", result.status
        pauses.append(result)
        message = adk.resume_message(result.pending.metadata["interrupt_id"], resume.decision)
        result = await adk.run_workflow(workflow, message, session_service=sessions)
    return pauses, result


@pytest.mark.parametrize("parity", CASES, ids=[c.name for c in CASES])
async def test_adk_matches_each_parity_case(parity):
    if parity.raises is not None:
        with pytest.raises(parity.raises):
            await run_adk(parity)
        return
    pauses, final = await run_adk(parity)
    assert_pauses_match(parity, pauses, durable=False)
    assert_final_matches(parity, final)
